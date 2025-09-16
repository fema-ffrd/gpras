"""Tools to wrangle HEC-RAS data into a format usable by the gaussian process regression model."""

import os
import pickle
import re
from datetime import datetime
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Self, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely
from hecdss import HecDss
from numpy.typing import NDArray
from pyproj import CRS
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely import Polygon
from sklearn.decomposition import PCA, IncrementalPCA

from gpras.ras.model import RasModel
from gpras.utils.plotting import ts_clipping
from gpras.utils.spatial_utils import ras_hdf_precip_transform

HecDss.set_global_debug_level(0)
HydraulicParameterType = Literal["wse", "depth", "velocity"]

DB_PATHS = {
    "hf": "hf_model.parquet",
    "lf": "lf_model.parquet",
    "cell_info": "cell_info.geoparquet",
    "ref_lines": "ref_lines.parquet",
}


class DataBuilder:
    """Convenience class for extracting data from RAS models and aligning high and low fidelity model data."""

    REFERENCE_LINE_NAME_PATH: str = (
        "/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Reference Lines/Name"
    )
    REFERENCE_LINE_FLOW_PATH: str = (
        "/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Reference Lines/Flow"
    )
    REFERENCE_LINE_WSE_PATH: str = (
        "/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Reference Lines/Water Surface"
    )
    BOUNDARY_CONDITION_PATH: str = (
        "/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Boundary Conditions/{bc_id}"
    )
    UNSTEADY_TIME_INDEX_PATH: str = (
        "/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp"
    )

    def __init__(
        self,
        hf_ras: RasModel,
        mesh_id: str,
        plans: list[str],
        area_of_interest: Polygon,
        cell_id_field: str = "cell_id",
        flow_convergence_threshold: float = 0.95,
        cutoffs: dict[str, tuple[int, int]] | None = None,
        hf_resampler: NDArray[Any] | None = None,
    ):
        """Construct class."""
        self.hf_ras = hf_ras
        self.mesh_id = mesh_id
        self.plans = plans
        self.area_of_interest = area_of_interest
        self.cell_id_field = cell_id_field
        self.flow_convergence_threshold = flow_convergence_threshold
        self.hf_resampler = hf_resampler
        self.set_spatial_resamplers()  # Does not overwrite
        self.cutoffs = cutoffs or {}
        self._hf_aligned: pd.DataFrame | None = None
        self._lf_aligned: pd.DataFrame | None = None

    def _align_datasets(self, plot_dir: str | None = None) -> None:
        hf_store = []
        lf_store = []
        # Need to do this incrementally to save RAM.
        for p in self.plans:
            # Load
            hf_data = self.get_hf_plan_data(p)
            lf_data = self.get_lf_plan_data(p)
            combo_df = pd.concat([hf_data, lf_data], axis=1)

            # Temporally subset
            if p not in self.cutoffs:
                self.cutoffs[p] = self.get_cutoff(combo_df.values)
                if plot_dir is not None:
                    self._plot_cutoff_diagnostic(combo_df.values, self.cutoffs[p], str(Path(plot_dir) / f"{p}.png"))
            cutoff = self.cutoffs[p]
            dur = cutoff[1] - cutoff[0]
            ts = np.arange(0, dur)

            # Format
            index = pd.MultiIndex.from_arrays([[p] * dur, ts], names=["run", "t"])
            hf_df = pd.DataFrame(hf_data.values[cutoff[0] : cutoff[1], :], columns=hf_data.columns, index=index)
            hf_store.append(hf_df)
            lf_df = pd.DataFrame(lf_data.values[cutoff[0] : cutoff[1], :], columns=lf_data.columns, index=index)
            lf_store.append(lf_df)

        self._hf_aligned = pd.concat(hf_store)
        self._lf_aligned = pd.concat(lf_store)

    @cached_property
    def aligned_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Trim WSE timeseries spatially and temporally."""
        if self._hf_aligned is None or self._lf_aligned is None:
            self._align_datasets()
        return self._hf_aligned, self._lf_aligned

    @cached_property
    def aligned_ref_line_df(self) -> pd.DataFrame:
        """Dataframe of WSE and flow for all reference lines."""
        store = []
        for p in self.plans:
            ref_data = self.get_ref_line_df(p)
            cutoff = self.cutoffs[p]
            store.append(ref_data.iloc[cutoff[0] : cutoff[1]].copy())
        return pd.concat(store)

    def get_cutoff(self, combo: NDArray[Any]) -> tuple[int, int]:
        """Determine when the model is 95% done changing and filter out warmup."""
        dx_dt = self._delta_cols_norm(combo)
        dx_dt = np.sum(dx_dt, axis=1) / np.sum(dx_dt)
        cum_dx_dt = np.cumsum(dx_dt)

        stop = cast(int, np.argmax(cum_dx_dt > self.flow_convergence_threshold))
        start = cast(int, np.argmax(cum_dx_dt > 10e-4))
        return (start, stop)

    def _delta_cols_norm(self, arr: NDArray[Any]) -> NDArray[Any]:
        """Normalized changes in feature or cell WSE across timesteps."""
        dx_dt = np.abs(np.diff(arr, axis=0))
        normalizer = np.sum(dx_dt, axis=0)
        normalizer[normalizer == 0] = 1
        dx_dt /= normalizer
        return cast(NDArray[Any], dx_dt)

    def _plot_cutoff_diagnostic(self, arr: NDArray[Any], cutoffs: tuple[int, int], out_path: str) -> None:
        """QC plot for temporal clipping."""
        Path(out_path).parent.mkdir(exist_ok=True, parents=True)
        dx_dt = self._delta_cols_norm(arr)
        ts_clipping(dx_dt, cutoffs, out_path)

    def get_hf_plan_data(self, plan: str) -> pd.DataFrame:
        """Get HF water surface elevation timeseries within an AOI from a HEC-RAS model."""
        dt_index = self.get_unsteady_timeseries_index(plan)
        asset = self.hf_ras.plan_hdfs[plan]
        vals: NDArray[Any] = asset.mesh_timeseries_output(self.mesh_id, "Water Surface").values
        vals = vals[:, self.hf_resampler]
        return pd.DataFrame(vals, index=dt_index, columns=self.hf_resampler)

    def get_lf_plan_data(self, _: str) -> pd.DataFrame:
        """Placeholder for subclass methods."""
        raise RuntimeError("Tried to call get_lf_plan_data() on DataBuilder. Use a subclass instead.")

    def set_spatial_resamplers(self) -> None:
        """Set the index arrays that are used to resample LF to HF."""
        if self.hf_resampler is None:
            self.hf_resampler = self.hf_geometry_aoi[self.cell_id_field].values

    def export_db(self, out_path: str) -> None:
        """Export hf and lf dataframes as well as cell area and cell elevation to parquet files."""
        out_path_ = Path(out_path)
        out_path_.mkdir(parents=True, exist_ok=True)
        hf_data_df, lf_data_df = self.aligned_datasets
        hf_data_df.to_parquet(out_path_ / DB_PATHS["hf"])
        self.cell_info_df.to_parquet(out_path_ / DB_PATHS["cell_info"], index=False)
        lf_data_df.to_parquet(out_path_ / DB_PATHS["lf"])
        self.aligned_ref_line_df.to_parquet(out_path_ / DB_PATHS["ref_lines"])

    @cached_property
    def cell_info_df(self) -> gpd.GeoDataFrame:
        """A dataframe with elevations and areas of the model cells within the area of interest."""
        return gpd.GeoDataFrame(
            {
                "hf_cell_id": self.hf_resampler,
                "elevation": self.cell_elevations,
                "area": self.cell_areas,
                "geometry": self.hf_geometry_aoi.geometry,
            }
        )

    @cached_property
    def _hf_geometry_full(self) -> gpd.GeoDataFrame:
        return self._get_geometry_full(self.hf_ras)

    def _get_geometry_full(self, model: RasModel) -> gpd.GeoDataFrame:
        return model.get_plan_geometry(self.plans, self.mesh_id)

    @cached_property
    def hf_geometry_aoi(self) -> gpd.GeoDataFrame:
        """Geometry for the high-fidelity model within the area of interest."""
        return self._hf_geometry_full[self._hf_mask].copy()

    @cached_property
    def _hf_mask(self) -> NDArray[Any]:
        return self._get_spatial_mask(self._hf_geometry_full)

    def _get_spatial_mask(self, geom: gpd.GeoDataFrame) -> NDArray[Any]:
        return cast(NDArray[Any], geom.intersects(self.area_of_interest).values)

    @cached_property
    def cell_areas(self) -> NDArray[Any]:
        """Area of cells within the area of interest."""
        return self.hf_ras.get_cell_areas(self.plans[0], self.mesh_id)[self.hf_resampler]

    @cached_property
    def cell_elevations(self) -> NDArray[Any]:
        """Elevation of cells within the area of interest."""
        return self.hf_ras.get_cell_minimum_elevation(self.plans[0], self.mesh_id)[self.hf_resampler]

    @cached_property
    def _rasterized_aoi(self) -> NDArray[Any]:
        """AOI for use in filtering precipitation rasters."""
        precip_meta = self.hf_ras.get_precip_attributes(self.plans[0])
        transform = ras_hdf_precip_transform(precip_meta)
        crs = CRS(precip_meta["Projection"])
        shapes = [(geom, 1) for geom in self.hf_geometry_aoi.geometry.to_crs(crs)]
        mask = rasterize(
            shapes,
            out_shape=(precip_meta["Raster Rows"], precip_meta["Raster Cols"]),
            transform=transform,
            fill=0,  # outside polygon
            all_touched=True,
            dtype="uint8",
        ).astype(bool)
        return cast(NDArray[Any], mask)

    def _export_rasterized_aoi(self) -> None:
        """Use to debug rasterized aoi."""
        precip_meta = self.hf_ras.get_precip_attributes(self.plans[0])
        transform = ras_hdf_precip_transform(precip_meta)
        crs = CRS(precip_meta["Projection"])
        with rasterio.open(
            "test_mask_export.tif",
            "w",
            driver="GTiff",
            height=precip_meta["Raster Rows"],
            width=precip_meta["Raster Cols"],
            count=1,
            dtype=np.int32,
            crs=crs,
            transform=transform,
            nodata=0,
        ) as dst:
            dst.write(self._rasterized_aoi.astype(np.int32), 1)

    def _export_precip_gtiff(self, data: NDArray[Any]) -> None:
        """Use to debug rasterized aoi."""
        precip_meta = self.hf_ras.get_precip_attributes(self.plans[0])
        transform = ras_hdf_precip_transform(precip_meta)
        crs = CRS(precip_meta["Projection"])
        with rasterio.open(
            "test_precip_export.tif",
            "w",
            driver="GTiff",
            height=precip_meta["Raster Rows"],
            width=precip_meta["Raster Cols"],
            count=1,
            dtype=np.float32,
            crs=crs,
            transform=transform,
            nodata=-9999,
        ) as dst:
            dst.write(data.astype(np.float32), 1)

    def get_precip_ts(self, plan: str) -> pd.DataFrame:
        """Get the excess precipitation values for all cells touching the area of interest across timesteps."""
        asset = self.hf_ras.plan_hdfs[plan]
        timesteps = asset.get("/Event Conditions/Meteorology/Precipitation/Timestamp")[:].astype(str)
        dt_index = pd.to_datetime(timesteps, format="%d%b%Y %H:%M:%S.%f")
        data = asset.get("/Event Conditions/Meteorology/Precipitation/Values")
        mask = self._rasterized_aoi
        shape = (data.shape[0], mask.shape[0], mask.shape[1])
        vals = np.reshape(data, shape)[:, mask]
        return pd.DataFrame(vals, index=dt_index, columns=[f"precip_{i}" for i in range(vals.shape[1])])

    def get_ref_line_df(self, plan: str) -> pd.DataFrame:
        """Get reference line WSE and flow across timesteps."""
        asset = self.hf_ras.plan_hdfs[plan]
        dt_index = self.get_unsteady_timeseries_index(plan)
        names = asset.get(self.REFERENCE_LINE_NAME_PATH)[:]
        flows = asset.get(self.REFERENCE_LINE_FLOW_PATH)[:]
        wse = asset.get(self.REFERENCE_LINE_WSE_PATH)[:]
        flows_df = pd.DataFrame(flows, index=dt_index, columns=[i.decode() + "_flows" for i in names])
        wse_df = pd.DataFrame(wse, index=dt_index, columns=[i.decode() + "_wse" for i in names])
        return pd.concat([flows_df, wse_df], axis=1)

    def get_bc_ts(self, plan: str, bc_id: str) -> pd.DataFrame:
        """Get bounary condition discharge across timesteps."""
        asset = self.hf_ras.plan_hdfs[plan]
        dt_index = self.get_unsteady_timeseries_index(plan)
        vals = asset.get(self.BOUNDARY_CONDITION_PATH.format(bc_id=bc_id))[:, 1]
        return pd.DataFrame(vals, index=dt_index, columns=[bc_id])

    def get_unsteady_timeseries_index(self, plan: str) -> pd.Series:
        """Get an index for all unsteady timeseries."""
        asset = self.hf_ras.plan_hdfs[plan]
        timesteps = asset.get(self.UNSTEADY_TIME_INDEX_PATH)[:].astype(str)
        return pd.to_datetime(timesteps, format="%d%b%Y %H:%M:%S")


class RasUpskillDataBuilder(DataBuilder):
    """Used to build datasets for upskilling low-fidelity HEC-RAS to high-fidelity HEC-RAS."""

    def __init__(
        self,
        hf_ras: RasModel,
        lf_ras: RasModel,
        mesh_id: str,
        plans: list[str],
        area_of_interest: Polygon,
        cell_id_field: str = "cell_id",
        flow_convergence_threshold: float = 0.95,
        cutoffs: dict[str, tuple[int, int]] | None = None,
        hf_resampler: NDArray[Any] | None = None,
        lf_resampler: NDArray[Any] | None = None,
    ):
        """Construct class."""
        self.lf_ras = lf_ras
        super().__init__(
            hf_ras=hf_ras,
            mesh_id=mesh_id,
            plans=plans,
            area_of_interest=area_of_interest,
            cell_id_field=cell_id_field,
            flow_convergence_threshold=flow_convergence_threshold,
            cutoffs=cutoffs,
            hf_resampler=hf_resampler,
        )
        if hf_resampler is None or lf_resampler is None:
            self.set_spatial_resamplers()
        else:
            self.hf_resampler = hf_resampler
            self.lf_resampler = lf_resampler

    def get_lf_plan_data(self, plan: str) -> pd.DataFrame:
        """Get water surface elevation timeseries from a HEC-RAS model."""
        dt_index = self.get_lf_unsteady_timeseries_index(plan)
        asset = self.lf_ras.plan_hdfs[plan]
        vals: NDArray[Any] = asset.mesh_timeseries_output(self.mesh_id, "Water Surface").values
        vals = vals[:, self.lf_resampler]
        mask = vals < self.cell_elevations
        vals[mask] = np.repeat(self.cell_elevations[:, np.newaxis], vals.shape[0], axis=1).T[mask]
        return pd.DataFrame(vals, index=dt_index, columns=self.hf_resampler)

    def get_lf_unsteady_timeseries_index(self, plan: str) -> pd.Series:
        """Get an index for all unsteady timeseries."""
        asset = self.lf_ras.plan_hdfs[plan]
        timesteps = asset.get(self.UNSTEADY_TIME_INDEX_PATH)[:].astype(str)
        return pd.to_datetime(timesteps, format="%d%b%Y %H:%M:%S")

    def set_spatial_resamplers(self) -> None:
        """Set the index arrays that are used to resample LF to HF."""
        hf_geom = self.hf_geometry_aoi
        lf_geom = self.lf_geometry_aoi

        mesh_resampled = gpd.overlay(
            hf_geom, lf_geom[[self.cell_id_field, "geometry"]], how="intersection", keep_geom_type=True
        )
        mesh_resampled["area"] = mesh_resampled.geometry.area
        mesh_resampled = mesh_resampled.sort_values(by="area")
        mesh_resampled = mesh_resampled.drop_duplicates(subset=f"{self.cell_id_field}_1", keep="last")
        mesh_resampled = mesh_resampled[[f"{self.cell_id_field}_1", f"{self.cell_id_field}_2"]]
        order = hf_geom[self.cell_id_field]
        mesh_resampled = mesh_resampled.set_index(f"{self.cell_id_field}_1").loc[order].reset_index()

        self.hf_resampler = mesh_resampled[f"{self.cell_id_field}_1"].values
        self.lf_resampler = mesh_resampled[f"{self.cell_id_field}_2"].values

    @cached_property
    def _lf_geometry_full(self) -> gpd.GeoDataFrame:
        return self._get_geometry_full(self.lf_ras)

    def _get_geometry_full(self, model: RasModel) -> gpd.GeoDataFrame:
        return model.get_plan_geometry(self.plans, self.mesh_id)

    @cached_property
    def lf_geometry_aoi(self) -> gpd.GeoDataFrame:
        """Geometry for the low-fidelity model within the area of interest."""
        return self._lf_geometry_full[self._lf_mask].copy()

    @cached_property
    def _lf_mask(self) -> NDArray[Any]:
        return self._get_spatial_mask(self._lf_geometry_full)

    def _get_spatial_mask(self, geom: gpd.GeoDataFrame) -> NDArray[Any]:
        return cast(NDArray[Any], geom.intersects(self.area_of_interest).values)

    @cached_property
    def cell_info_df(self) -> pd.DataFrame:
        """A dataframe with elevations and areas of the model cells within the area of interest."""
        df = super().cell_info_df
        df["lf_cell_id"] = self.lf_resampler
        return df


class HmsUpskillDataBuilder(DataBuilder):
    """Used to build datasets for upskilling precip and inflow timeseries to high-fidelity HEC-RAS."""

    def __init__(
        self,
        hf_ras: RasModel,
        inflow_dss_dir: str,
        inflow_hms_elements: list[str],
        precip_dss_dir: str,
        precip_spatial_mode_count: int,
        mesh_id: str,
        plans: list[str],
        area_of_interest: Polygon,
        cell_id_field: str = "cell_id",
        flow_convergence_threshold: float = 0.95,
        cutoffs: dict[str, tuple[int, int]] | None = None,
        hf_resampler: NDArray[Any] | None = None,
    ):
        """Construct class."""
        super().__init__(
            hf_ras, mesh_id, plans, area_of_interest, cell_id_field, flow_convergence_threshold, cutoffs, hf_resampler
        )

        self.inflow_dss_dir = inflow_dss_dir
        self.inflow_hms_elements = inflow_hms_elements
        self.precip_dss_dir = precip_dss_dir
        self.precip_spatial_mode_count = precip_spatial_mode_count

    def get_lf_plan_data(self, plan: str) -> pd.DataFrame:
        """Get boundary condition features for the HF model."""
        all_cols = []
        for bc in self.inflow_hms_elements:
            all_cols.append(self.get_hms_inflow_ts(plan, bc))
        all_cols.append(self.get_hms_precip_ts(plan))
        return pd.concat(all_cols, axis=1).fillna(0)

    def get_hms_inflow_ts(self, plan: str, bc_id: str) -> pd.DataFrame:
        """Get the outflow from a HEC-HMS element."""
        dss = HecDss(str(Path(self.inflow_dss_dir) / f"{plan}.dss"))
        dss_path = [str(i) for i in dss.get_catalog() if bc_id[0] == i.B and bc_id[1] == i.C][0]
        data = dss.get(dss_path)
        return pd.DataFrame(data.values, index=data.times, columns=[f"{bc_id[0]}_{bc_id[1]}"])

    def get_hms_precip_ts(self, plan: str) -> pd.DataFrame:
        """Geet a HEC-HMS excess precip grid timeseries."""
        # Load DSS
        dss = HecDss(str(Path(self.precip_dss_dir) / f"{plan}.dss"))

        # Process
        ts = []
        dt_index = []
        for i in dss.get_catalog():
            t = re.search(r"\d{2}[A-Za-z]{3}\d{4}:\d{4}", str(i))
            if not t:
                raise ValueError(f"Could not parse datetime from DSS catalog entry: {i}")
            dt_index.append(datetime.strptime(t.group(), "%d%b%Y:%H%M"))
            record = dss.get(str(i))
            data = np.flipud(record.data)
            ts.append(data[self._aoi_precip_mask])
        vals = np.array(ts)
        return pd.DataFrame(vals, index=dt_index, columns=[f"precip_{i}" for i in range(vals.shape[1])])

    # def compare_p(self, plan) -> None:  # type: ignore
    #     """For debugging, add to git then deprecate."""
    #     asset = self.hf_ras.plan_hdfs[plan]
    #     ras = asset.get("/Event Conditions/Meteorology/Precipitation/Values")[:]
    #     ras_mask = self._rasterized_aoi
    #     shape = (ras.shape[0], ras_mask.shape[0], ras_mask.shape[1])
    #     ras = np.reshape(ras, shape)

    #     dss = HecDss(str(Path(self.precip_dss_dir) / f"{plan}.dss"))
    #     ts = []
    #     dt_index = []
    #     for i in dss.get_catalog():
    #         t = re.search(r"\d{2}[A-Za-z]{3}\d{4}:\d{4}", str(i))
    #         if not t:
    #             raise ValueError(f"Could not parse datetime from DSS catalog entry: {i}")
    #         dt_index.append(datetime.strptime(t.group(), "%d%b%Y:%H%M"))
    #         record = dss.get(str(i))
    #         ts.append(record.data)
    #     dt_index, ts = zip(*sorted(zip(dt_index, ts, strict=False)), strict=False)
    #     hms_mask = self._aoi_precip_mask
    #     hms = np.array(ts)
    #     hms_flip = np.array([np.flipud(i) for i in ts])

    #     hms_mean = hms_flip[:, hms_mask].mean(axis=1)
    #     ras_mean = ras[:, ras_mask].mean(axis=1)

    #     hms_mean = hms_flip[:, hms_mask].mean(axis=1)
    #     ras_mean = ras[:, ras_mask].mean(axis=1)

    #     import imageio
    #     import matplotlib.pyplot as plt

    #     fig, ax = plt.subplots()
    #     ax.plot(hms_mean, label="hms")
    #     ax.plot(ras_mean, label="ras")
    #     ax.legend()
    #     fig.tight_layout()
    #     fig.savefig("precip_ts.png")
    #     plt.close(fig)

    #     fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
    #     t = 2
    #     ras_img = ras.copy()
    #     ras_img[ras_img == 0] = np.nan
    #     hms_img = hms.copy()
    #     hms_img[hms_img == 0] = np.nan
    #     axs[0, 0].imshow(ras_img.reshape(hms.shape[0], hms.shape[1], hms.shape[2], order="C")[t])
    #     axs[0, 1].imshow(np.flipud(hms_img[t]))
    #     axs[1, 0].imshow(ras_mask.reshape(hms.shape[1], hms.shape[2], order="C"))
    #     axs[1, 1].imshow(hms_mask)
    #     fig.tight_layout()
    #     fig.savefig("check_precip.png")
    #     plt.close(fig)

    #     frames = []
    #     for t in range(hms.shape[0]):  # iterate over time axis
    #         fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))

    #         ras_img = ras.copy()
    #         ras_img[ras_img == 0] = np.nan
    #         hms_img = hms.copy()
    #         hms_img[hms_img == 0] = np.nan

    #         axs[0, 0].imshow(ras_img.reshape(hms.shape[0], hms.shape[1], hms.shape[2], order="C")[t])
    #         axs[0, 1].imshow(np.flipud(hms_img[t]))
    #         axs[1, 0].imshow(ras_mask.reshape(hms.shape[1], hms.shape[2], order="C"))
    #         axs[1, 1].imshow(hms_mask)

    #         fig.tight_layout()

    #         # save this frame to a buffer
    #         fig.canvas.draw()
    #         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    #         image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #         frames.append(image)

    #         plt.close(fig)

    #     # save all frames as a gif
    #     imageio.mimsave("check_precip.gif", frames, fps=3)  # adjust fps as needed

    @cached_property
    def _aoi_precip_mask(self) -> NDArray[Any]:
        """Precipitation array mask for area of interest (applies to HMS dss file data)."""
        # Load DSS
        plan = self.plans[0]
        dss = HecDss(str(Path(self.precip_dss_dir) / f"{plan}.dss"))

        # Get AOI mask
        catalog = dss.get_catalog()
        record_template = dss.get(next(iter(catalog)))

        # Make transform
        pixel_size = record_template.cellSize
        height = record_template.numberOfCellsY

        upper_left_x = (record_template.lowerLeftCellX) * pixel_size
        upper_left_y = (record_template.lowerLeftCellY + height) * pixel_size
        transform = from_origin(upper_left_x, upper_left_y, pixel_size, pixel_size)

        # Get mask
        src = pyproj.CRS(self.hf_geometry_aoi.crs)
        dest = pyproj.CRS(record_template.srsDefinition)
        project = pyproj.Transformer.from_crs(src, dest, always_xy=True).transform
        shape = shapely.ops.transform(project, self.area_of_interest)
        shapes = [(shape, 1)]

        mask = rasterize(
            shapes,
            out_shape=(record_template.numberOfCellsY, record_template.numberOfCellsX),
            transform=transform,
            fill=0,  # outside polygon
            all_touched=True,
            dtype="uint8",
        ).astype(bool)

        return mask


class PseudoSurfaceDataBuilder(DataBuilder):
    """Used to build datasets for upskilling a predicted WSE surface to high-fidelity HEC-RAS."""

    def __init__(
        self,
        hf_ras: RasModel,
        mesh_id: str,
        plans: list[str],
        area_of_interest: Polygon,
        cell_id_field: str = "cell_id",
        flow_convergence_threshold: float = 0.95,
        cutoffs: dict[str, tuple[int, int]] | None = None,
        hf_resampler: NDArray[Any] | None = None,
    ):
        """Construct class."""
        super().__init__(
            hf_ras, mesh_id, plans, area_of_interest, cell_id_field, flow_convergence_threshold, cutoffs, hf_resampler
        )


class RasReader:
    """Lightweight equivalent to RasExtracter that get data from database instead of models."""

    def __init__(self, db_path: str):
        """Construct class."""
        self.db_path = Path(db_path)

    @property
    def aligned_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Trim WSE timeseries spatially and temporally."""
        hf_df = pd.read_parquet(self.db_path / DB_PATHS["hf"])
        lf_df = pd.read_parquet(self.db_path / DB_PATHS["lf"])
        return hf_df, lf_df

    @property
    def hf_resampler(self) -> NDArray[Any]:
        """Cell IDs from the HF model within the AOI."""
        return cast(NDArray[Any], self._cell_info["hf_cell_id"].values)

    @property
    def lf_resampler(self) -> NDArray[Any]:
        """Cell IDs from the LF model resampled to match HF model cells within the AOI."""
        return cast(NDArray[Any], self._cell_info["lf_cell_id"].values)

    @property
    def cell_elevations(self) -> NDArray[Any]:
        """Elevation of cells within the area of interest."""
        return cast(NDArray[Any], self._cell_info["elevation"].values)

    @property
    def cell_areas(self) -> NDArray[Any]:
        """Area of cells within the area of interest."""
        return cast(NDArray[Any], self._cell_info["area"].values)

    @cached_property
    def _cell_info(self) -> pd.DataFrame:
        """Read cell info table."""
        # Read as GeoDataFrame to match GeoParquet; numeric columns still accessible
        return gpd.read_parquet(self.db_path / DB_PATHS["cell_info"])

    @cached_property
    def hf_geometry_aoi(self) -> gpd.GeoDataFrame:
        """Geometry for the high-fidelity model within the area of interest."""
        gdf = self._cell_info.copy()
        gdf["cell_id"] = gdf["hf_cell_id"]  # TODO: figure out a way to not hard code cell_id
        return gdf

    @staticmethod
    def is_valid(db_path: str) -> bool:
        """Check if a db file has all required tables."""
        if not os.path.exists(db_path):
            return False
        files = os.listdir(db_path)
        needed = DB_PATHS.values()
        return all(i in files for i in needed)


class PreProcessor:
    """Class to transform HEC-RAS data for use in upskilling low-fidelity (lf) models to high-fidelity (hf)."""

    def __init__(
        self,
        spatial_mode_count: int = 0,
        input_mean: NDArray[Any] | None = None,
        wet_threshold: float = 0.03,
        elevations: NDArray[Any] | None = None,
        hydraulic_parameter: HydraulicParameterType = "wse",
        wetness_classes: NDArray[Any] | None = None,
        weights: NDArray[Any] | None = None,
        eofs: NDArray[Any] | None = None,
        eigenvalues: NDArray[Any] | None = None,
        n_samples_fit: float = 0,
        x_mean: NDArray[Any] | None = None,
        x_std: NDArray[Any] | None = None,
    ):
        """Preprocessor class constructor.

        When default values are used (None, for most arguments), their values will be set during the fit() method.

        Args:
            spatial_mode_count (int): Number of spatial modes for PCA. Defaults to 0.
            input_mean (NDArray[Any] | None, optional): Mean values for centering input data. Defaults to None.
            wet_threshold (float, optional): Threshold for determining whether a cell gets wet. Defaults to 0.03.
            elevations (NDArray[Any] | None, optional): Elevation values for the cells. Defaults to None.
            hydraulic_parameter (HydraulicParameterType, optional): Treat inputs as WSE, depth, or velocity. Defaults to WSE.
            wetness_classes (NDArray[Any] | None, optional): Wetness classification of cells. Defaults to None.
            weights (NDArray[Any] | None, optional): Weighting factors for the cells (typically cell area). Defaults to None.
            eofs (NDArray[Any] | None, optional): Empirical Orthogonal Functions (EOFs) from PCA. Defaults to None.
            eigenvalues (NDArray[Any] | None, optional): Eigenvalues from PCA. Defaults to None.
            n_samples_fit (float, optional): Number of samples used during PCA fitting. Defaults to 0.
            x_mean (NDArray[Any] | None, optional): Mean of spatial modes. Defaults to None.
            x_std (NDArray[Any] | None, optional): Standard deviation of spatial modes. Defaults to None.

        Returns:
            None
        """
        self.spatial_mode_count: int = spatial_mode_count

        self.input_mean: NDArray[Any] = input_mean if input_mean is not None else np.empty(0, dtype=float)

        self.wet_threshold = wet_threshold
        self.elevations: NDArray[Any] = elevations if elevations is not None else np.empty(0, dtype=float)
        self.hydraulic_parameter = hydraulic_parameter
        self.wetness_classes: NDArray[np.str_] = (
            wetness_classes if wetness_classes is not None else np.empty(0, dtype=np.str_)
        )

        self.weights: NDArray[Any] = weights if weights is not None else np.empty(0, dtype=float)

        self.eofs: NDArray[Any] = eofs if eofs is not None else np.empty(0, dtype=float)
        self.eigenvalues: NDArray[Any] = eigenvalues if eigenvalues is not None else np.empty(0, dtype=float)
        self.n_samples_fit = n_samples_fit

        self.x_mean: NDArray[Any] = x_mean if x_mean is not None else np.empty(0, dtype=float)
        self.x_std: NDArray[Any] = x_std if x_std is not None else np.empty(0, dtype=float)

    @property
    def dry_indices(self) -> NDArray[np.bool_]:
        """Identify cells that are always dry.

        Returns:
            NDArray[np.bool_]: Boolean array indicating dry cells.
        """
        if self.wetness_classes is None:
            raise ValueError("wetness_classes must be numpy array to access dry_indices")
        return np.equal(self.wetness_classes, "AD")

    @property
    def eof(self) -> NDArray[Any]:
        """Get the Empirical Orthogonal Functions (EOFs).

        Returns:
            NDArray[Any]: Array of EOFs.
        """
        if self.eofs is None:
            raise ValueError("EOFs have not been computed")
        return self.eofs

    def fit(
        self,
        x: NDArray[Any],
        elevations: NDArray[Any],
        weights: NDArray[Any] | None = None,
        spatial_mode_count: int | None = None,
    ) -> None:
        """Fit the preprocessor to the input data using PCA.

        Filters out always-dry cells, centers the data, optionally applies weights,
        and fits a PCA model. Determines number of modes using North's Rule, if not set.

        Args:
            x (NDArray[Any]): Array of shape (samples, cells) representing water surface elevations.
            elevations (NDArray[Any]): Elevation values for each cell.
            weights (NDArray[Any]): Optional weighting array for cells (e.g., cell area).
            spatial_mode_count (int): Optional number of spatial modes to use.  Otherwise uses North's rule.

        Returns:
            None
        """
        # Filter cells that are always dry or always wet
        self.elevations = elevations
        if self.hydraulic_parameter == "depth":
            x = self.wse_2_depth(x)
            self.wetness_classes = self.classify_wetness_depth(x)
        elif self.hydraulic_parameter == "wse":
            self.wetness_classes = self.classify_wetness_wse(x, elevations)
        x = x[:, ~self.dry_indices]

        # Apply first round of scaling
        self.input_mean = x.mean(axis=0)
        x = x - self.input_mean

        # Weight by cell area (or other)
        if weights is not None:
            self.weights = weights[~self.dry_indices]
            x *= self.weights

        # Fit PCA
        pca = IncrementalPCA()  # Documentation says that the function can batch itself
        pca.fit(x)

        # Reduce modes
        if spatial_mode_count is None:
            self.spatial_mode_count = compute_norths_rule(pca)
            # TODO: Consider these methods https://stats.stackexchange.com/questions/33917/how-to-determine-significant-principal-components-using-bootstrapping-or-monte-c
        else:
            self.spatial_mode_count = spatial_mode_count

        # Set results
        self.eofs = pca.components_[: self.spatial_mode_count]
        self.eigenvalues = pca.explained_variance_
        self.n_samples_fit = pca.n_samples_seen_

        # Set second round of standardization
        x = np.dot(x, self.eofs.T)
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0)

    def transform(self, x: NDArray[Any]) -> NDArray[Any]:
        """Transform new input data using the fitted PCA model.

        Applies centering, weighting, and projects onto retained EOFs.

        Args:
            x (NDArray[Any]): Array of shape (samples, cells) to be transformed.

        Returns:
            NDArray[Any]: Array of transformed data in EOF space (samples, spatial_mode_count).
        """
        # Filter cells that are always dry or always wet
        if self.hydraulic_parameter == "depth":
            x = self.wse_2_depth(x)
        x = x[:, ~self.dry_indices].copy()

        # Apply first round of scaling
        x = x - self.input_mean

        # Weight by cell area (or other)
        if self.weights is not None:
            x *= self.weights

        # Apply PCA
        x = np.dot(x, self.eofs.T)

        # Standardize
        x = (x - self.x_mean) / self.x_std

        return x

    def wse_2_depth(self, x: NDArray[Any]) -> NDArray[Any]:
        """Convert water surface elevation data to depths."""
        d: NDArray[Any] = x - self.elevations
        d[d < 0] = 0
        return d

    def reverse_transform(
        self, mean: NDArray[Any], var: NDArray[Any] | None = None
    ) -> NDArray[Any] | tuple[NDArray[Any], NDArray[Any]]:
        """Reverse the PCA transformation back to the original space.

        Reconstructs the full water surface elevation field, filling in
        always-dry cells with their original elevation values.

        Args:
            mean (NDArray[Any]): Array of GPR mean estimates of shape (samples, spatial_mode_count) in EOF space.
            var (NDArray[Any]): Array of GPR variance estimates of shape (samples, spatial_mode_count) in EOF space.

        Returns:
            NDArray[Any]: Array of shape (samples, cells) in original space.
        """
        mean = (mean * self.x_std) + self.x_mean
        mean = np.dot(mean, self.eofs)
        if self.weights is not None:
            mean /= self.weights
        mean += self.input_mean
        x_full = np.empty((mean.shape[0], self.dry_indices.shape[0]))
        if self.hydraulic_parameter == "depth":
            x_full[:, self.dry_indices] = 0
        else:
            x_full[:, self.dry_indices] = self.elevations[self.dry_indices]
        x_full[:, ~self.dry_indices] = mean
        if var is None:
            return x_full
        else:
            var_prop = var.dot(self._linear_transform_for_var)
            var_prop_full = np.empty((var_prop.shape[0], self.dry_indices.shape[0]))
            var_prop_full[:, self.dry_indices] = 0
            var_prop_full[:, ~self.dry_indices] = var_prop
            return x_full, var_prop_full

    @cached_property
    def _linear_transform_for_var(self) -> NDArray[Any]:
        """Squared linear transform for error propogation in reverse transform."""
        a = np.diag(self.x_std)
        a = a.dot(self.eofs)
        if self.weights is not None:
            a /= self.weights.reshape(1, -1)
        return a**2

    def classify_wetness_wse(self, x: NDArray[Any], elevations: NDArray[Any]) -> NDArray[np.str_]:
        """Classify each cell as always dry (AD), always flooded (AF), or transitionally flooded (TF).

        Classification is based on how the depth varies across samples relative to a wetness threshold.

        Args:
            x (NDArray[Any]): Array of shape (samples, cells) representing water surface elevations.
            elevations (NDArray[Any]): Elevation values for each cell.

        Returns:
            NDArray[Any]: Array of strings ("AD", "AF", or "TF") indicating wetness class per cell.
        """
        max_depth = x.max(axis=0) - elevations
        min_depth = x.min(axis=0) - elevations
        return self._classify_depths(max_depth, min_depth)

    def classify_wetness_depth(self, x: NDArray[Any]) -> NDArray[np.str_]:
        """Classify each cell as always dry (AD), always flooded (AF), or transitionally flooded (TF).

        Classification is based on how the depth varies across samples relative to a wetness threshold.

        Args:
            x (NDArray[Any]): Array of shape (samples, cells) representing water surface elevations.
            elevations (NDArray[Any]): Elevation values for each cell.

        Returns:
            NDArray[Any]: Array of strings ("AD", "AF", or "TF") indicating wetness class per cell.
        """
        max_depth = x.max(axis=0)
        min_depth = x.min(axis=0)
        return self._classify_depths(max_depth, min_depth)

    def _classify_depths(self, max_depth: NDArray[Any], min_depth: NDArray[Any]) -> NDArray[np.str_]:
        classes = np.empty(max_depth.shape, dtype="<U2")
        classes[max_depth < self.wet_threshold] = "AD"  # Always Dry
        classes[max_depth > self.wet_threshold] = "TF"  # Transitionally Flooded
        classes[min_depth > self.wet_threshold] = "AF"  # Always Flooded
        return classes

    def to_dict(self) -> dict[str, Any]:
        """Dictionary representation of class for serialization."""
        return {
            "spatial_mode_count": self.spatial_mode_count,
            "wet_threshold": self.wet_threshold,
            "hydraulic_parameter": self.hydraulic_parameter,
            "elevations": self.elevations,
            "wetness_classes": self.wetness_classes,
            "input_mean": self.input_mean,
            "weights": self.weights,
            "eofs": self.eofs,
            "eigenvalues": self.eigenvalues,
            "n_samples_fit": self.n_samples_fit,
            "x_mean": self.x_mean,
            "x_std": self.x_std,
        }

    def to_file(self, out_path: str | PathLike[str]) -> None:
        """Save dict representation of self to file."""
        with open(out_path, mode="wb") as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def from_file(cls, in_path: str | PathLike[str]) -> Self:
        """Deserialize instance of self from a file representation."""
        with open(in_path, mode="rb") as f:
            d = pickle.load(f)
        return cls(**d)


class HmsPreProcessor:
    """Preprocessor for feature engineering from precip and inflow boundary conditions."""

    def __init__(
        self,
        precip_spatial_mode_count: int = 0,
        bc_mask: NDArray[Any] | None = None,
        precip_mask: NDArray[Any] | None = None,
        eofs: NDArray[Any] | None = None,
        eigenvalues: NDArray[Any] | None = None,
        n_samples_fit: float = 0,
        x_mean: NDArray[Any] | None = None,
        x_std: NDArray[Any] | None = None,
        input_mean: NDArray[Any] | None = None,
    ):
        """Preprocessor class constructor.

        When default values are used (None, for most arguments), their values will be set during the fit() method.

        Args:
            precip_spatial_mode_count (int): Number of spatial modes for PCA. Defaults to 0.
            bc_mask (NDArray[Any] | None, optional): Boolean mask for boundary condition columns. Defaults to None.
            precip_mask (NDArray[Any] | None, optional): Boolean mask for precipitation columns. Defaults to None.
            eofs (NDArray[Any] | None, optional): Empirical Orthogonal Functions (EOFs) from PCA. Defaults to None.
            eigenvalues (NDArray[Any] | None, optional): Eigenvalues from PCA. Defaults to None.
            n_samples_fit (float, optional): Number of samples used during PCA fitting. Defaults to 0.
            x_mean (NDArray[Any] | None, optional): Mean of spatial modes. Defaults to None.
            x_std (NDArray[Any] | None, optional): Standard deviation of spatial modes. Defaults to None.
            input_mean (NDArray[Any] | None, optional): Input data means. Defaults to None.

        Returns:
            None
        """
        self.precip_spatial_mode_count: int = precip_spatial_mode_count
        self.bc_mask = bc_mask if bc_mask is not None else np.empty(0, dtype=float)
        self.precip_mask = precip_mask if precip_mask is not None else np.empty(0, dtype=float)
        self.eofs: NDArray[Any] = eofs if eofs is not None else np.empty(0, dtype=float)
        self.eigenvalues: NDArray[Any] = eigenvalues if eigenvalues is not None else np.empty(0, dtype=float)
        self.n_samples_fit = n_samples_fit
        self.x_mean: NDArray[Any] = x_mean if x_mean is not None else np.empty(0, dtype=float)
        self.x_std: NDArray[Any] = x_std if x_std is not None else np.empty(0, dtype=float)
        self.input_mean: NDArray[Any] = input_mean if input_mean is not None else np.empty(0, dtype=float)

    def fit(
        self,
        x: NDArray[Any],
        bc_mask: NDArray[Any],
        precip_mask: NDArray[Any],
        precip_spatial_mode_count: int | None = None,
    ) -> None:
        """Fit the preprocessor to the input data using PCA.

        Args:
            x (NDArray[Any]): Array of shape (samples, features) representing discharge+precip. into the model.
            bc_mask (NDArray[Any] | None, optional): Boolean mask for boundary condition columns. Defaults to None.
            precip_mask (NDArray[Any] | None, optional): Boolean mask for precipitation columns. Defaults to None.
            precip_spatial_mode_count (int): Optional number of spatial modes to use for precipitation PCA.

        Returns:
            None
        """
        self.input_mean = x.mean(axis=0)
        x = x - self.input_mean

        self.bc_mask = bc_mask
        self.precip_mask = precip_mask
        x_bc = x[:, self.bc_mask]
        x_precip = x[:, self.precip_mask]

        # Fit PCA
        pca = IncrementalPCA()  # Documentation says that the function can batch itself
        pca.fit(x_precip)

        # Reduce modes
        if precip_spatial_mode_count is None:
            self.precip_spatial_mode_count = compute_norths_rule(pca)
            # TODO: Consider these methods https://stats.stackexchange.com/questions/33917/how-to-determine-significant-principal-components-using-bootstrapping-or-monte-c
        else:
            self.precip_spatial_mode_count = precip_spatial_mode_count

        # Set results
        self.eofs = pca.components_[: self.precip_spatial_mode_count]
        self.eigenvalues = pca.explained_variance_
        self.n_samples_fit = pca.n_samples_seen_

        # Derive precip features
        avg_precip = np.mean(x_precip, axis=1)
        api_1 = self.calc_antecedent_precipitation_index(avg_precip)
        api_2 = self.calc_antecedent_precipitation_index(avg_precip, k=1)
        precip_reduced = np.dot(x_precip, self.eofs.T)

        # Combine all features
        x = np.concatenate([x_bc, precip_reduced, avg_precip[:, None], api_1, api_2], axis=1)

        # Set second round of standardization
        self.x_mean = x.mean(axis=0)
        self.x_std = np.array([np.std(x[x[:, i] != 0, i]) for i in range(x.shape[1])])

    def transform(self, x: NDArray[Any]) -> NDArray[Any]:
        """Transform boundary condition data to reduced data space and derive features."""
        # Split feature types.
        x = x - self.input_mean
        x_bc = x[:, self.bc_mask]
        x_precip = x[:, self.precip_mask]

        # Derive precip features
        avg_precip = np.mean(x_precip, axis=1)
        api_1 = self.calc_antecedent_precipitation_index(avg_precip)
        api_2 = self.calc_antecedent_precipitation_index(avg_precip, k=1)
        precip_reduced = np.dot(x_precip, self.eofs.T)

        # Combine all features
        x = np.concatenate([x_bc, precip_reduced, avg_precip[:, None], api_1, api_2], axis=1)

        # Standardize
        x = (x - self.x_mean) / self.x_std

        return cast(NDArray[Any], x)

    def calc_antecedent_precipitation_index(
        self, x: NDArray[Any], k: float = 0.85, window: int | None = None
    ) -> NDArray[Any]:
        """Get precipitation index for a timeseries.

        https://glossary.ametsoc.org/wiki/Antecedent_precipitation_index. Exponential decay kernel.
        """
        if window is None:
            window = len(x)
        weights = np.array([k**i for i in range(window)])
        return np.convolve(x, weights, mode="full")[: len(x), np.newaxis]

    def to_dict(self) -> dict[str, Any]:
        """Dictionary representation of class for serialization."""
        return {
            "precip_spatial_mode_count": self.precip_spatial_mode_count,
            "bc_mask": self.bc_mask,
            "precip_mask": self.precip_mask,
            "eofs": self.eofs,
            "eigenvalues": self.eigenvalues,
            "n_samples_fit": self.n_samples_fit,
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "input_mean": self.input_mean,
        }

    def to_file(self, out_path: str | PathLike[str]) -> None:
        """Save dict representation of self to file."""
        with open(out_path, mode="wb") as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def from_file(cls, in_path: str | PathLike[str]) -> Self:
        """Deserialize instance of self from a file representation."""
        with open(in_path, mode="rb") as f:
            d = pickle.load(f)
        return cls(**d)


def compute_norths_rule(pca: PCA | IncrementalPCA) -> int:
    """Determine the optimal number of PCA modes using North's Rule.

    North's Rule compares the drop-off between successive eigenvalues
    to an estimate of sampling uncertainty. Components with eigenvalues less
    than one are automatically dropped.

    Args:
        pca (PCAType): A fitted PCA or IncrementalPCA object.

    Returns:
        int: Number of significant EOF modes to retain.
    """
    if isinstance(pca, PCA):
        n = pca.n_samples_
    elif isinstance(pca, IncrementalPCA):
        n = pca.n_samples_seen_
    else:
        return 0
    eigenvalues = pca.explained_variance_
    eigenvalues = eigenvalues[eigenvalues > 1]  # Kaiser rule. Filter out eigenvalues <= 1
    if len(eigenvalues) == 0:
        return 0

    d_eigen = np.abs(np.diff(eigenvalues))
    d_error = np.sqrt(2 / n) * eigenvalues[:-1]
    ind = np.argmax(d_eigen <= d_error)
    if ind == 0:
        return int(len(eigenvalues))
    else:
        return int(ind)
