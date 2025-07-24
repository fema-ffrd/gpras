"""HEC-RAS model class and associated functions."""

import shutil
from functools import cached_property
from pathlib import Path
from typing import Any, TypeVar, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from hecstac.ras.assets import GenericAsset, GeometryHdfAsset, PlanAsset, PlanHdfAsset, UnsteadyFlowAsset
from hecstac.ras.item import RASModelItem
from numpy.typing import NDArray
from pystac import Asset

from gpras.ras.flow import UnsteadyFlowFile
from gpras.ras.plan import (
    BoundaryType,
    PlanFile,
    update_hdf_attributes,
    update_hdf_data,
)
from gpras.utils.file_utils import detect_file_properties, get_filename

GenAsset = TypeVar("GenAsset", bound=GenericAsset)


class RasModel(RASModelItem):  # type: ignore[misc]
    """Subclass of hecstac with enhanced read/write/execute."""

    def make_new_plan(
        self, template_run: str, plan_attrs: dict[str, Any], boundary_conditions: list[BoundaryType]
    ) -> tuple[str, str]:
        """Append a new plan to the model."""
        # Define paths
        src_path: str = self.assets[template_run].href
        new_run = self.increment_suffix(self.plan_assets, "p")
        dst_path: str = src_path.replace(template_run, new_run + ".hdf")
        src_txt_path: str = src_path.replace(".hdf", "")
        dst_txt_path: str = dst_path.replace(".hdf", "")

        # Copy file
        shutil.copy(src_path, dst_path)
        shutil.copy(src_txt_path, dst_txt_path)

        # Update metadata
        update_hdf_attributes(dst_path, "Plan Data/Plan Information", plan_attrs["hdf"])
        update_text_attributes(dst_txt_path, plan_attrs["txt"])

        # Update project file
        add_file_to_prj_file(self.pf.fpath, dst_txt_path.split(".")[-1])

        # Write data
        for i in boundary_conditions:
            update_hdf_data(dst_path, i.path, i.data)

        # Add assets
        self.add_asset(get_filename(dst_path), Asset(dst_path, get_filename(dst_path)))
        self.add_asset(get_filename(dst_txt_path), Asset(dst_txt_path, get_filename(dst_txt_path)))
        self.plan_assets.append(self.assets[get_filename(dst_txt_path)])
        return dst_path, dst_txt_path

    def add_text_file(self, file: UnsteadyFlowFile | PlanFile) -> str:
        """Add a new text file (such as plan or flow file) to the RAS .prj."""
        if isinstance(file, UnsteadyFlowFile):
            line_base = "Unsteady File={}"
            existing_files = self.unsteady_flow_assets
            letter = "u"
        elif isinstance(file, PlanFile):
            line_base = "Plan File={}"
            existing_files = self.plan_assets
            letter = "p"
        else:
            return  # TODO: add more
        new_file = self.increment_suffix(existing_files, letter)
        new_path = Path(self.pm.model_root_dir) / new_file
        file.to_file(str(new_path))
        add_file_to_prj_file(self.pf.fpath, line_base.format(new_file.split(".")[-1]))
        asset = Asset(str(new_path), new_path.name)
        self.add_asset(new_path.name, asset)
        return str(new_path)

    def increment_suffix(self, paths: list[GenAsset], suffix_letter: str) -> str:
        """Take a list of paths, find the highest-numbered, and return that file incremented by 1."""
        file_hrefs = [i.href for i in paths]
        file_dict = {int(get_filename(i).split(".")[-1][1:]): get_filename(i) for i in file_hrefs}
        assert all(
            i.split(".")[1][0] == suffix_letter for i in file_hrefs
        ), f"Mismatched file types in path list: {file_hrefs}"
        new_plan_ind = 1
        while new_plan_ind in file_dict:
            new_plan_ind += 1
        new_suffix = f"{suffix_letter}{str(new_plan_ind).zfill(2)}"
        return cast(str, self.pm.derived_item_asset(f"{self.id}.{new_suffix}"))

    @property
    def unsteady_flow_assets(self) -> list[UnsteadyFlowAsset]:
        """Return any UnsteadyFlowAsset in assets."""
        return [a for a in self.assets.values() if isinstance(a, UnsteadyFlowAsset)]

    @property
    def plan_assets(self) -> list[PlanAsset]:
        """Return any RasGeomHdf in assets."""
        return [a for a in self.assets.values() if isinstance(a, PlanAsset)]

    @cached_property
    def plan_hdfs(self) -> dict[str, PlanHdfAsset]:
        """Get a dictionary mapping plan name to PlanHdfAsset."""
        return {
            i.extra_fields["HEC-RAS:plan_information_plan_name"]: i.file.hdf_object
            for i in self.assets.values()
            if isinstance(i, PlanHdfAsset)
        }

    @cached_property
    def geometry_hdfs(self) -> dict[str, GeometryHdfAsset]:
        """Get a dictionary mapping geometry name to GeometryHdfAsset."""
        return {
            i.file.hdf_object.get_geom_attrs()["Title"]: i.file.hdf_object
            for i in self.assets.values()
            if isinstance(i, GeometryHdfAsset)
        }

    def get_cell_minimum_elevation(self, plan: str, mesh_id: str) -> NDArray[Any]:
        """Get minimum elevation values for cells in a plan."""
        plan_asset = self.plan_hdfs[plan]
        mesh_path = f"Geometry/2D Flow Areas/{mesh_id}/Cells Minimum Elevation"
        elevations: NDArray[Any] = plan_asset[mesh_path][()]
        elevations = elevations[~np.isnan(elevations)]
        return elevations

    def get_plan_wsels(self, plans: list[str], mesh_id: str) -> pd.DataFrame:
        """Extract water surface elevations from a HEC-RAS plan at each computational cell."""
        store = []
        for p in plans:
            plan_asset = self.plan_hdfs[p]
            wse = plan_asset.mesh_timeseries_output(mesh_id, "Water Surface").values
            df = pd.DataFrame(wse)
            df["run"] = p
            df["t"] = df.index.to_list()
            store.append(df)
        merged = pd.concat(store)
        merged = merged.set_index(["run", "t"])
        return merged

    def get_plan_depths(self, plans: list[str], mesh_id: str) -> pd.DataFrame:
        """Extract depths from a HEC-RAS plan at each computational cell."""
        elevations = self.get_cell_minimum_elevation(plans[0], mesh_id)
        wsels = self.get_plan_wsels(plans, mesh_id)
        depths = wsels - elevations
        return depths

    def get_cell_areas(self, plan: str, mesh_id: str) -> NDArray[Any]:
        """Cached property that computes cell areas as weights.

        Returns:
            - A dictionary mapping cell IDs to surface area (used as weights).
            - A GeoDataFrame with cell_id, area, and geometry (useful for plotting).

        Only computed if `use_cell_weights` is enabled.
        """
        plan_asset = self.plan_hdfs[plan]
        mesh_path = f"Geometry/2D Flow Areas/{mesh_id}/Cells Surface Area"
        areas: NDArray[Any] = plan_asset[mesh_path][()]
        areas = areas[(~np.isnan(areas)) & (~np.isclose(areas, 0, 1e-3))]
        return areas

    def get_plan_geometry(self, plans: list[str], mesh_id: str) -> gpd.GeoDataFrame:
        """Get the geometry of a HEC-RAS plan."""
        geoms = [j.get_attrs(j.PLAN_INFO_PATH)["Geometry Title"] for i, j in self.plan_hdfs.items() if i in plans]
        assert all(i == geoms[0] for i in geoms), "Multiple geometries found in the hf model runs."
        geom = self.geometry_hdfs[geoms[0]]
        meshes = geom.mesh_cell_polygons()
        return meshes[meshes["mesh_name"] == mesh_id]


def add_file_to_prj_file(prj_path: str, file_row: str) -> None:
    """Add a plan suffix to a project file."""
    encoding, newline = detect_file_properties(prj_path)
    with open(prj_path, encoding=encoding) as f:
        lines = f.readlines()
    key = file_row.split("=")[0]
    last_index = max((ind for ind, line in enumerate(lines) if line.startswith(key)), default=len(lines))
    lines.insert(last_index + 1, file_row + "\n")
    with open(prj_path, mode="w", encoding=encoding, newline=newline) as f:
        f.writelines(lines)


def update_text_attributes(txt_path: str, attrs: dict[str, str]) -> None:
    """Update an attribute in a ras text file."""
    encoding, newline = detect_file_properties(txt_path)
    with open(txt_path, encoding=encoding) as f:
        lines = f.readlines()
    for ind, i in enumerate(lines):
        splitted = i.split("=")
        key = "=".join(splitted[:-1])
        if key in attrs:
            splitted[-1] = attrs[key] + "\n"
            lines[ind] = "=".join(splitted)
    with open(txt_path, mode="w", encoding=encoding, newline=newline) as f:
        f.writelines(lines)
