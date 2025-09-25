"""Example script for running a GPR model training workflow."""

import inspect
import json
import time
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from gpras.gpr import GPRAS
from gpras.metrics import export_metric_summary
from gpras.preprocess import (
    DataBuilder,
    HmsPreProcessor,
    PreProcessor,
    RasReader,
)
from gpras.utils.plotting import (
    ec_pairplot,
    map_detection_categories,
    map_mesh_errors,
    performance_cdf,
    performance_scatterplot,
    plot_eof_maps,
    plot_timeseries_metrics,
    summary_plots,
)
from production.analysis.data_models import Config


def get_data_extracter(
    config: Config, plans: list[str], db_path: str, save: bool, plot_temporal_clips: bool
) -> RasReader | DataBuilder:
    """Get a data extracter (test or train) for the config and create one if necessary."""
    if not config.data_reader.is_valid(db_path):
        init_params = inspect.signature(config.data_builder.__init__).parameters
        init_dict = {k: getattr(config, k) for k in init_params if k != "self" and hasattr(config, k)}
        init_dict["plans"] = plans
        data_builder = config.data_builder(**init_dict)
        plot_dir = str(config.plot_dir / "temporal_clipping") if plot_temporal_clips else None
        data_builder._align_datasets(plot_dir)
        if save:
            data_builder.export_db(db_path)
        else:
            return data_builder
    return config.data_reader(db_path)


def get_hf_pre_processor(
    config: Config, data: pd.DataFrame, extracter: DataBuilder | RasReader, save: bool
) -> PreProcessor:
    """Get a HEC-RAS preprocessor for the config and create one if necessary."""
    if not (config.hf_preprocessor_path).exists():
        reducer = PreProcessor(wet_threshold=config.wet_threshold_depth, hydraulic_parameter=config.hydraulic_parameter)
        reducer.fit(data.values, extracter.cell_elevations, extracter.cell_areas, config.spatial_mode_count)
        if save:
            reducer.to_file(config.hf_preprocessor_path)
    else:
        reducer = PreProcessor.from_file(config.hf_preprocessor_path)
    return reducer


def get_hms_preprocessor(config: Config, data: pd.DataFrame, save: bool) -> HmsPreProcessor:
    """Get a HEC-HMS preprocessor for the config and create one if necessary."""
    if not (config.lf_preprocessor_path).exists():
        reducer = HmsPreProcessor()
        precip_mask = np.array([i.startswith("precip_") for i in data.columns])
        bc_mask = ~precip_mask
        reducer.fit(data.values, bc_mask, precip_mask, config.precip_spatial_mode_count)
        if save:
            reducer.to_file(config.lf_preprocessor_path)
    else:
        reducer = HmsPreProcessor.from_file(config.lf_preprocessor_path)
    return reducer


def get_pre_processors(
    config: Config, hf_data: pd.DataFrame, lf_data: pd.DataFrame, extracter: DataBuilder | RasReader, save: bool
) -> tuple[PreProcessor, PreProcessor | HmsPreProcessor]:
    """Get lf and hf data preprocessors."""
    hf_preprocessor = get_hf_pre_processor(config, hf_data, extracter, save)
    if config.lf_model_type in ["ras_upskill", "pseudo_surface", "ras_interpolate"]:
        return hf_preprocessor, hf_preprocessor
    elif config.lf_model_type == "hms_upskill":
        return hf_preprocessor, get_hms_preprocessor(config, lf_data, save)
    else:
        raise RuntimeError(f"No preprocessor setup available for LF model type: {config.lf_model_type}")


def gen_plots(
    config: Config,
    gpr: GPRAS,
    hf_mesh: gpd.GeoDataFrame,
    x: NDArray[Any],
    y: NDArray[Any],
    x_test: NDArray[Any],
    y_test: NDArray[Any],
    hf_data_df: pd.DataFrame,
    lf_test_data_df: pd.DataFrame,
    hf_test_data_df: pd.DataFrame,
    y_test_pred: NDArray[Any],
    mean_pred: NDArray[Any],
    lf_test_data_depth: NDArray[Any],
    hf_test_data_depth: NDArray[Any],
    y_test_pred_depth: NDArray[Any],
    eofs: NDArray[Any],
    wet_cell_ids: NDArray[Any],
) -> None:
    """Generate diagnostic plots for a pipeline run."""
    ec_pairplot(
        x,
        x,
        min(config.spatial_mode_count, 5),
        config.plot_dir / "inducing_fitted.png",
        gpr.models[0].inducing_variable.Z,
    )
    ec_pairplot(x_test, y_test, min(config.spatial_mode_count, 5), config.plot_dir / "pairplot_test.png")
    ec_pairplot(x, y, min(config.spatial_mode_count, 5), config.plot_dir / "pairplot.png")
    # ec_timeseries(x, y, min(config.spatial_mode_count, 5), hf_data_df.index, config.plot_dir / "ec_timeseries")
    # ec_timeseries_alt(x, y, 5, hf_data_df.index, ["u/s bc", "p1", "p2", "p3", "p4", "p5", "api1", "api2"], config.plot_dir / "ec_timeseries")
    performance_scatterplot(
        lf_test_data_df.values,
        hf_test_data_df.values,
        y_test_pred,
        config.plot_dir / "performance_scatterplot.png",
    )
    performance_cdf(
        lf_test_data_df.values,
        hf_test_data_df.values,
        y_test_pred,
        config.plot_dir / "performance_cdf.png",
    )
    ec_pairplot(mean_pred, y_test, min(config.spatial_mode_count, 5), config.plot_dir / "pairplot_test_predicted.png")
    performance_scatterplot(
        lf_test_data_depth,
        hf_test_data_depth,
        y_test_pred_depth,
        config.plot_dir / "performance_scatterplot_depth.png",
        depth=True,
    )
    map_mesh_errors(
        hf_mesh,
        config.metric_dir / "performance_metrics.db",
        config.plot_dir / "error_maps",
        suffix="rmse",
        error_field="rmse_cell_toi",
        error_metric="RMSE",
    )

    map_mesh_errors(
        hf_mesh,
        config.metric_dir / "performance_metrics.db",
        config.plot_dir / "error_maps",
        suffix="mts_error",
        error_field="err_cell_mts",
        error_metric="Max Depth Error",
    )

    map_mesh_errors(
        hf_mesh,
        config.metric_dir / "performance_metrics.db",
        config.plot_dir / "error_maps",
        suffix="mean_error",
        error_field="err_cell_toi",
        error_metric="Mean Error",
    )

    map_detection_categories(
        hf_mesh,
        hf_test_data_depth,
        y_test_pred_depth,
        hf_test_data_df.index.values,
        hf_test_data_df.columns.values,
        output_plot_path=config.plot_dir / "error_maps",
        include_correct_negative=True,
        wet_threshold_depth=config.wet_threshold_depth,
    )

    plot_timeseries_metrics(
        config.metric_dir / "performance_metrics.db",
        config.plot_dir / "error_timeseries",
        metrics_field=["rmse_aoi_ts", "err_aoi_ts"],
        metrics=["RMSE", "Mean Error"],
        overlay=True,
    )

    summary_plots(
        config.metric_dir / "performance_metrics.db",
        config.plot_dir,
        metrics={
            "cell_metrics": {
                "rmse_cell_toi": "Spatial RMSE",
                "err_cell_mts": "Spatial Mean Error (Max)",
                "err_cell_toi": "Spatial Mean Error",
            },
            "scalar_metrics": {
                "nse_aoi_mts": "NSE",
                "err_aoi_mts": "Max Error",
                "fi_aoi_toi": "Fidelity Index",
            },
            "timeseries_metrics": {"rmse_aoi_ts": "Temporal RMSE", "err_aoi_ts": "Temporal Mean Error"},
        },
    )

    plot_eof_maps(
        eofs, wet_cell_ids, hf_mesh, config.plot_dir, n_modes=3, cell_id_field=config.cell_id_field, cmap="viridis"
    )


def pipeline(config: Config) -> None:
    """Automate the process of loading HEC-RAS model data, fitting the GPR, and making predictions."""
    ### Load data ###
    t1 = time.perf_counter()
    print("Loading data")
    extracter = get_data_extracter(
        config, config.train_plans, config.training_data_db, config.save_dbs, config.generate_plots
    )
    hf_data_df, lf_data_df = extracter.aligned_datasets
    hf_data = hf_data_df.values
    lf_data = lf_data_df.values

    test_extracter = get_data_extracter(
        config, config.test_plans, config.testing_data_db, config.save_dbs, config.generate_plots
    )
    hf_test_data_df, lf_test_data_df = test_extracter.aligned_datasets
    hf_test_data = hf_test_data_df.values
    lf_test_data = lf_test_data_df.values

    ### Preprocess data ###
    t2 = time.perf_counter()
    print("Preprocessing data")
    hf_reducer, lf_reducer = get_pre_processors(config, hf_data_df, lf_data_df, extracter, config.save_preprocessor)
    y = hf_reducer.transform(hf_data)
    x = lf_reducer.transform(lf_data)
    y_test = hf_reducer.transform(hf_test_data)
    x_test = lf_reducer.transform(lf_test_data)

    ### Fit GPR ###
    t3 = time.perf_counter()
    print("Fitting GPR")
    if True:  # Manually set to train or load
        gpr = GPRAS(config.kernel)
        gpr.fit(
            x,
            y,
            config.inducing_pt_count,
            config.induction_pt_initializer,
            config.optimizer,
            **config.optimizer_kwargs,  #
        )
        gpr.to_file(config.model_path)
    gpr = GPRAS.from_file(config.model_path)

    ### Predict test data ###
    t4 = time.perf_counter()
    print("Making predictions")
    mean_pred, var_pred = gpr.predict(x_test)
    y_test_pred, y_test_var = hf_reducer.reverse_transform(mean_pred, var_pred)
    _ = y_test_pred + (norm.ppf(0.975) * np.sqrt(y_test_var))  # High est
    _ = y_test_pred + (norm.ppf(0.025) * np.sqrt(y_test_var))  # Low est

    if config.hydraulic_parameter == "depth":
        y_test_pred += hf_reducer.elevations
    lf_test_data_depth = (
        hf_reducer.wse_2_depth(lf_test_data)
        if config.lf_model_type in ["ras_upskill", "pseudo_surface", "ras_interpolate"]
        else lf_test_data
    )
    hf_test_data_depth = hf_reducer.wse_2_depth(hf_test_data)
    y_test_pred_depth = hf_reducer.wse_2_depth(y_test_pred)

    ### Assess performance and plot diagnostics ###
    t5 = time.perf_counter()
    print("Calculating metrics and making performance plots")
    export_metric_summary(
        pd.DataFrame(hf_test_data_depth, index=hf_test_data_df.index, columns=hf_test_data_df.columns),
        pd.DataFrame(y_test_pred_depth, index=hf_test_data_df.index, columns=hf_test_data_df.columns),
        pd.DataFrame(np.sqrt(y_test_var), index=hf_test_data_df.index, columns=hf_test_data_df.columns),
        config.metric_db_path,
    )
    with open(config.timer_path, mode="w") as f:
        json.dump(
            {"load_data": t2 - t1, "preprocess_data": t3 - t2, "fit_model": t4 - t3, "make_predictions": t5 - t4},
            f,
            indent=4,
        )
    if config.generate_plots:
        gen_plots(
            config,
            gpr,
            extracter.hf_geometry_aoi,
            x,
            y,
            x_test,
            y_test,
            hf_data_df,
            lf_test_data_df,
            hf_test_data_df,
            y_test_pred,
            mean_pred,
            lf_test_data_depth,
            hf_test_data_depth,
            y_test_pred_depth,
            hf_reducer.eofs,
            extracter.hf_geometry_aoi[config.cell_id_field][~hf_reducer.dry_indices].tolist(),
        )


def gen_plots_post_hoc(config: Config) -> None:
    """Generate plots on a pre-trained model."""
    ### Load data ###
    print("Loading data")
    extracter = get_data_extracter(
        config, config.train_plans, config.training_data_db, config.save_dbs, config.generate_plots
    )
    hf_data_df, lf_data_df = extracter.aligned_datasets
    hf_data = hf_data_df.values
    lf_data = lf_data_df.values

    test_extracter = get_data_extracter(
        config, config.test_plans, config.testing_data_db, config.save_dbs, config.generate_plots
    )
    hf_test_data_df, lf_test_data_df = test_extracter.aligned_datasets
    hf_test_data = hf_test_data_df.values
    lf_test_data = lf_test_data_df.values

    ### Preprocess data ###
    print("Preprocessing data")
    hf_reducer, lf_reducer = get_pre_processors(config, hf_data_df, lf_data_df, extracter, config.save_preprocessor)
    y = hf_reducer.transform(hf_data)
    x = lf_reducer.transform(lf_data)
    y_test = hf_reducer.transform(hf_test_data)
    x_test = lf_reducer.transform(lf_test_data)

    ### Load GPR ###
    print("Loading GPR")
    gpr = GPRAS.from_file(config.model_path)

    ### Predict test data ###
    print("Making predictions")
    mean_pred, _ = gpr.predict(x_test)
    y_test_pred = hf_reducer.reverse_transform(mean_pred)
    if config.hydraulic_parameter == "depth":
        y_test_pred += hf_reducer.elevations
    lf_test_data_depth = hf_reducer.wse_2_depth(lf_test_data) if config.lf_model_type == "ras_upskill" else lf_test_data
    hf_test_data_depth = hf_reducer.wse_2_depth(hf_test_data)
    y_test_pred_depth = hf_reducer.wse_2_depth(y_test_pred)

    ### Assess performance and plot diagnostics ###
    print("Making performance plots")
    gen_plots(
        config,
        gpr,
        extracter.hf_geometry_aoi,
        x,
        y,
        x_test,
        y_test,
        hf_data_df,
        lf_test_data_df,
        hf_test_data_df,
        y_test_pred,
        mean_pred,
        lf_test_data_depth,
        hf_test_data_depth,
        y_test_pred_depth,
        hf_reducer.eofs,
        extracter.hf_geometry_aoi[config.cell_id_field][~hf_reducer.dry_indices].tolist(),
    )


if __name__ == "__main__":
    config = Config.from_file("data/ras_upskill/pipeline.config.json")
    pipeline(config)
    # gen_plots_post_hoc(config)
