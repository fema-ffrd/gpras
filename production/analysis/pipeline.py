"""Example script for running a GPR model training workflow."""

import inspect
import json
import time
from typing import Any

import pandas as pd
from numpy.typing import NDArray

from gpras.gpr import GPRAS
from gpras.metrics import export_metric_summary
from gpras.preprocess import (
    DataBuilder,
    PreProcessor,
    RasReader,
)
from gpras.utils.plotting import ec_pairplot, performance_cdf, performance_scatterplot
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


def get_pre_processor(
    config: Config, hf_data: NDArray[Any], extracter: DataBuilder | RasReader, save: bool
) -> PreProcessor:
    """Get a preprocessor for the config and create one if necessary."""
    if not (config.preprocessor_path).exists():
        reducer = PreProcessor(wet_threshold=config.wet_threshold_depth, hydraulic_paramter=config.hydraulic_parameter)
        reducer.fit(hf_data, extracter.cell_elevations, extracter.cell_areas, config.spatial_mode_count)
        if save:
            reducer.to_file(config.preprocessor_path)
    else:
        reducer = PreProcessor.from_file(config.preprocessor_path)
    return reducer


def gen_plots(
    config: Config,
    gpr: GPRAS,
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
    reducer = get_pre_processor(config, hf_data, extracter, config.save_preprocessor)
    y = reducer.transform(hf_data)
    x = reducer.transform(lf_data)
    x_test = reducer.transform(lf_test_data)
    y_test = reducer.transform(hf_test_data)

    ### Fit GPR ###
    t3 = time.perf_counter()
    print("Fitting GPR")
    gpr = GPRAS(config.kernel)
    gpr.fit(
        x, y, config.inducing_pt_count, config.induction_pt_initializer, config.optimizer, **config.optimizer_kwargs
    )
    gpr.to_file(config.model_path)

    ### Predict test data ###
    t4 = time.perf_counter()
    print("Making predictions")
    mean_pred, _ = gpr.predict(x_test)
    y_test_pred = reducer.reverse_transform(mean_pred)
    if config.hydraulic_parameter == "depth":
        y_test_pred += reducer.elevations
    lf_test_data_depth = reducer.wse_2_depth(lf_test_data)
    hf_test_data_depth = reducer.wse_2_depth(hf_test_data)
    y_test_pred_depth = reducer.wse_2_depth(y_test_pred)

    ### Assess performance and plot diagnostics ###
    t5 = time.perf_counter()
    print("Calculating metrics and making performance plots")
    export_metric_summary(
        pd.DataFrame(hf_test_data_depth, index=hf_test_data_df.index),
        pd.DataFrame(y_test_pred_depth, index=hf_test_data_df.index),
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
        )


if __name__ == "__main__":
    config = Config.from_file("data/ras_upskill/pipeline.config.json")
    pipeline(config)
