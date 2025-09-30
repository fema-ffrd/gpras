"""Cross-validation analysis for optimizing GPR hyperparameters."""

import json
from dataclasses import asdict
from functools import cached_property
from pathlib import Path
from typing import Any

from production.analysis.data_models import Config
from production.analysis.pipeline import get_data_extracter, get_pre_processors, pipeline


class CVConfig(Config):
    """Subclass of regular config to access cross-validation plans instead of regular train-test."""

    @cached_property
    def train_plans(self) -> list[str]:
        """A list of HEC-RAS plans that should be used for training."""
        return [i["plan_title"] for i in self.event_plan_json if i["type"] == "Train" and i["set"] == "AEP"]

    @cached_property
    def test_plans(self) -> list[str]:
        """A List of HEC-RAS plans that should be used for testing."""
        return [i["plan_title"] for i in self.event_plan_json if i["type"] == "Train" and i["set"] == "Diverse"]


def setup(config_path: str) -> None:
    """Prepare data and pre-processor that will be reused in subsequent cross-validation."""
    config = CVConfig.from_file(config_path)
    extracter = get_data_extracter(config, config.train_plans, config.training_data_db, True, True)
    get_data_extracter(config, config.test_plans, config.testing_data_db, True, True)
    hf_data_df, lf_data_df = extracter.aligned_datasets
    get_pre_processors(config, hf_data_df, lf_data_df, extracter, True)


def run_cv(config: CVConfig, parameter: str, options: list[Any]) -> None:
    """Benchmark peneric performance."""
    base_dir = Path(config.working_directory) / parameter
    for i in options:
        print(i)
        out_dir = base_dir / str(i)
        if out_dir.exists():
            continue  # TODO: this logic could be improved.
        out_dir.mkdir(parents=True)
        plot_dir = out_dir / "plots"
        plot_dir.mkdir()
        config.metric_db_path = out_dir / "performance_metrics.db"
        config.plot_dir = plot_dir
        setattr(config, parameter, i)
        pipeline(config)


def run_cv_serial(config: CVConfig, options: dict[str, Any], base_dir: Path) -> None:
    """Benchmark peneric performance."""
    # Save baseline
    base_dir.mkdir(parents=True, exist_ok=True)
    with open(base_dir / "defaults.config.json", mode="w") as f:
        json.dump(asdict(config), f, indent=4)

    # Run cross-validation
    for i in range(len(options[next(iter(options))])):
        print(i)
        out_dir = base_dir / str(i)
        if out_dir.exists():
            continue  # TODO: this logic could be improved.
        out_dir.mkdir()
        plot_dir = out_dir / "plots"
        plot_dir.mkdir()
        config.metric_db_path = out_dir / "performance_metrics.db"
        config.plot_dir = plot_dir
        print("Running config:")
        for k in options:
            print(f" - {k} = {options[k][i]}")
            setattr(config, k, options[k][i])
            if k == "spatial_mode_count":  # Can't reuse pre-processor
                config.hf_preprocessor_path = out_dir / "model" / "hf_preprocessor.pkl"
                if config.lf_model_type == "ras_upskill":
                    config.lf_preprocessor_path = config.hf_preprocessor_path
                else:
                    config.lf_preprocessor_path = out_dir / "model" / "lf_preprocessor.pkl"
                config.model_dir = out_dir / "model"
                config.model_dir.mkdir(exist_ok=True)
        pipeline(config)
        with open(out_dir / "config.json", mode="w") as f:
            json.dump(asdict(config), f, indent=4)


def run_kernels(config_path: str) -> None:
    """Benchmark kernel performance."""
    config = CVConfig.from_file(config_path)
    options = {"kernel": ["Matern12", "Matern32", "Matern52", "RBF", "Exponential"]}
    base_dir = Path(config.working_directory) / "kernel"
    run_cv_serial(config, options, base_dir)


def run_spatial_modes(config_path: str) -> None:
    """Benchmark spatial mode count sensitivity."""
    # TODO: this could be sped up if we wanted to write some custom functions to reuse preprocessors and mode models
    config = CVConfig.from_file(config_path)
    options = {"spatial_mode_count": [1, 3, 5, 7, 10, 15, 20, 30, 50]}
    base_dir = Path(config.working_directory) / "spatial_mode_count"
    run_cv_serial(config, options, base_dir)


def run_inducing_points(config_path: str) -> None:
    """Benchmark inducing point sensitivity."""
    config = CVConfig.from_file(config_path)
    options = {"inducing_pt_count": [1, 3, 5, 10, 20, 50, 100, 300]}
    base_dir = Path(config.working_directory) / "inducing_pt_count"
    run_cv_serial(config, options, base_dir)


def run_optimization_method(config_path: str) -> None:
    """Benchmark inducing point sensitivity."""
    config = CVConfig.from_file(config_path)
    options = {
        "optimizer": ["two-stage", "adam", "L-BFGS-B", "stochastic", "diffential_evolution", "three-stage", "adadelta"],
        "optimizer_kwargs": [
            {"max_iter": 5000},
            {"max_iter": 10000},
            {"max_iter": 10000},
            {"n_starts": 50, "iter_initial": 100, "iter_final": 5000},
            {"popsize": 5, "max_iter": 100},
            {"max_iter": 333},
            {"max_iter": 10000},
        ],
    }
    base_dir = Path(config.working_directory) / "optimizer"
    run_cv_serial(config, options, base_dir)


if __name__ == "__main__":
    config_path = "data/cv/pseudo_surface/pipeline.config.json"
    setup(config_path)
    # run_optimization_method(config_path)
    run_kernels(config_path)
    run_spatial_modes(config_path)
    run_inducing_points(config_path)
