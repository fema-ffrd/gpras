"""Cross-validation analysis for optimizing GPR hyperparameters."""

from functools import cached_property
from pathlib import Path
from typing import Any

from production.analysis.data_models import Config
from production.analysis.pipeline import get_data_extracter, get_pre_processor, pipeline


class CVConfig(Config):
    """Subclass of regular config to access cross-validation plans instead of regular train-test."""

    @cached_property
    def train_plans(self) -> list[str]:
        """A list of HEC-RAS plans that should be used for training."""
        return [i["plan_title"] for i in self.event_plan_json if i["type"] == "Train" and i["set"] == "AEP"]

    @cached_property
    def test_plans(self) -> list[str]:
        """A List of HEC-RAS plans that should be used for testing."""
        return [i["plan_title"] for i in self.event_plan_json if i["type"] == "Test" and i["set"] == "diverse"]


def setup(config_path: str) -> None:
    """Prepare data and pre-processor that will be reused in subsequent cross-validation."""
    config = CVConfig.from_file(config_path)
    extracter = get_data_extracter(config, config.train_plans, config.training_data_db, True, True)
    get_data_extracter(config, config.test_plans, config.testing_data_db, True, True)
    hf_data_df, _ = extracter.aligned_datasets
    get_pre_processor(config, hf_data_df.values, extracter, True)


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
        config.metric_dir = out_dir
        config.plot_dir = plot_dir
        setattr(config, parameter, i)
        pipeline(config)


def run_kernels(config_path: str) -> None:
    """Benchmark kernel performance."""
    config = CVConfig.from_file(config_path)
    options = ["Matern12", "Matern32", "Matern52", "RBF", "Exponential"]
    parameter = "kernel"
    run_cv(config, parameter, options)


def run_spatial_modes(config_path: str) -> None:
    """Benchmark spatial mode count sensitivity."""
    config = CVConfig.from_file(config_path)
    options = [3, 5, 7, 10, 12, 15, 20, 30, 50, 100, 300]
    parameter = "spatial_mode_count"
    run_cv(config, parameter, options)


def run_inducing_points(config_path: str) -> None:
    """Benchmark inducing point sensitivity."""
    config = CVConfig.from_file(config_path)
    options = [1, 3, 5, 10, 20, 50, 100, 300]
    parameter = "inducing_pt_count"
    run_cv(config, parameter, options)


if __name__ == "__main__":
    config_path = "data/cross_validation/cv.config.json"
    setup(config_path)
    run_kernels(config_path)
    run_spatial_modes(config_path)
    run_inducing_points(config_path)
