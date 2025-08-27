"""Methods for GPR training, tuning, and prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Self, TypeVar

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.models import SGPR
from numpy.typing import NDArray
from sklearn.cluster import KMeans

T = TypeVar("T", bound="GPRAS")

KERNEL_FACTORY = {
    "Matern12": gpflow.kernels.Matern12,
    "Matern32": gpflow.kernels.Matern32,
    "Matern52": gpflow.kernels.Matern52,
    "RBF": gpflow.kernels.SquaredExponential,
    "Linear": gpflow.kernels.Linear,
    "Polynomial": gpflow.kernels.Polynomial,
    "Periodic": gpflow.kernels.Periodic,
    "Exponential": gpflow.kernels.Exponential,
    # Add more kernels as needed
    # Combination kernels
    # gpflow_kernel = gpflow.kernels.Sum([gpflow.kernels.RBF(), gpflow.kernels.Linear()])
    # OR
    # gpflow_kernel = gpflow.kernels.Product([gpflow.kernels.RBF(), gpflow.kernels.Periodic(base_kernel=gpflow.kernels.RBF(), period=1.0)])
    # Custom kernels can also be defined using gpflow.kernels.Kernel
    # Consider Change-points kernel
}

KernelType = Literal["Matern12", "Matern32", "Matern52", "RBF", "Linear", "Polynomial", "Periodic", "Exponential"]


class GPRAS:
    """Gaussian Process Regression for HEC-RAS model upskilling and emulation."""

    def __init__(self, kernel: KernelType) -> None:
        """Initialize the GPRAS class with a specified kernel.

        Args:
            kernel (str): The name of the kernel to use for Gaussian Process Regression.

        Returns:
            None
        """
        self.kernel_str = kernel
        self.kernel = KERNEL_FACTORY[kernel]

        self.models: list[SGPR] = []

    def fit(self, x: NDArray[Any], y: NDArray[Any], inducing_fraction: float) -> None:
        """Fit the Gaussian Process Regression model to observed data.

        Args:
            x (NDArray[Any]): The input data of shape (n_samples, n_features).
            y (NDArray[Any]): The target data of shape (n_samples, n_outputs).
            inducing_fraction (float): Fraction of the data to use as inducing points. This can tune training speed.

        Returns:
            None
        """
        # Cast variables to float64 for gpflow
        x = x.astype(np.float64)
        y = y.astype(np.float64)

        # Initialize inducing points with kmeans
        km = KMeans(n_clusters=round(x.shape[0] * inducing_fraction), random_state=0, n_init="auto")
        km.fit(x)
        inducing_variable = km.cluster_centers_.astype(np.float64)

        # Define gpr training optimizer

        # Train GPR models
        self.models = []
        for i in range(y.shape[1]):
            # Subset data & instantiate GPR
            y_i = np.c_[y[:, i]]
            kernel_i = self.kernel()
            model_i = SGPR(data=(x, y_i), kernel=kernel_i, inducing_variable=inducing_variable)
            options = {"maxiter": 500, "gtol": 1e-6, "ftol": 1e-9}

            # Optimize inducing points
            gpflow.set_trainable(model_i.kernel.variance, False)
            gpflow.set_trainable(model_i.kernel.lengthscales, False)
            gpflow.set_trainable(model_i.likelihood.variance, False)

            opt = gpflow.optimizers.Scipy()

            result = opt.minimize(
                model_i.training_loss,
                model_i.trainable_variables,
                method="L-BFGS-B",
                options=options,
            )
            print("Number of iterations:", result.nit)

            # Optimize kernel and model parameters
            gpflow.set_trainable(model_i.kernel.variance, True)
            gpflow.set_trainable(model_i.kernel.lengthscales, True)
            gpflow.set_trainable(model_i.likelihood.variance, True)
            gpflow.set_trainable(model_i.inducing_variable.Z, False)

            result = opt.minimize(
                model_i.training_loss, model_i.trainable_variables, method="L-BFGS-B", options=options
            )
            print("Number of iterations:", result.nit)
            print()

            # Log model
            self.models.append(model_i)

    def predict(self, x: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Predict high-fidelity data using the trained Gaussian Process Regression model.

        Args:
            x (NDArray[Any]): The input data of shape (n_samples, n_features).

        Returns:
            tuple:
                - NDArray[Any]: Predicted mean values of shape (n_samples, n_outputs).
                - NDArray[Any]: Predicted variances of shape (n_samples, n_outputs).
        """
        x = x.astype(np.float64)
        means = []
        variances = []
        for i in self.models:
            # x_i = np.c_[x[:, i]]
            pred = i.predict_y(x)
            means.append(pred[0])
            variances.append(pred[1])
        full_means = np.concatenate(means, axis=1)
        full_variances = np.concatenate(variances, axis=1)
        return full_means, full_variances

    def to_file(self, json_path: str | Path, model_dir: str | Path | None = None) -> None:
        """Serialize the trained Gaussian Process Regression model to disk.

        This method saves each trained model in a separate directory and creates a
        JSON representation that includes the kernel type and the paths to the saved models.

        Args:
            json_path (str): The path to the JSON file where the metadata will be saved.
            model_dir (str | None, optional): The directory where the models will be saved.
                                            If None, a default directory will be created
                                            relative to the JSON file path.

        Returns:
            None
        """
        if model_dir is None:
            model_dir = str(Path(json_path).parent / "gpr_model")
        dirs = []
        for ind, i in enumerate(self.models):
            tmp_dir = Path(model_dir) / f"model_{ind}"
            tmp_dir.mkdir(exist_ok=True, parents=True)
            tf.saved_model.save(i, tmp_dir)
            dirs.append(str(tmp_dir))

        json_rep = {"kernel": self.kernel_str, "models": dirs}
        with open(json_path, mode="w") as f:
            json.dump(json_rep, f, indent=4)

    @classmethod
    def from_file(cls, json_path: str) -> Self:
        """Load a Gaussian Process Regression model from a JSON representation.

        Args:
            json_path (str): The path to the JSON file containing the serialized model metadata.

        Returns:
            GPRAS: An instance of the GPRAS class with the loaded models and kernel.
        """
        with open(json_path) as f:
            json_rep = json.load(f)
        inst = cls(json_rep["kernel"])
        inst.models = [tf.saved_model.load(i) for i in json_rep["models"]]
        return inst
