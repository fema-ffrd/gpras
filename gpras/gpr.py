"""Methods for GPR training, tuning, and prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Self, TypeVar, cast

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import SGPR
from numpy.typing import NDArray
from scipy.optimize import differential_evolution
from sklearn.cluster import KMeans

gpflow.config.set_default_float(tf.float64)
T = TypeVar("T", bound="GPRAS")

KERNEL_FACTORY = {
    "Matern12": gpflow.kernels.Matern12,
    "Matern32": gpflow.kernels.Matern32,
    "Matern52": gpflow.kernels.Matern52,
    "RBF": gpflow.kernels.SquaredExponential,
    "Linear": gpflow.kernels.Linear,  # currently not working with optimizers
    "Polynomial": gpflow.kernels.Polynomial,  # currently not working with optimizers
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
OptimizerType = Literal["two-stage", "adam", "L-BFGS-B", "stochastic", "diffential_evolution"]
InductionInitializerType = Literal["kmeans", "grid"]


def _optimize_differential_evolutions(model: SGPR, popsize: int = 15, maxiter: int = 500) -> None:
    """Use differential evolution to optimize any trainable model parameters."""
    # Hyperparameter bounds
    param_bounds = [(-1, 1), (-1, 1), (-3, 0)]

    # Inducing variable bounds
    x = model.data[0].numpy()
    z_shape = model.inducing_variable.Z.shape
    mins, maxs = x.min(axis=0), x.max(axis=0)
    for d in range(z_shape[1]):
        param_bounds.extend([(mins[d], maxs[d])] * z_shape[0])

    # Define objective
    def objective(params: list[NDArray[Any]]) -> Any:
        model.kernel.variance.assign(10 ** params[0])
        model.kernel.lengthscales.assign(10 ** params[1])
        model.likelihood.variance.assign(10 ** params[2])
        model.inducing_variable.Z = np.array(params[3:]).reshape(z_shape)

        print(model.training_loss().numpy())
        return model.training_loss().numpy()

    # Run optimization
    result = differential_evolution(objective, param_bounds, popsize=popsize, maxiter=maxiter)

    # Set results
    model.kernel.variance.assign(10 ** result.x[0])
    model.kernel.lengthscales.assign(10 ** result.x[1])
    model.likelihood.variance.assign(10 ** result.x[2])
    model.inducing_variable.Z = np.array(result.x[3:]).reshape(z_shape)


def _optimize_multi_start(model: SGPR, n_starts: int = 40, iter_initial: int = 20, iter_final: int = 1000) -> None:
    """Run an initial coarse parameter optimization, identify the best parameters, then fine tune."""
    # Initialize random number generator
    np.random.seed(1)
    rng = np.random.default_rng()

    # Define shapes of hyperparameters
    x = model.data[0].numpy()
    z = model.inducing_variable.Z
    mins, maxs = x.min(axis=0), x.max(axis=0)
    z_shape = (z.shape[0], x.shape[1])

    # Multiple starts
    best_loss = None
    for _ in range(n_starts):
        model.kernel.variance.assign(10 ** rng.uniform(-1, 1))
        model.kernel.lengthscales.assign(10 ** rng.uniform(-1, 1))
        model.likelihood.variance.assign(10 ** rng.uniform(-3, 0))
        model.inducing_variable.Z = rng.uniform(mins, maxs, size=z_shape)

        _optimize_adam(model, iter_initial)

        loss = model.training_loss().numpy()
        if best_loss is None or loss < best_loss:
            best_params = [
                model.kernel.variance.numpy(),
                model.kernel.lengthscales.numpy(),
                model.likelihood.variance.numpy(),
                model.inducing_variable.Z,
            ]

    # Final tuning
    model.kernel.variance.assign(best_params[0])
    model.kernel.lengthscales.assign(best_params[1])
    model.likelihood.variance.assign(best_params[2])
    model.inducing_variable.Z = best_params[3]
    _optimize_bfgs(model, iter_final)


def _optimize_two_stage(model: SGPR, max_iter: int = 100) -> None:
    """Use Adam algorithm to optimize any trainable model parameters but optimizing inducing points first."""
    # Optimize inducing points
    gpflow.set_trainable(model, False)
    gpflow.set_trainable(model.inducing_variable.Z, True)
    _optimize_adam(model, max_iter)

    # Optimize other parameters
    gpflow.set_trainable(model, True)
    gpflow.set_trainable(model.inducing_variable.Z, False)
    _optimize_adam(model, max_iter)

    # Cleanup
    gpflow.set_trainable(model.inducing_variable.Z, True)


def _optimize_adam(model: SGPR, max_iter: int) -> None:
    """Use Adam algorithm to optimize any trainable model parameters."""
    opt = tf.keras.optimizers.Adam()

    @tf.function  # type: ignore[misc]
    def step() -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = model.training_loss()
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables, strict=False))
        return loss

    for _ in range(max_iter):
        step()


def _optimize_bfgs(model: SGPR, max_iter: int) -> None:
    """Use Scipy Limited-memory BFGS (gradient descent) to optimize any trainable model parameters."""
    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        model.training_loss,
        model.trainable_variables,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )


OPTIMIZERS: dict[str, Any] = {
    "two-stage": _optimize_two_stage,
    "adam": _optimize_adam,
    "L-BFGS-B": _optimize_bfgs,
    "stochastic": _optimize_multi_start,
    "diffential_evolution": _optimize_differential_evolutions,
}


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

    def fit(
        self,
        x: NDArray[Any],
        y: NDArray[Any],
        n_inducing: int,
        inducing_initializer: InductionInitializerType = "kmeans",
        optimization_method: OptimizerType = "two-stage",
        **opt_kwargs: dict[Any, Any],
    ) -> None:
        """Fit the Gaussian Process Regression model to observed data.

        Args:
            x (NDArray[Any]): The input data of shape (n_samples, n_features).
            y (NDArray[Any]): The target data of shape (n_samples, n_outputs).
            n_inducing (int): Number of inducing points to use for sparse approximation.
            inducing_initializer (InductionInitializerType, optional):
                Strategy for initializing inducing points (e.g., "kmeans", "random").
                Defaults to "kmeans".
            optimization_method (OptimizerType, optional):
                Optimization method to use for training
                (e.g., "adam", "l-bfgs", "two-stage"). Defaults to "two-stage".
            **opt_kwargs (dict[str, Any], optional):
                Additional keyword arguments passed to the optimizer.

        Returns:
            None
        """
        # Cast data to float64
        x = x.astype(np.float64)
        y = y.astype(np.float64)

        # Create base models
        self._init_models(x, y, n_inducing, inducing_initializer)

        # Optimize hyperparameters
        opt = OPTIMIZERS[optimization_method]
        for i in self.models:
            opt(i, **opt_kwargs)

    def _init_models(
        self,
        x: NDArray[Any],
        y: NDArray[Any],
        n_inducing: int,
        inducing_initializer: InductionInitializerType = "kmeans",
    ) -> None:
        """Create one model per spatial mode using base model settings."""
        # Make an inducing variable
        inducing = self._create_inducing(x, n_inducing, inducing_initializer)

        # Create models for each spatial mode
        self.models = []
        for i in range(y.shape[1]):
            # Subset to one spatial mode
            y_i = np.c_[y[:, i]]

            # Initialize model
            kernel_i = self.kernel()
            model = SGPR(data=(x, y_i), kernel=kernel_i, inducing_variable=inducing)

            # Set priors to avoid pathological models (hopefully)
            # TODO: investigate making this generic across kernels with model.kernel.parameters
            model.kernel.variance.prior = tfp.distributions.LogNormal(gpflow.utilities.to_default_float(0), 1.0)
            model.kernel.lengthscales.prior = tfp.distributions.LogNormal(gpflow.utilities.to_default_float(0), 1.0)
            model.likelihood.variance.prior = tfp.distributions.LogNormal(gpflow.utilities.to_default_float(0), 1.0)

            # Add model to list
            self.models.append(model)

    def _create_inducing(self, x: NDArray[Any], n_inducing: int, method: InductionInitializerType) -> NDArray[Any]:
        """Create an array representing locations in dataspace."""
        if method == "kmeans":
            km = KMeans(n_clusters=n_inducing, random_state=0, n_init="auto")
            km.fit(x)
            return cast(NDArray[Any], km.cluster_centers_.astype(np.float64))
        elif method == "grid":
            inducing_variable = np.c_[np.linspace(x[:, 0].min(), x[:, 0].max(), n_inducing)]
            for j in range(1, x.shape[1]):
                inducing_variable = np.c_[inducing_variable, np.linspace(x[:, j].min(), x[:, j].max(), n_inducing)]
            return cast(NDArray[Any], inducing_variable)

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
