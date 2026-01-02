"""
Linear Time-Invariant (LTI) Model base class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class LTIModel(ABC):
    """
    Abstract base class for Linear Time-Invariant (LTI) models.
    """

    def __init__(
        self,
        Ts: float = 1.0,
        nu: int = 1,
        ny: int = 1,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ):
        """
        Initialize the LTI model.

        Args:
            Ts: Sampling time in seconds.
            nu: Number of inputs.
            ny: Number of outputs.
            input_names: List of input channel names.
            output_names: List of output channel names.
        """
        if input_names is None:
            input_names = []
        if output_names is None:
            output_names = []

        self.Ts = Ts
        self.info = {}
        self.nu = nu
        self.ny = ny
        self.input_names = input_names
        self.output_names = output_names

        # Identification results
        self.aic: Optional[float] = None
        self.bic: Optional[float] = None
        self.residuals: Optional[np.ndarray] = None
        self.residuals_analysis: Optional[Dict[str, Any]] = None

    def impulse_response(
        self, N: int = 100, uncertainty: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Simulate the impulse response of the system.

        Args:
            N: Number of time steps to simulate.
            uncertainty: If True, return standard deviation of the response.

        Returns:
            If uncertainty is False:
                Tuple[np.ndarray, np.ndarray]: Time vector and output response.
            If uncertainty is True:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: Time vector, output response, and standard deviation.
        """
        t = np.linspace(0, (N - 1) * self.Ts, N)
        u = np.zeros((N, self.nu))
        # Impulse with area 1: height = 1/Ts, width = Ts
        u[0, :] = 1.0 / self.Ts

        y = self.simulate(u)

        if uncertainty:
            if not hasattr(self, "cov") or self.cov is None:
                return t, y, None

            def func():
                return self.simulate(u).ravel()

            y_std = self._propagate_uncertainty(func)
            y_std = y_std.reshape(y.shape)
            return t, y, y_std

        return t, y

    def step_response(
        self, N: int = 100, uncertainty: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Simulate the step response of the system.

        Args:
            N: Number of time steps to simulate.
            uncertainty: If True, return standard deviation of the response.

        Returns:
            If uncertainty is False:
                Tuple[np.ndarray, np.ndarray]: Time vector and output response.
            If uncertainty is True:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: Time vector, output response, and standard deviation.
        """
        t = np.linspace(0, (N - 1) * self.Ts, N)
        u = np.ones((N, self.nu))

        y = self.simulate(u)

        if uncertainty:
            if not hasattr(self, "cov") or self.cov is None:
                return t, y, None

            def func():
                return self.simulate(u).ravel()

            y_std = self._propagate_uncertainty(func)
            y_std = y_std.reshape(y.shape)
            return t, y, y_std

        return t, y

    def _propagate_uncertainty(self, func) -> np.ndarray:
        """
        Compute standard deviation of func() output using error propagation.

        Args:
            func: Callable that returns a 1D array (or scalar).
                  It should use the model's current parameters.

        Returns:
            std: Standard deviation of the output (same shape as func output).
        """
        if not hasattr(self, "vectorize") or not hasattr(self, "reshape"):
            raise NotImplementedError("Model must implement vectorize() and reshape() for uncertainty propagation.")

        theta_opt = self.vectorize()
        n_params = len(theta_opt)
        epsilon = 1e-8

        # Compute nominal output
        y_nominal = func()
        n_out = y_nominal.size

        # Compute Jacobian
        # Handle complex output if necessary, though currently we wrap to real
        is_complex = np.iscomplexobj(y_nominal)
        dtype = np.complex128 if is_complex else np.float64

        J = np.zeros((n_out, n_params), dtype=dtype)

        for i in range(n_params):
            theta_perturbed = theta_opt.copy()
            theta_perturbed[i] += epsilon

            self.reshape(theta_perturbed)
            y_perturbed = func()

            J[:, i] = (y_perturbed - y_nominal) / epsilon

        # Restore parameters
        self.reshape(theta_opt)

        # Compute variance: diag(J @ cov @ J.T)
        cov = self.cov

        if is_complex:
            var = np.sum((J @ cov) * np.conj(J), axis=1)
            var = np.real(var)
        else:
            var = np.sum((J @ cov) * J, axis=1)

        var = np.maximum(var, 0.0)

        return np.sqrt(var)

    def compare(self, y: np.ndarray, u: np.ndarray) -> float:
        """
        Compare model output with measured output.

        Args:
            y: Measured output.
            u: Input signal.

        Returns:
            float: Normalized Root Mean Squared Error (NRMSE) fit index (1 - NRMSE).
                   1.0 means perfect fit.
        """
        y_hat = self.simulate(u)
        # NRMSE returns error ratio, so 1 - NRMSE is the fit
        return 1.0 - self.NRMSE(y, y_hat)

    @staticmethod
    def residuals(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """Calculate residuals (error)."""
        return y.ravel() - y_hat.ravel()

    @staticmethod
    def SE(e: np.ndarray) -> np.ndarray:
        """Squared Error."""
        return np.power(e.ravel(), 2)

    @staticmethod
    def SSE(e: np.ndarray) -> float:
        """Sum of Squared Errors."""
        e_ = e.ravel()
        return float(e_.T @ e_)

    @staticmethod
    def MSE(e: np.ndarray) -> float:
        """Mean Squared Error."""
        e_ = e.ravel()
        return float((1.0 / len(e_)) * (e_.T @ e_))

    @staticmethod
    def RMSE(e: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(LTIModel.MSE(e))

    @staticmethod
    def NRMSE(y: np.ndarray, y_hat: np.ndarray, normalization: str = "matlab") -> float:
        """
        Normalized Root Mean Squared Error.

        Args:
            y: True output.
            y_hat: Predicted output.
            normalization: Normalization method ('matlab', 'mean', 'ptp').
                           'matlab': norm(e) / norm(y - mean(y))
                           'mean': RMSE(e) / mean(y)
                           'ptp': RMSE(e) / peak_to_peak(y)

        Returns:
            float: NRMSE value.
        """
        e = LTIModel.residuals(y, y_hat)
        if normalization == "matlab":
            # MATLAB's 'goodnessOfFit' NRMSE cost function:
            # norm(y - y_hat) / norm(y - mean(y))
            nrmse = np.linalg.norm(e) / np.linalg.norm(y - np.mean(y))
        elif normalization == "mean":
            nrmse = LTIModel.RMSE(e) / np.mean(y)
        elif normalization == "ptp":
            nrmse = LTIModel.RMSE(e) / np.ptp(y)
        else:
            raise ValueError(f"Unknown normalization method {normalization}")
        return float(nrmse)

    @abstractmethod
    def simulate(self, u: np.ndarray, uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate the model response to input u.

        Args:
            u: Input signal array of shape (N, nu).
            uncertainty: If True, return standard deviation of the response.

        Returns:
            If uncertainty is False:
                np.ndarray: Output signal array of shape (N, ny).
            If uncertainty is True:
                Tuple[np.ndarray, np.ndarray]: Output signal and standard deviation.
        """
        pass

    @abstractmethod
    def frequency_response(
        self, omega: Optional[np.ndarray] = None, uncertainty: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Calculate frequency response.

        Args:
            omega: Frequency vector (rad/s). If None, a default range is used.
            uncertainty: If True, return standard deviation of magnitude and phase.

        Returns:
            If uncertainty is False:
                Tuple[np.ndarray, np.ndarray]: Frequency vector and complex response.
            If uncertainty is True:
                Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                    Frequency vector, complex response, magnitude std, phase std.
        """
        pass
