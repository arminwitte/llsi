"""
Linear Time-Invariant (LTI) Model base class.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

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

    def impulse_response(self, N: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the impulse response of the system.

        Args:
            N: Number of time steps to simulate.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Time vector and output response.
        """
        t = np.linspace(0, (N - 1) * self.Ts, N)
        u = np.zeros((N, self.nu))
        # Impulse with area 1: height = 1/Ts, width = Ts
        u[0, :] = 1.0 / self.Ts

        y = self.simulate(u)

        return t, y

    def step_response(self, N: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the step response of the system.

        Args:
            N: Number of time steps to simulate.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Time vector and output response.
        """
        t = np.linspace(0, (N - 1) * self.Ts, N)
        u = np.ones((N, self.nu))

        y = self.simulate(u)

        return t, y

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
    def simulate(self, u: np.ndarray) -> np.ndarray:
        """
        Simulate the model response to input u.

        Args:
            u: Input signal array of shape (N, nu).

        Returns:
            np.ndarray: Output signal array of shape (N, ny).
        """
        pass

    @abstractmethod
    def frequency_response(self, omega: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate frequency response.

        Args:
            omega: Frequency vector (rad/s). If None, a default range is used.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Frequency vector and complex response.
        """
        pass
