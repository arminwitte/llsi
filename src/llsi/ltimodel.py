"""
Linear Time-Invariant (LTI) Model base class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.signal


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

    def compute_residuals_analysis(self, data: Any) -> Dict[str, Any]:
        """
        Compute residual analysis metrics (ACF, CCF) on validation data.

        Args:
            data: SysIdData object containing validation u and y.

        Returns:
            Dict containing:
                'residuals': The residuals (y - y_hat)
                'acf': Auto-correlation function of residuals (normalized)
                'ccf': Cross-correlation function of residuals and input (normalized)
                'lags': Lags for the correlations
                'conf_interval': 99% confidence interval value
        """
        u, y = self._extract_data(data)

        y_pred = self.simulate(u)
        residuals = y - y_pred

        # Use first output for summary analysis
        res_i = residuals[:, 0]
        res_i_centered = res_i - np.mean(res_i)

        # ACF
        acf = scipy.signal.correlate(res_i_centered, res_i_centered, mode="full")
        if np.max(acf) > 0:
            acf = acf / np.max(acf)  # Normalize
        lags = scipy.signal.correlation_lags(len(res_i_centered), len(res_i_centered))

        # CCF (Input 0 vs Residuals)
        u_0 = u[:, 0]
        u_0_centered = u_0 - np.mean(u_0)
        ccf = scipy.signal.correlate(res_i_centered, u_0_centered, mode="full")
        # Normalize CCF
        denom = np.std(res_i_centered) * np.std(u_0_centered) * len(res_i_centered)
        if denom > 0:
            ccf = ccf / denom
        else:
            ccf = np.zeros_like(ccf)

        # Handle NaNs in CCF (e.g. if std is NaN)
        if np.any(np.isnan(ccf)):
            ccf = np.nan_to_num(ccf)

        # Confidence interval (99%)
        N = len(residuals)
        conf_interval = 2.58 / np.sqrt(N)

        return {
            "residuals": residuals,
            "acf": acf,
            "ccf": ccf,
            "lags": lags,
            "conf_interval": conf_interval,
        }

    def aic(self, data: Any) -> float:
        """
        Calculate the Akaike Information Criterion (AIC).

        The AIC is a measure of the quality of a statistical model for a given set of data.
        It estimates the quality of each model, relative to each of the other models.

        Formula:
            $AIC = N \\ln(SSE/N) + 2k$

        where:
        - $N$ is the number of samples.
        - $SSE$ is the sum of squared errors.
        - $k$ is the number of estimated parameters.

        Args:
            data: Validation data (SysIdData or object with u, y).

        Returns:
            float: The AIC value.
        """
        return self._information_criterion(data, penalty_factor=2.0)

    def bic(self, data: Any) -> float:
        """
        Calculate the Bayesian Information Criterion (BIC).

        The BIC is a criterion for model selection among a finite set of models;
        the model with the lowest BIC is preferred. It is based, in part, on the likelihood function
        and it is closely related to the Akaike information criterion (AIC).

        Formula:
            $BIC = N \\ln(SSE/N) + k \\ln(N)$

        where:
        - $N$ is the number of samples.
        - $SSE$ is the sum of squared errors.
        - $k$ is the number of estimated parameters.

        Args:
            data: Validation data (SysIdData or object with u, y).

        Returns:
            float: The BIC value.
        """
        # For BIC, penalty is ln(N)
        # We need N first, so we can't just pass a constant penalty factor
        # unless we extract N inside _information_criterion.
        # Let's implement logic here or make _information_criterion flexible.

        # Extract u and y
        u, y = self._extract_data(data)
        N = len(y)

        return self._information_criterion(data, penalty_factor=np.log(N))

    def _extract_data(self, data: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Helper to extract u and y from data object."""
        # Case 1: Object has .u and .y attributes
        if hasattr(data, "u") and hasattr(data, "y"):
            u = np.atleast_2d(data.u)
            y = np.atleast_2d(data.y)
            return u, y

        # Case 2: Use model's input/output names
        if self.input_names and self.output_names:
            try:
                u_list = [data[name] for name in self.input_names]
                u = np.column_stack(u_list)
                y_list = [data[name] for name in self.output_names]
                y = np.column_stack(y_list)
                return u, y
            except (KeyError, TypeError):
                pass  # Fall through

        # Case 3: Try default 'u' and 'y' keys
        try:
            u = data["u"]
            y = data["y"]
            # Ensure 2D (N, 1) if 1D
            if u.ndim == 1:
                u = u.reshape(-1, 1)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            return u, y
        except (KeyError, TypeError, AttributeError):
            pass

        raise ValueError(
            "Could not extract 'u' and 'y' from data. "
            "Ensure data object has 'u'/'y' attributes, or 'u'/'y' keys, "
            "or model has input_names/output_names matching data keys."
        )

    def _information_criterion(self, data: Any, penalty_factor: float) -> float:
        """
        Calculate information criterion.

        IC = N * ln(SSE/N) + k * penalty_factor
        """
        u, y = self._extract_data(data)
        N = len(y)

        y_hat = self.simulate(u)
        e = self.residuals(y, y_hat)
        sse = self.SSE(e)

        # Number of parameters k
        # This depends on the model type.
        # We can try to use vectorize() to count parameters if available.
        try:
            if hasattr(self, "vectorize"):
                k = len(self.vectorize())
            else:
                # Fallback or raise error
                # For StateSpace: k = nx*nx + nx*nu + ny*nx + ny*nu + nx (x0)
                # For Polynomial: k = na + nb
                raise NotImplementedError("Model must implement vectorize() or provide parameter count.")
        except Exception:
            # If vectorize is not implemented or fails, try to deduce
            if hasattr(self, "nx"):  # StateSpace
                nx = self.nx
                nu = self.nu
                ny = self.ny
                k = nx * nx + nx * nu + ny * nx + ny * nu + nx  # +nx for x0
            elif hasattr(self, "na") and hasattr(self, "nb"):  # Polynomial
                k = self.na + self.nb
            else:
                k = 0  # Should not happen for implemented models

        # Handle SSE=0 (perfect fit)
        if sse <= 0:
            term1 = -np.inf
        else:
            term1 = N * np.log(sse / N)

        return term1 + k * penalty_factor

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
