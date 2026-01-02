"""
Base class for system identification algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import scipy.signal

from .ltimodel import LTIModel
from .sysiddata import SysIdData


class SysIdAlgBase(ABC):
    """Base class for system identification algorithms."""

    def __init__(
        self,
        data: SysIdData,
        y_name: Union[str, List[str]],
        u_name: Union[str, List[str]],
        settings: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the system identification algorithm.

        Args:
            data: The system identification data.
            y_name: Name(s) of the output channel(s).
            u_name: Name(s) of the input channel(s).
            settings: Dictionary of settings for the algorithm.
        """
        self.input_names = u_name if isinstance(u_name, list) else [u_name]
        self.output_names = y_name if isinstance(y_name, list) else [y_name]

        # Extract output data
        y_names_list = [y_name] if isinstance(y_name, str) else y_name
        y_data = [data[name] for name in y_names_list]
        self.y = np.column_stack(y_data)

        # Extract input data
        u_names_list = [u_name] if isinstance(u_name, str) else u_name
        u_data = [data[name] for name in u_names_list]
        self.u = np.column_stack(u_data)

        self.Ts = data.Ts
        self.settings = settings if settings is not None else {}

    def ident(self, order: Any) -> LTIModel:
        """
        Identify the model.

        Args:
            order: The order of the model.

        Returns:
            The identified LTI model.
        """
        model = self._ident(order)

        # Calculate and store identification metrics
        model.aic = self.aic(model)
        model.bic = self.bic(model)
        model.residuals_analysis = self.compute_residuals_analysis(model)
        model.residuals = model.residuals_analysis["residuals"]

        return model

    @abstractmethod
    def _ident(self, order: Any) -> LTIModel:
        """
        Implementation of the identification algorithm.

        Args:
            order: The order of the model.

        Returns:
            The identified LTI model.
        """
        pass

    def compute_residuals_analysis(self, model: LTIModel, data: Optional[SysIdData] = None) -> Dict[str, Any]:
        """
        Compute residual analysis metrics (ACF, CCF) on validation data.

        Args:
            model: The LTI model to evaluate.
            data: SysIdData object containing validation u and y. If None, use training data.

        Returns:
            Dict containing:
                'residuals': The residuals (y - y_hat)
                'acf': Auto-correlation function of residuals (normalized)
                'ccf': Cross-correlation function of residuals and input (normalized)
                'lags': Lags for the correlations
                'conf_interval': 99% confidence interval value
        """
        if data is None:
            return _compute_residuals_analysis_arrays(model, self.u, self.y)
        return compute_residuals_analysis(model, data)

    def aic(self, model: LTIModel, data: Optional[SysIdData] = None) -> float:
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
            model: The LTI model to evaluate.
            data: Validation data (SysIdData). If None, use training data.

        Returns:
            float: The AIC value.
        """
        if data is None:
            return _information_criterion_arrays(model, self.u, self.y, penalty_factor=2.0)
        return aic(model, data)

    def bic(self, model: LTIModel, data: Optional[SysIdData] = None) -> float:
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
            model: The LTI model to evaluate.
            data: Validation data (SysIdData). If None, use training data.

        Returns:
            float: The BIC value.
        """
        if data is None:
            N = len(self.y)
            return _information_criterion_arrays(model, self.u, self.y, penalty_factor=np.log(N))
        return bic(model, data)


def compute_residuals_analysis(model: LTIModel, data: SysIdData) -> Dict[str, Any]:
    """
    Compute residual analysis metrics (ACF, CCF) on validation data.

    Args:
        model: The LTI model to evaluate.
        data: SysIdData object containing validation u and y.

    Returns:
        Dict containing:
            'residuals': The residuals (y - y_hat)
            'acf': Auto-correlation function of residuals (normalized)
            'ccf': Cross-correlation function of residuals and input (normalized)
            'lags': Lags for the correlations
            'conf_interval': 99% confidence interval value
    """
    u, y = data.to_io_data(model.input_names, model.output_names)
    return _compute_residuals_analysis_arrays(model, u, y)


def _compute_residuals_analysis_arrays(model: LTIModel, u: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    y_pred = model.simulate(u)
    residuals = y - y_pred

    # Ensure residuals is at least 2D for indexing, or handle 1D
    if residuals.ndim == 1:
        res_i = residuals
    else:
        res_i = residuals[:, 0]

    res_i_centered = res_i - np.mean(res_i)

    # ACF
    acf = scipy.signal.correlate(res_i_centered, res_i_centered, mode="full")
    if np.max(acf) > 0:
        acf = acf / np.max(acf)  # Normalize
    lags = scipy.signal.correlation_lags(len(res_i_centered), len(res_i_centered))

    # CCF (Input 0 vs Residuals)
    if u.ndim == 1:
        u_0 = u
    else:
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


def aic(model: LTIModel, data: SysIdData) -> float:
    """
    Calculate the Akaike Information Criterion (AIC).

    Args:
        model: The LTI model to evaluate.
        data: Validation data (SysIdData).

    Returns:
        float: The AIC value.
    """
    return _information_criterion(model, data, penalty_factor=2.0)


def bic(model: LTIModel, data: SysIdData) -> float:
    """
    Calculate the Bayesian Information Criterion (BIC).

    Args:
        model: The LTI model to evaluate.
        data: Validation data (SysIdData).

    Returns:
        float: The BIC value.
    """
    u, y = data.to_io_data(model.input_names, model.output_names)
    N = len(y)
    return _information_criterion_arrays(model, u, y, penalty_factor=np.log(N))


def _information_criterion(model: LTIModel, data: SysIdData, penalty_factor: float) -> float:
    u, y = data.to_io_data(model.input_names, model.output_names)
    return _information_criterion_arrays(model, u, y, penalty_factor)


def _information_criterion_arrays(model: LTIModel, u: np.ndarray, y: np.ndarray, penalty_factor: float) -> float:
    N = len(y)

    y_hat = model.simulate(u)
    e = LTIModel.residuals(y, y_hat)
    sse = LTIModel.SSE(e)

    # Number of parameters k
    try:
        if hasattr(model, "vectorize"):
            k = len(model.vectorize())
        else:
            raise NotImplementedError("Model must implement vectorize() or provide parameter count.")
    except Exception:
        if hasattr(model, "nx"):  # StateSpace
            nx = model.nx
            nu = model.nu
            ny = model.ny
            k = nx * nx + nx * nu + ny * nx + ny * nu + nx  # +nx for x0
        elif hasattr(model, "na") and hasattr(model, "nb"):  # Polynomial
            k = model.na + model.nb
        else:
            k = 0

    # Handle SSE=0 (perfect fit)
    if sse <= 0:
        term1 = -np.inf
    else:
        term1 = N * np.log(sse / N)

    return term1 + k * penalty_factor

    @staticmethod
    @abstractmethod
    def name() -> str:
        """
        Get the name of the algorithm.

        Returns:
            The name of the algorithm.
        """
        pass

    @staticmethod
    def _sse(y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Calculate the Sum of Squared Errors (SSE).

        Args:
            y: Measured output.
            y_hat: Predicted output.

        Returns:
            The SSE.
        """
        e = y - y_hat
        # Handle potential overflows/invalids gracefully if needed, though usually we want to know.
        # Original code used errstate ignore.
        with np.errstate(over="ignore", invalid="ignore"):
            # e is (N, ny)
            # We want sum of squared errors over all samples and channels.
            # e.T @ e would be (ny, ny). Trace of that is sum of squares.
            # Or just np.sum(e**2)

            # Original: sse = e.T @ e; return np.sum(sse)
            # If e is (N, 1), e.T @ e is scalar.
            # If e is (N, ny), e.T @ e is (ny, ny). Sum of that is sum of all cross terms?
            # Usually SSE is sum(e_i^2).
            # np.sum(e**2) is safer and clearer.

            sse = np.sum(e**2)

        return float(sse)
