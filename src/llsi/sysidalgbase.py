"""
Base class for system identification algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np

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

    @abstractmethod
    def ident(self, order: Any) -> LTIModel:
        """
        Identify the model.

        Args:
            order: The order of the model.

        Returns:
            The identified LTI model.
        """
        pass

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
