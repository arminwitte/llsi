"""
ARX and FIR model identification methods.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg

from .polynomialmodel import PolynomialModel
from .sysidalgbase import SysIdAlgBase
from .sysiddata import SysIdData


class ARX(SysIdAlgBase):
    """
    AutoRegressive with eXogenous input (ARX) model identification.

    Estimates an ARX model of the form:
    A(q)y(t) = B(q)u(t-nk) + e(t)

    using least squares methods.
    """

    def __init__(
        self,
        data: SysIdData,
        y_name: Union[str, List[str]],
        u_name: Union[str, List[str]],
        settings: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ARX identification.

        Args:
            data: The system identification data.
            y_name: Name(s) of the output channel(s).
            u_name: Name(s) of the input channel(s).
            settings: Configuration dictionary.
                      - 'lstsq_method': Method for least squares ('qr', 'pinv', 'lstsq', 'svd'). Default 'qr'.
                      - 'lambda': Regularization parameter for 'qr' and 'svd'. Default 0.0.
        """
        if settings is None:
            settings = {}
        super().__init__(data, y_name, u_name, settings=settings)
        
        # ARX implementation currently supports SISO only
        if self.u.shape[1] > 1:
            raise NotImplementedError("Multiple inputs are not yet implemented for ARX.")

        if self.y.shape[1] > 1:
            raise NotImplementedError("Multiple outputs are not yet implemented for ARX.")

        self.u = self.u.ravel()
        self.y = self.y.ravel()
        self.logger = logging.getLogger(__name__)

    def ident(self, order: Tuple[int, int, int]) -> PolynomialModel:
        """
        Identify the ARX model.

        Args:
            order: A tuple (na, nb, nk) where:
                   na: Order of the A polynomial (autoregressive part).
                   nb: Order of the B polynomial (exogenous input part).
                   nk: Input delay (dead time).

        Returns:
            PolynomialModel: The identified ARX model.
        """
        na, nb, nk = order
        Phi, y = self._observations(na, nb, nk)

        lstsq_method = self.settings.get("lstsq_method", "qr")
        lmb = self.settings.get("lambda", 0.0)

        if lstsq_method == "pinv":
            theta, cov = self._lstsq_pinv(Phi, y)
        elif lstsq_method == "lstsq":
            theta, cov = self._lstsq_lstsq(Phi, y)
        elif lstsq_method == "qr":
            theta, cov = self._lstsq_qr(Phi, y, lmb)
        elif lstsq_method == "svd":
            theta, cov = self._lstsq_svd(Phi, y, lmb)
        else:
            self.logger.warning(f"Unknown lstsq_method '{lstsq_method}'. Defaulting to 'qr'.")
            theta, cov = self._lstsq_qr(Phi, y, lmb)

        self.logger.debug(f"theta:\n{theta}")

        b = theta[:nb]
        a = np.hstack(([1.0], theta[nb:]))

        mod = PolynomialModel(
            b=b,
            a=a,
            nk=nk,
            Ts=self.Ts,
            cov=cov,
            input_names=self.input_names,
            output_names=self.output_names,
        )

        return mod

    def _observations(self, na: int, nb: int, nk: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct the regression matrix Phi and target vector y.
        """
        u = self.u
        y = self.y
        nn = max(nb + nk, na)
        N = u.shape[0]
        
        # Pre-allocate Phi
        Phi = np.empty((N - nn, nb + na))
        
        # Fill Phi with lagged inputs
        for i in range(nb):
            Phi[:, i] = u[nn - i - nk : N - i - nk]
            
        # Fill Phi with lagged outputs (negative)
        for i in range(na):
            Phi[:, nb + i] = -y[nn - i - 1 : N - i - 1]

        y_ = y[nn:N]

        return Phi, y_

    @staticmethod
    def _lstsq_lstsq(Phi: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Least squares using scipy.linalg.lstsq."""
        theta, res, rank, s = scipy.linalg.lstsq(Phi, y)

        e = y - (Phi @ theta)
        var_e = np.var(e)
        # Note: inv(Phi.T @ Phi) can be unstable if Phi is ill-conditioned
        cov = var_e * scipy.linalg.inv(Phi.T @ Phi)
        return theta, cov

    @staticmethod
    def _lstsq_pinv(Phi: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Least squares using Moore-Penrose pseudoinverse."""
        theta = scipy.linalg.pinv(Phi) @ y

        e = y - (Phi @ theta)
        var_e = np.var(e)
        cov = var_e * scipy.linalg.inv(Phi.T @ Phi)
        return theta, cov

    @staticmethod
    def _lstsq_qr(Phi: np.ndarray, y: np.ndarray, lmb: float) -> Tuple[np.ndarray, np.ndarray]:
        """Least squares using QR decomposition with optional regularization."""
        # Regularization by appending rows
        Phi_ = np.vstack([Phi, lmb * np.eye(Phi.shape[1])])
        y_ = np.vstack([y.reshape(-1, 1), np.zeros((Phi.shape[1], 1))]).ravel()
        
        Q, R = scipy.linalg.qr(Phi_, mode="economic")
        theta = scipy.linalg.solve_triangular(R, Q.T @ y_)

        e = y - (Phi @ theta)
        var_e = np.var(e)
        
        # More stable covariance calculation: cov = var_e * (R.T @ R)^-1
        # (R.T @ R)^-1 = R^-1 @ R^-T
        try:
            R_inv = scipy.linalg.solve_triangular(R, np.eye(R.shape[0]))
            cov = var_e * (R_inv @ R_inv.T)
        except Exception:
             # Fallback if R is singular (though QR usually handles this, solve_triangular might fail)
            cov = var_e * scipy.linalg.pinv(R.T @ R)
            
        return theta, cov

    @staticmethod
    def _lstsq_svd(Phi: np.ndarray, y: np.ndarray, lmb: float) -> Tuple[np.ndarray, np.ndarray]:
        """Least squares using SVD with optional regularization."""
        U, s, Vh = scipy.linalg.svd(Phi, full_matrices=False)
        Sigma = np.diag(1 / s)

        if lmb > 0:
            rho = np.diag(s**2 / (s**2 + lmb))
            theta = Vh.T @ rho @ Sigma @ U.T @ y
        else:
            theta = Vh.T @ Sigma @ U.T @ y

        e = y - (Phi @ theta)
        var_e = np.var(e)
        
        if lmb > 0:
            Sigma_sqr = np.diag(s**2 / (s**2 + lmb) ** 2)
        else:
            Sigma_sqr = np.diag(1 / s**2)
            
        cov = var_e * (Vh.T @ Sigma_sqr @ Vh)
        return theta, cov

    @staticmethod
    def name() -> str:
        return "arx"


class FIR(ARX):
    """
    Finite Impulse Response (FIR) model identification.
    
    Special case of ARX where na=0.
    y(t) = B(q)u(t-nk) + e(t)
    """
    
    def __init__(
        self,
        data: SysIdData,
        y_name: Union[str, List[str]],
        u_name: Union[str, List[str]],
        settings: Optional[Dict[str, Any]] = None,
    ):
        if settings is None:
            settings = {}
        super().__init__(data, y_name, u_name, settings=settings)

    def ident(self, order: Union[Tuple[int, int], Tuple[int, int, int]]) -> PolynomialModel:
        """
        Identify the FIR model.

        Args:
            order: A tuple (nb, nk) or (na, nb, nk). 
                   If 3 elements are provided, the first element (na) is ignored and set to 0.
        """
        order_list = list(order)
        if len(order_list) == 2:
            # Assume (nb, nk) provided
            nb, nk = order_list
            order_list = [0, nb, nk]
        elif len(order_list) == 3:
            # Assume (na, nb, nk) provided, force na=0
            order_list[0] = 0
        else:
            raise ValueError("Order must be a tuple of 2 (nb, nk) or 3 (na, nb, nk) integers.")
            
        return super().ident(tuple(order_list))

    @staticmethod
    def name() -> str:
        return "fir"
