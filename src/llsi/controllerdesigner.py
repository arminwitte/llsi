from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import scipy.signal
from numpy.polynomial import Polynomial
from scipy.signal import TransferFunction


class ControllerDesigner(ABC):
    """
    Abstract base class for stable inversion controller design methods.
    """

    def __init__(self, sys: TransferFunction):
        """
        Initialize the designer.

        Args:
            sys: Transfer function of the system to be controlled
        """
        self.sys = sys
        self.designed_tf: Optional[TransferFunction] = None
        self.Ts = sys.dt

    @abstractmethod
    def design(self) -> TransferFunction:
        """
        Design the controller using the specific inversion method.
        Must be implemented by concrete classes.
        """
        pass

    def add_low_pass(self, N: int = 1, Wn: Optional[float] = None) -> TransferFunction:
        """
        Add Butterworth low-pass filter to the designed controller.

        Args:
            N: Filter order (default: 1)
            Wn: Normalized cutoff frequency (default: 0.5)
        """
        if self.designed_tf is None:
            raise ValueError("Controller must be designed before adding low-pass filter")

        Wn = 0.5 if Wn is None else Wn
        b, a = scipy.signal.butter(N, Wn)

        num = Polynomial(np.flip(self.designed_tf.num)) * Polynomial(np.flip(b))
        den = Polynomial(np.flip(self.designed_tf.den)) * Polynomial(np.flip(a))

        self.designed_tf = self._create_tf(num, den)
        return self.designed_tf

    def _extract_roots_gain(
        self, acceptable_threshold: float = 0.99
    ) -> Tuple[List[complex], List[complex], List[complex], float]:
        """Extract and categorize system zeros based on their magnitude."""
        sys_ = self.sys.to_zpk()
        zeros = np.array(sys_.zeros)
        poles = np.array(sys_.poles)

        mask = np.abs(zeros) < acceptable_threshold
        acceptable_sys_zeros = zeros[mask].tolist()
        unacceptable_sys_zeros = zeros[~mask].tolist()

        return acceptable_sys_zeros, unacceptable_sys_zeros, poles.tolist(), sys_.gain

    def _polynomials(self) -> Tuple[Polynomial, Polynomial, Polynomial]:
        """Compute system polynomials from roots."""
        acceptable_sys_zeros, unacceptable_sys_zeros, sys_poles, gain = self._extract_roots_gain()

        B_a = Polynomial.fromroots(acceptable_sys_zeros) if acceptable_sys_zeros else Polynomial([1.0])
        B_u = Polynomial.fromroots(unacceptable_sys_zeros) if unacceptable_sys_zeros else Polynomial([1.0])
        A = Polynomial.fromroots(sys_poles) * (1 / gain)

        return B_a, B_u, A

    def _create_tf(self, num: Polynomial, den: Polynomial) -> TransferFunction:
        """Create a TransferFunction from numerator and denominator polynomials."""
        return TransferFunction(np.flip(num.coef), np.flip(den.coef), dt=self.Ts)

    @staticmethod
    def _rel_deg(A: Polynomial, B: Polynomial) -> int:
        """Calculate relative degree between two polynomials."""
        return A.degree() - B.degree()

    @staticmethod
    def _rel_deg_poly(q: int) -> Polynomial:
        """Create a polynomial z^q."""
        coef = np.zeros(q + 1)
        coef[-1] = 1.0
        return Polynomial(coef)


class NPZICDesigner(ControllerDesigner):
    """
    Designer for Non-minimum Phase Zero Ignore Controller (NPZIC) method.

    References:
        Ohnishi, W., & Fujimoto, H. (2018). Perfect tracking control method by multirate
        feedforward and state trajectory generation based on time axis reversal.
    """

    def design(self) -> TransferFunction:
        """
        Design controller by ignoring non-minimum phase zeros.
        """
        B_a, B_u, A = self._polynomials()

        q = self._rel_deg(A, B_a)
        z_q = self._rel_deg_poly(q)

        num = A
        den = z_q * B_a * np.sum(B_u.coef)

        self.designed_tf = self._create_tf(num, den)
        return self.designed_tf


class ZPETCDesigner(ControllerDesigner):
    """
    Designer for Zero Phase Error Tracking Controller (ZPETC) method.

    References:
        Tomizuka, M. (1987). Zero phase error tracking algorithm for digital control.
        Journal of Dynamic Systems, Measurement, and Control, 109(1), 65-68.
    """

    def design(self) -> TransferFunction:
        """
        Design controller using zero phase error tracking method.
        """
        B_a, B_u, A = self._polynomials()
        B_u_ast = Polynomial(np.flip(B_u.coef))

        q = self._rel_deg(A * B_u_ast, B_a)
        z_q = self._rel_deg_poly(q)

        num = A * B_u_ast
        den = z_q * B_a * np.sum(B_u.coef) ** 2

        self.designed_tf = self._create_tf(num, den)
        return self.designed_tf


class ZMETCDesigner(ControllerDesigner):
    """
    Designer for Zero Magnitude Error Tracking Controller (ZMETC) method.

    References:
        To be added with implementation.
    """

    def design(self) -> TransferFunction:
        """
        Design controller using zero magnitude error tracking method.
        """
        # Implementation needed
        raise NotImplementedError("ZMETC design method not yet implemented")


def create_designer(sys: TransferFunction, method: str = "npzic") -> ControllerDesigner:
    """
    Factory function to create the appropriate controller designer.

    Args:
        sys: Transfer function of the system
        method: Design method ('npzic', 'zpetc', or 'zmetc')

    Returns:
        StableInversionDesigner: Appropriate designer instance
    """
    designers = {"npzic": NPZICDesigner, "zpetc": ZPETCDesigner, "zmetc": ZMETCDesigner}

    designer_class = designers.get(method.lower())
    if designer_class is None:
        raise ValueError(f"Unknown design method: {method}. Must be one of {list(designers.keys())}")

    return designer_class(sys)
