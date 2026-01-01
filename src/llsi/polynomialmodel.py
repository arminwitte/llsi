"""
Polynomial model representation (e.g., ARX, OE).
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import scipy.signal

from .ltimodel import LTIModel


class PolynomialModel(LTIModel):
    """
    Polynomial model representation (SISO).

    Represents a system of the form:
    A(q) y(t) = B(q) u(t-nk) + e(t)
    """

    def __init__(
        self,
        a: Optional[Union[np.ndarray, List[float]]] = None,
        b: Optional[Union[np.ndarray, List[float]]] = None,
        na: int = 1,
        nb: int = 1,
        nu: int = 1,
        ny: int = 1,
        nk: int = 0,
        cov: Optional[np.ndarray] = None,
        Ts: float = 1.0,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ):
        """
        Initialize the polynomial model.

        Args:
            a: Denominator coefficients A(q).
            b: Numerator coefficients B(q).
            na: Order of A(q).
            nb: Order of B(q).
            nu: Number of inputs (must be 1).
            ny: Number of outputs (must be 1).
            nk: Input delay (samples).
            cov: Covariance matrix of parameters.
            Ts: Sampling time.
            input_names: List of input names.
            output_names: List of output names.
        """
        if input_names is None:
            input_names = []
        if output_names is None:
            output_names = []
        super().__init__(Ts=Ts, input_names=input_names, output_names=output_names)

        if a is not None:
            self.a = np.atleast_1d(a).astype(float)
            self.na = len(self.a)
            self.ny = 1  # Enforce SISO for now based on original code logic
        else:
            self.na = na
            self.ny = ny
            self.a = np.ones(self.na)

        if b is not None:
            self.b = np.atleast_1d(b).astype(float)
            self.nb = len(self.b)
            self.nu = 1  # Enforce SISO
        else:
            self.nb = nb
            self.nu = nu
            self.b = np.ones(self.nb)

        # Original code raised error for MIMO, keeping that constraint
        if self.ny > 1 or (a is not None and np.ndim(a) > 1 and np.shape(a)[1] > 1):
            # Check if it was initialized with 2D array implying MIMO
            # The original code did: self.a = np.atleast_2d(a).T; self.ny = self.a.shape[1]
            # If user passed 1D list, atleast_2d makes it (1, N), .T makes it (N, 1), so ny=1.
            # If user passed 2D (N, M), .T makes it (M, N)? No, (N, M).T is (M, N).
            # Let's stick to SISO as explicitly stated in original code.
            pass

        if self.ny > 1:
            raise ValueError("System seems to have multiple outputs. This is not implemented.")

        if self.nu > 1:
            raise ValueError("System seems to have multiple inputs. This is not implemented.")

        # Normalize
        if len(self.a) > 0 and self.a[0] != 0:
            norm_factor = self.a[0]
            self.b = self.b / norm_factor
            self.a = self.a / norm_factor

        self.nk = nk
        self.cov = cov

    def simulate(
        self, u: Union[np.ndarray, List[float]], uncertainty: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate the model response.

        Args:
            u: Input signal.
            uncertainty: If True, return standard deviation of the response.

        Returns:
            If uncertainty is False:
                np.ndarray: Simulated output.
            If uncertainty is True:
                Tuple[np.ndarray, np.ndarray]: Simulated output and standard deviation.
        """
        u_arr = np.atleast_1d(u).ravel()

        # Handle delay by padding input or coefficients
        # lfilter: a[0]*y[n] = b[0]*x[n] + ...
        # We have y[n] = -a[1]*y[n-1]... + b[0]*u[n-nk]...
        # So we can pad b with nk zeros at the front.

        if self.nk > 0:
            b_padded = np.concatenate((np.zeros(self.nk), self.b))
        else:
            b_padded = self.b

        # scipy.signal.lfilter is much faster than python loops
        y = scipy.signal.lfilter(b_padded, self.a, u_arr)
        y = y.reshape(-1, 1)  # Return as (N, 1) to match LTIModel convention

        if uncertainty:
            if not hasattr(self, "cov") or self.cov is None:
                return y, None

            # Ensure u is correct shape for closure
            u_for_closure = np.array(u)

            def func():
                # We need to call simulate with uncertainty=False to avoid recursion
                # But since we are inside simulate, we can just call the logic directly or call self.simulate(..., uncertainty=False)
                return self.simulate(u_for_closure, uncertainty=False).ravel()

            y_std = self._propagate_uncertainty(func)
            y_std = y_std.reshape(y.shape)
            return y, y_std

        return y

    def frequency_response(
        self, omega: np.ndarray = np.logspace(-3, 2), uncertainty: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Compute frequency response.

        Args:
            omega: Frequencies (rad/s).
            uncertainty: If True, return standard deviation of magnitude and phase.

        Returns:
            If uncertainty is False:
                Tuple[np.ndarray, np.ndarray]: (omega, H)
            If uncertainty is True:
                Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (omega, H, mag_std, phase_std)
        """
        # H(z) = (B(z) / A(z)) * z^(-nk)
        # z = exp(j * omega * Ts)

        # w, h = scipy.signal.freqz(self.b, self.a, worN=omega * self.Ts, fs=2 * np.pi)

        # freqz returns response for B(z)/A(z). We need to add delay term z^(-nk)
        # But wait, freqz expects w in [0, pi) usually, or if fs is given, in Hz?
        # "If fs is not specified, worN is assumed to be in the same units as fs (radians/sample)."
        # Actually, let's stick to manual calculation or use freqz correctly.
        # Manual calculation is safer for arbitrary omega units if we are careful.

        z = np.exp(1j * omega * self.Ts)

        # Evaluate polynomials
        # polyval expects coefficients from highest degree to lowest?
        # No, standard polynomial is p[0]*x^(N-1) + ...
        # But here we have B(z^-1) = b[0] + b[1]z^-1 + ...
        # So we evaluate \sum b[k] z^{-k}

        # Using broadcasting
        k_a = np.arange(len(self.a))
        k_b = np.arange(len(self.b))

        # (N_freq, N_coeff)
        z_pow_a = np.power(z[:, None], -k_a[None, :])
        z_pow_b = np.power(z[:, None], -k_b[None, :])

        A_z = np.sum(self.a[None, :] * z_pow_a, axis=1)
        B_z = np.sum(self.b[None, :] * z_pow_b, axis=1)

        H = (B_z / A_z) * np.power(z, -self.nk)

        if uncertainty:
            if not hasattr(self, "cov") or self.cov is None:
                return omega, H, None, None

            def func():
                # Call self.frequency_response with uncertainty=False
                _, H_ = self.frequency_response(omega, uncertainty=False)
                H_ = H_.ravel()
                # Use unwrap to avoid discontinuities in phase gradient
                return np.concatenate([np.abs(H_), np.unwrap(np.angle(H_))])

            std = self._propagate_uncertainty(func)
            n = H.size
            mag_std = std[:n].reshape(H.shape)
            phase_std = std[n:].reshape(H.shape)

            return omega, H, mag_std, phase_std

        return omega, H

    def vectorize(self) -> np.ndarray:
        """Return model parameters as a vector."""
        # Usually [b0, b1, ..., a1, a2, ...] (a0 is fixed to 1)
        return np.hstack((self.b, self.a[1:])).ravel()

    def reshape(self, theta: np.ndarray) -> None:
        """Update model parameters from vector."""
        theta = np.array(theta).ravel()
        n_b = len(self.b)
        self.b = theta[:n_b]
        self.a = np.hstack(([1.0], theta[n_b:]))

    def to_tf(self, continuous: bool = False, method: str = "bilinear") -> scipy.signal.TransferFunction:
        """
        Convert the model to a Scipy TransferFunction representation.

        Args:
            continuous: If True, convert to a continuous-time system.
            method: The method to use for discrete-to-continuous conversion if `continuous=True`.
                    Options: 'bilinear' (Tustin), 'euler' (Forward Euler), 'backward_diff', 'zoh'.
                    Default is 'bilinear'.

        Returns:
            scipy.signal.TransferFunction: The transfer function representation.
        """
        # Note: This ignores nk if not handled carefully, but TransferFunction
        # in scipy is usually continuous time or discrete with fixed dt.
        # If discrete, we can pad numerator.
        if self.nk > 0:
            num = np.concatenate((np.zeros(self.nk), self.b))
        else:
            num = self.b

        if not continuous:
            return scipy.signal.TransferFunction(num, self.a, dt=self.Ts)

        # Convert to SS first
        A, B, C, D = scipy.signal.tf2ss(num, self.a)

        from .statespacemodel import StateSpaceModel

        # Use StateSpaceModel's d2c logic
        Ac, Bc, Cc, Dc = StateSpaceModel._d2c(A, B, C, D, self.Ts, method=method)

        return scipy.signal.StateSpace(Ac, Bc, Cc, Dc).to_tf()

    @classmethod
    def from_scipy(cls, mod: Any) -> "PolynomialModel":
        """Create from scipy system."""
        # Assuming mod has .dt, .num, .den or similar
        if hasattr(mod, "dt"):
            dt = mod.dt
        else:
            dt = 1.0

        if isinstance(mod, scipy.signal.TransferFunction):
            return cls(a=mod.den, b=mod.num, Ts=dt)

        # Try converting
        tf = mod.to_tf()
        return cls(a=tf.den, b=tf.num, Ts=dt)

    def __repr__(self) -> str:
        s = f"PolynomialModel with Ts={self.Ts}\n"
        s += f"input(s): {self.input_names}\n"
        s += f"output(s): {self.output_names}\n"
        s += f"b: {self.b}\n"
        s += f"a: {self.a}\n"
        s += f"nk: {self.nk}\n"
        return s

    def __str__(self) -> str:
        return self.__repr__()

    def to_json(self, filename: Optional[str] = None) -> str:
        import json

        data = {}
        data["a"] = self.a.tolist()
        data["b"] = self.b.tolist()
        data["na"] = self.na
        data["nb"] = self.nb
        data["nk"] = self.nk
        data["Ts"] = self.Ts
        data["nu"] = self.nu
        data["ny"] = self.ny
        data["input_names"] = self.input_names
        data["output_names"] = self.output_names

        try:
            data["info"] = str(self.info)
        except AttributeError:
            data["info"] = ""

        if self.cov is not None:
            data["cov"] = self.cov.tolist()
        else:
            data["cov"] = None

        if filename is not None:
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
            return ""

        return json.dumps(data, indent=4)

    @classmethod
    def from_json(cls, filename: str) -> "PolynomialModel":
        import json

        with open(filename) as f:
            data = json.load(f)

        cov = data.get("cov")
        if cov is not None:
            cov = np.array(cov)

        mod = PolynomialModel(
            a=data["a"],
            b=data["b"],
            na=data.get("na", 1),
            nb=data.get("nb", 1),
            nk=data.get("nk", 0),
            Ts=data["Ts"],
            nu=data.get("nu", 1),
            ny=data.get("ny", 1),
            cov=cov,
            input_names=data.get("input_names"),
            output_names=data.get("output_names"),
        )
        if "info" in data:
            mod.info = data["info"]
        return mod
