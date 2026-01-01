"""
State-space model representation.
"""

import json
import logging
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg
import scipy.signal

from .ltimodel import LTIModel
from .math import evaluate_state_space


class StateSpaceModel(LTIModel):
    """
    State Space model class.

    Represents a discrete-time system:
    x[k+1] = A x[k] + B u[k]
    y[k]   = C x[k] + D u[k]
    """

    def __init__(
        self,
        A: Optional[Union[np.ndarray, List[List[float]]]] = None,
        B: Optional[Union[np.ndarray, List[List[float]]]] = None,
        C: Optional[Union[np.ndarray, List[List[float]]]] = None,
        D: Optional[Union[np.ndarray, List[List[float]]]] = None,
        Ts: float = 1.0,
        nx: int = 0,
        nu: int = 1,
        ny: int = 1,
        nk: int = 0,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ):
        """
        Initialize StateSpaceModel.

        Args:
            A: State transition matrix.
            B: Input matrix.
            C: Output matrix.
            D: Feedthrough matrix.
            Ts: Sampling time.
            nx: Number of states (used if A is None).
            nu: Number of inputs (used if B is None).
            ny: Number of outputs (used if C is None).
            nk: Input delay (samples).
            input_names: List of input names.
            output_names: List of output names.
        """
        if input_names is None:
            input_names = []
        if output_names is None:
            output_names = []
        super().__init__(Ts=Ts, input_names=input_names, output_names=output_names)

        self.nk = nk

        # set A matrix and number of states
        if A is not None:
            self.A = np.array(A, dtype=float)
            self.nx = self.A.shape[0]
        else:
            self.nx = nx
            self.A = np.zeros((self.nx, self.nx))

        # set B matrix and number of inputs
        if B is not None:
            self.B = np.array(B, dtype=float).reshape(self.nx, -1)
            self.nu = self.B.shape[1]
        else:
            self.nu = nu
            self.B = np.zeros((self.nx, self.nu))

        # set C matrix and number of outputs
        if C is not None:
            self.C = np.array(C, dtype=float).reshape(-1, self.nx)
            self.ny = self.C.shape[0]
        else:
            self.ny = ny
            self.C = np.zeros((self.ny, self.nx))

        if D is not None:
            self.D = np.array(D, dtype=float).reshape(self.ny, self.nu)
        else:
            self.D = np.zeros((self.ny, self.nu))

        self.x_init = np.zeros((self.nx, 1))
        self.cov: Optional[np.ndarray] = None
        self.logger = logging.getLogger(__name__)

    def vectorize(self, include_init_state: bool = True) -> np.ndarray:
        """Vectorize model parameters."""
        theta = np.vstack(
            [
                self.A.reshape(-1, 1),
                self.B.reshape(-1, 1),
                self.C.reshape(-1, 1),
                self.D.reshape(-1, 1),
            ]
        )
        if include_init_state:
            if self.x_init is None:
                self.x_init = np.zeros((self.nx, 1))
            theta = np.vstack([theta, self.x_init.reshape(-1, 1)])

        return np.array(theta).ravel()

    def reshape(self, theta: np.ndarray, include_init_state: bool = True) -> None:
        """Update model parameters from vector."""
        nx = self.nx
        nu = self.nu
        ny = self.ny

        na = nx * nx
        nb = nx * nu
        nc = ny * nx
        nd = ny * nu

        self.A = theta[:na].reshape(nx, nx)
        self.B = theta[na : na + nb].reshape(nx, nu)
        self.C = theta[na + nb : na + nb + nc].reshape(ny, nx)
        self.D = theta[na + nb + nc : na + nb + nc + nd].reshape(ny, nu)

        if include_init_state:
            self.x_init = theta[na + nb + nc + nd :].reshape(nx, 1)

    def simulate(
        self, u: np.ndarray, uncertainty: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Simulate the model."""
        u = np.atleast_2d(u)
        if u.shape[0] != self.nu:
            # Try transposing if dimensions mismatch but match after transpose
            if u.shape[1] == self.nu:
                u = u.T

        # Ensure u is (nu, N) for evaluate_state_space?
        # evaluate_state_space expects u as (nu, N) or (N, nu)?
        # Looking at math.py:
        # u : (nu, N) array

        # But wait, usually we pass (N, nu) or (N,) to simulate.
        # The original code did: u = u.reshape(self.nu, -1)
        # This implies it expects input to be flattened into (nu, N) somehow?
        # If u is (N, nu), reshape(nu, -1) might scramble data if not careful.
        # If u is (N, 1), reshape(1, -1) -> (1, N). Correct.
        # If u is (N, 2), reshape(2, -1) -> (2, N). Correct ONLY if data is row-major and we want to split rows?
        # No, if u is (N, nu), we want (nu, N). u.T is better.

        # Let's assume u is (N, nu) or (N,).
        if u.ndim == 1:
            u = u.reshape(1, -1)  # (1, N)
        elif u.shape[0] == self.nu:
            pass  # Already (nu, N)
        elif u.shape[1] == self.nu:
            u = u.T  # Convert (N, nu) to (nu, N)
        else:
            # Fallback to original behavior but it's risky
            u = u.reshape(self.nu, -1)

        # Apply input delay
        if self.nk > 0:
            # Prepend nk columns of zeros
            zeros = np.zeros((self.nu, self.nk))
            u = np.hstack((zeros, u[:, : -self.nk]))

        u = np.ascontiguousarray(u)

        if self.x_init is None:
            x1 = np.zeros((self.nx, 1))
        else:
            x1 = self.x_init

        y = evaluate_state_space(
            self.A.astype(np.float64),
            self.B.astype(np.float64),
            self.C.astype(np.float64),
            self.D.astype(np.float64),
            u.astype(np.float64),
            x1.astype(np.float64),
        )
        # y is returned as (N, ny) from evaluate_state_space

        if uncertainty:
            if not hasattr(self, "cov") or self.cov is None:
                return y, None

            # Ensure u is correct shape for closure
            # We need to pass the original u (or close to it) to the closure
            # But we modified u above.
            # Let's just use the modified u, but we need to be careful about recursion.
            # Actually, evaluate_state_space is static/external, so we can just wrap that?
            # No, _propagate_uncertainty perturbs parameters (A, B, C, D) and calls func().
            # So func() needs to call simulate() or evaluate_state_space() with NEW parameters.
            # Calling self.simulate(..., uncertainty=False) is the right way.
            # But we need to pass the original input format if possible, or the processed one?
            # If we pass processed u (nu, N), simulate might try to process it again.
            # If we pass (nu, N), simulate checks:
            # if u.shape[0] == self.nu: pass. So it works.

            u_for_closure = u.copy()

            def func():
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
        """Compute frequency response."""
        A = self.A
        B = self.B
        C = self.C
        D = self.D

        z = np.exp(1j * omega * self.Ts)
        H = []

        # H(z) = C(zI - A)^-1 B + D
        # This loop is slow for many frequencies.
        # Could use scipy.signal.freqresp if we convert to ss.

        eye = np.eye(A.shape[0])
        for z_ in z:
            # Solve (zI - A) X = B -> X = (zI - A)^-1 B
            # Then H = C X + D
            try:
                X = scipy.linalg.solve(z_ * eye - A, B)
                h = C @ X + D
            except scipy.linalg.LinAlgError:
                h = np.full((self.ny, self.nu), np.nan)
            H.append(h)

        H_arr = np.array(H)

        if uncertainty:
            if not hasattr(self, "cov") or self.cov is None:
                return omega, H_arr, None, None

            def func():
                _, H_ = self.frequency_response(omega, uncertainty=False)
                H_ = H_.ravel()
                return np.concatenate([np.abs(H_), np.unwrap(np.angle(H_))])

            std = self._propagate_uncertainty(func)
            n = H_arr.size
            mag_std = std[:n].reshape(H_arr.shape)
            phase_std = std[n:].reshape(H_arr.shape)

            return omega, H_arr, mag_std, phase_std

        return omega, H_arr
        # Could use scipy.signal.freqresp if we convert to ss.

        eye = np.eye(A.shape[0])
        for z_ in z:
            # Solve (zI - A) X = B -> X = (zI - A)^-1 B
            # Then H = C X + D
            try:
                X = scipy.linalg.solve(z_ * eye - A, B)
                h = C @ X + D
            except scipy.linalg.LinAlgError:
                h = np.full((self.ny, self.nu), np.nan)
            H.append(h)

        return omega, np.array(H)

    @classmethod
    def from_PT1(cls, K: float, tauC: float, Ts: float = 1.0) -> "StateSpaceModel":
        """Create from PT1 parameters."""
        t = 2 * tauC
        tt = 1 / (Ts + t)
        b = K * Ts * tt
        a = (Ts - t) * tt

        B = [[(1 - a) * b]]
        D = [[b]]

        A = [[-a]]
        C = [[1]]

        mod = cls(A=A, B=B, C=C, D=D, Ts=Ts, nx=1)

        return mod

    def to_ss(self, continuous: bool = False, method: str = "bilinear") -> scipy.signal.StateSpace:
        """
        Convert the model to a Scipy StateSpace representation.

        Args:
            continuous: If True, convert to a continuous-time system.
            method: The method to use for discrete-to-continuous conversion if `continuous=True`.
                    Options: 'bilinear' (Tustin), 'euler' (Forward Euler).
                    Default is 'bilinear'.

        Returns:
            scipy.signal.StateSpace: The state-space representation.
        """
        if continuous:
            A, B, C, D = self._d2c(self.A, self.B, self.C, self.D, self.Ts, method=method)
            sys = scipy.signal.StateSpace(A, B, C, D)
        else:
            sys = scipy.signal.StateSpace(self.A, self.B, self.C, self.D, dt=self.Ts)
        return sys

    def d2c(self, method: str = "bilinear") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert the discrete-time model matrices to continuous-time matrices.

        Args:
            method: The method to use for conversion.
                    Options: 'bilinear' (Tustin), 'euler' (Forward Euler).
                    Default is 'bilinear'.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The continuous-time matrices (Ac, Bc, Cc, Dc).
        """
        return self._d2c(self.A, self.B, self.C, self.D, self.Ts, method=method)

    @staticmethod
    def _d2c(
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        Ts: float,
        method: str = "bilinear",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if method == "bilinear":
            return StateSpaceModel._d2c_bilinear(A, B, C, D, Ts)
        else:
            return StateSpaceModel._d2c_euler(A, B, C, D, Ts)

    @staticmethod
    def _d2c_bilinear(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, Ts: float):
        eye = np.eye(*A.shape)
        AI = scipy.linalg.inv(A + eye)
        A_ = 2.0 / Ts * (A - eye) @ AI
        B_ = 2.0 / Ts * (eye - (A - eye) @ AI) @ B
        C_ = C @ AI
        D_ = D - C @ AI @ B
        return A_, B_, C_, D_

    @staticmethod
    def _d2c_euler(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, Ts: float):
        A_ = (A - np.eye(*A.shape)) / Ts
        B_ = B / Ts
        C_ = C
        D_ = D
        return A_, B_, C_, D_

    def to_tf(self, continuous: bool = False, method: str = "bilinear") -> scipy.signal.TransferFunction:
        """
        Convert the model to a Scipy TransferFunction representation.

        Args:
            continuous: If True, convert to a continuous-time system.
            method: The method to use for discrete-to-continuous conversion if `continuous=True`.
                    Options: 'bilinear' (Tustin), 'euler' (Forward Euler).
                    Default is 'bilinear'.

        Returns:
            scipy.signal.TransferFunction: The transfer function representation.
        """
        sys = self.to_ss(continuous=continuous, method=method)
        return sys.to_tf()

    def to_zpk(self, continuous=False, method="bilinear"):
        sys = self.to_ss(continuous=continuous, method=method)
        return sys.to_zpk()

    def to_controllable_form(self):
        tf = self.to_tf()
        ss = tf.to_ss()
        return StateSpaceModel(A=ss.A, B=ss.B, C=ss.C, D=ss.D, Ts=self.Ts)

    def reduce_order(self, n: int) -> Tuple["StateSpaceModel", np.ndarray]:
        """
        Perform order reduction using balanced truncation.

        Args:
            n: New (reduced) model order.

        Returns:
            Tuple[StateSpaceModel, np.ndarray]: (Reduced model, Hankel singular values)
        """
        A = self.A
        B = self.B
        C = self.C

        if n > A.shape[0]:
            raise ValueError(f"New model order has to be <= {A.shape[0]} but is {n}")

        # controllability gramian
        W_c = scipy.linalg.solve_discrete_lyapunov(A, B @ B.T)

        # observability gramian
        W_o = scipy.linalg.solve_discrete_lyapunov(A.T, C.T @ C)

        # controllability matrix
        S = scipy.linalg.cholesky(W_c)

        # observability matrix
        R = scipy.linalg.cholesky(W_o)

        U, s, V = scipy.linalg.svd(S @ R.T)

        # truncation
        U1 = U[:, :n]
        V1 = V[:, :n]

        # balancing-free square root algorithm
        W, X = scipy.linalg.qr(S.T @ U1, mode="economic")
        Z, Y = scipy.linalg.qr(R.T @ V1, mode="economic")
        UE, sE, VE = scipy.linalg.svd(Z.T @ W)

        SigmaE = np.diag(1 / sE)
        T_l = np.sqrt(SigmaE) @ UE.T @ Z.T
        T_r = W @ VE @ np.sqrt(SigmaE)

        # apply transformation
        A_ = T_l @ A @ T_r
        B_ = T_l @ B
        C_ = C @ T_r

        return StateSpaceModel(A=A_, B=B_, C=C_, D=self.D, Ts=self.Ts), s

    @classmethod
    def from_scipy(cls, mod: Any) -> "StateSpaceModel":
        if hasattr(mod, "ss"):
            ss = mod.ss()
        elif isinstance(mod, scipy.signal.StateSpace):
            ss = mod
        else:
            ss = mod.to_ss()

        # Check for dt
        dt = getattr(ss, "dt", 1.0)
        if dt is None:
            dt = 1.0

        mod_out = cls(A=ss.A, B=ss.B, C=ss.C, D=ss.D, Ts=dt)
        return mod_out

    @classmethod
    def from_fir(cls, mod: Any) -> "StateSpaceModel":
        """Create state-space model from FIR model (PolynomialModel)."""
        nk = getattr(mod, "nk", 0)
        b_coeffs = getattr(mod, "b", np.array([1.0]))

        b = np.vstack([np.zeros((nk, 1)), b_coeffs.reshape(-1, 1)])
        n = b.ravel().shape[0] - 1

        if n < 1:
            # Trivial case
            return cls(A=[[0]], B=[[0]], C=[[0]], D=[[b[0, 0]]], Ts=mod.Ts)

        A = np.diag(np.ones((n - 1,)), k=-1)
        B = np.zeros((n, 1))
        B[0] = 1.0
        C = b[1:].reshape(1, -1)
        D = b[0]
        mod_out = cls(
            A=A,
            B=B,
            C=C,
            D=D,
            Ts=mod.Ts,
            input_names=mod.input_names,
            output_names=mod.output_names,
        )
        return mod_out

    def to_json(self, filename: Optional[str] = None) -> str:
        data = {}
        data["A"] = self.A.tolist()
        data["B"] = self.B.tolist()
        data["C"] = self.C.tolist()
        data["D"] = self.D.tolist()
        data["Ts"] = self.Ts
        try:
            data["info"] = str(self.info)
        except AttributeError:
            data["info"] = ""

        data["nx"] = self.nx
        data["nu"] = self.nu
        data["ny"] = self.ny
        data["nk"] = self.nk
        data["input_names"] = self.input_names
        data["output_names"] = self.output_names

        if self.x_init is not None:
            data["x_init"] = self.x_init.tolist()

        if self.cov is not None:
            data["cov"] = self.cov.tolist()

        if filename is not None:
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
            return ""

        return json.dumps(data, indent=4)

    @classmethod
    def from_json(cls, filename: str) -> "StateSpaceModel":
        with open(filename) as f:
            data = json.load(f)
        mod = StateSpaceModel(
            A=data["A"],
            B=data["B"],
            C=data["C"],
            D=data["D"],
            Ts=data["Ts"],
            nx=data.get("nx", 0),
            nu=data.get("nu", 1),
            ny=data.get("ny", 1),
            nk=data.get("nk", 0),
            input_names=data.get("input_names"),
            output_names=data.get("output_names"),
        )
        if "info" in data:
            mod.info = data["info"]

        if "x_init" in data:
            mod.x_init = np.array(data["x_init"])

        if "cov" in data:
            mod.cov = np.array(data["cov"])

        return mod

    def __repr__(self) -> str:
        s = f"StateSpaceModel with Ts={self.Ts}\n"
        s += f"input(s): {self.input_names}\n"
        s += f"output(s): {self.output_names}\n"
        s += f"A:\n{self.A}\n"
        s += f"B:\n{self.B}\n"
        s += f"C:\n{self.C}\n"
        s += f"D:\n{self.D}\n"
        return s

    def __str__(self) -> str:
        return self.__repr__()
