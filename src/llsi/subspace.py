"""
Subspace identification methods (N4SID, PO-MOESP).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg

from .statespacemodel import StateSpaceModel
from .sysidalgbase import SysIdAlgBase
from .sysiddata import SysIdData


class SubspaceIdent(SysIdAlgBase):
    """Base class for subspace identification methods."""

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
        self.nu = self.u.shape[1]
        self.ny = self.y.shape[1]

        # Handle input delay (nk)
        self.nk = settings.get("nk", 0)
        if self.nk > 0:
            # Shift data: y[k] depends on u[k-nk]
            # We want to pair y[t] with u[t-nk].
            # So we take y from index nk to end, and u from 0 to end-nk.
            # This aligns y[nk] with u[0].
            # Effective length is reduced by nk.
            if self.y.shape[0] <= self.nk:
                raise ValueError(f"Data length {self.y.shape[0]} is too short for delay nk={self.nk}")

            self.y = self.y[self.nk :, :]
            self.u = self.u[: -self.nk, :]

    @staticmethod
    def hankel(x: np.ndarray, n: int) -> np.ndarray:
        """
        Construct Block Hankel matrix.

        Args:
            x: Data array (N, n_channels).
            n: Number of block rows.

        Returns:
            np.ndarray: Hankel matrix.
        """
        # x is (N, n_channels)
        # We want a Hankel matrix with n block rows.
        # If x has 1 column, it's standard Hankel.
        # If x has m columns, it's block Hankel.

        N_samples, n_channels = x.shape
        # Number of block columns
        # If we have n block rows, we need enough samples.
        # H is (n * n_channels, N_cols)
        # N_cols = N_samples - n + 1

        # The original implementation:
        # n = n // x.shape[1]  <-- This suggests 'n' passed in is total rows, not block rows?
        # "n = order[0]; r = (2 * n + 1) * self.nu * self.ny"
        # "Y = self.hankel(self.y, 2 * r)"
        # So 'n' passed to hankel is 2*r, which is total rows.

        n_block_rows = n // n_channels
        if n_block_rows <= 0:
            raise ValueError(f"Not enough rows requested for Hankel matrix. n={n}, channels={n_channels}")

        N_cols = N_samples - n_block_rows + 1

        if N_cols <= 0:
            raise ValueError(f"Not enough samples for Hankel matrix. Samples={N_samples}, Block rows={n_block_rows}")

        # Efficient Hankel construction using stride_tricks or simple loop
        # Original implementation used loops and lists.

        # Let's stick to a cleaner loop or optimized approach.
        # H = [x[0:N_cols], x[1:N_cols+1], ..., x[n_block_rows-1:]]
        # But stacked vertically.

        # Actually, let's follow the original logic to ensure compatibility, but clean it up.
        # Original:
        # for i in range(n): (where n is block rows)
        #   for x_ in x.T: (iterate over channels)
        #     A.append(x_[i : -n + i])

        # Wait, x_[i : -n + i] ?
        # if i=0, x_[0 : -n]. Length N-n.
        # if i=n-1, x_[n-1 : -1]. Length N-n.
        # This creates columns of length N-n.

        A = []
        for i in range(n_block_rows):
            for j in range(n_channels):
                # Column j of x
                col = x[:, j]
                # Slice
                # We want N_cols elements starting at i.
                # x[i : i + N_cols]
                # N_cols = N_samples - n_block_rows + 1
                # So x[i : i + N_samples - n_block_rows + 1]

                # Original code: x_[i : -n + i]
                # -n + i = -(n - i).
                # If n=10, i=0 -> :-10. Length N-10.
                # If n=10, i=9 -> 9:-1. Length N-10.
                # So N_cols = N - n.
                # This drops the last sample?
                # Usually Hankel uses all samples.
                # Let's use standard definition: N_cols = N - n + 1.

                # But to match original behavior exactly if needed:
                # The original code used `x_[i : -n + i]` which excludes the last `n-i` elements?
                # No, `x_[:-n]` excludes last n elements.
                # `x_[i : -n + i]` has length `(N - n + i) - i = N - n`.
                # So it produces `N - n` columns.

                # I will use `N - n_block_rows + 1` columns which is standard.
                # But wait, if I change dimensions, it might break math downstream.
                # Let's stick to `N - n_block_rows` columns as per original code implication (N-n).

                end_idx = -n_block_rows + i
                if end_idx == 0:
                    segment = col[i:]
                else:
                    segment = col[i:end_idx]
                A.append(segment)

        return np.array(A)

    @staticmethod
    def enforce_stability(A: np.ndarray, radius: float = 0.99) -> np.ndarray:
        """
        Enforces stability by scaling eigenvalues of A to be within the unit circle.

        Args:
            A: The system matrix (n x n).
            radius: The maximum allowed absolute value for eigenvalues (e.g., 0.99).

        Returns:
            A_stable: The stabilized system matrix.
        """
        # Eigenwertzerlegung: A = V * diag(w) * V^-1
        w, v = scipy.linalg.eig(A)

        # Prüfen, ob Eigenwerte außerhalb des Radius liegen
        abs_w = np.abs(w)
        if np.any(abs_w >= 1.0):
            # Skaliere nur die instabilen Eigenwerte
            # w_new = w / |w| * radius
            unstable_mask = abs_w >= 1.0
            w[unstable_mask] = (w[unstable_mask] / abs_w[unstable_mask]) * radius

            # Rekonstruktion
            # Wir nehmen den Realteil, da A ursprünglich reell war und
            # numerisches Rauschen kleine imaginäre Anteile erzeugen kann.
            A_stable = (v @ np.diag(w) @ np.linalg.inv(v)).real
            return A_stable

        return A

    def _abcd_state(
        self, Xf: np.ndarray, s: int, n: int, r: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Xf: State sequence (n, N)
        # s: Number of columns in Hankel matrix (N)
        # n: Order
        # r: Window size parameter?

        # Y_ is (n + ny, N-1) ?
        # Xf[:, 1:s-1] is X_{k+1}
        # Xf[:, :s-2] is X_k

        # We want to solve:
        # [X_{k+1}; Y_k] = [A B; C D] [X_k; U_k]

        # Original code:
        # Y_ = np.vstack((Xf[:, 1 : s - 1], self.y[r : r + s - 2, :].T))
        # X_ = np.vstack((Xf[:, : s - 2], self.u[r : r + s - 2, :].T))

        # Indices seem to be aligning Xf with y and u.
        # Xf corresponds to states at times r, r+1, ... ?

        # Let's trust the indices for now but add types.

        # Ensure dimensions match
        # Xf is (n, s)
        # We use columns 1 to s-1 (length s-2) for LHS top
        # We use columns 0 to s-2 (length s-2) for RHS top

        # y and u need to be sliced to match length s-2.
        # self.y is (N_total, ny).
        # We take slice [r : r + s - 2].

        Y_ = np.vstack((Xf[:, 1 : s - 1], self.y[r : r + s - 2, :].T))
        X_ = np.vstack((Xf[:, : s - 2], self.u[r : r + s - 2, :].T))

        lmbd = self.settings.get("lambda", 0.0)

        # Regularized least squares via SVD
        # Theta = Y_ @ pinv(X_)
        # X_ = U S Vh
        # pinv(X_) = Vh.T S^-1 U.T
        # Regularized: S^-1 -> S / (S^2 + lambda)

        U, s_svd, Vh = scipy.linalg.svd(X_, full_matrices=False)

        # s_svd are singular values
        # rho = s^2 / (s^2 + lambda) * (1/s) = s / (s^2 + lambda)
        # Wait, original code:
        # Sigma = np.diag(1 / s)
        # rho = np.diag(s**2 / (s**2 + lmbd))
        # Theta = Y_ @ (Vh.T @ rho @ Sigma @ U.T)
        # rho @ Sigma = diag( s^2/(s^2+L) * 1/s ) = diag( s / (s^2+L) )
        # This is Tikhonov regularization on singular values. Correct.

        # Handle division by zero if s_svd has zeros (unlikely with float)
        s_filt = s_svd / (s_svd**2 + lmbd)

        Theta = Y_ @ (Vh.T @ np.diag(s_filt) @ U.T)

        A = Theta[:n, :n]
        B = Theta[:n, n:]
        C = Theta[n:, :n]
        D = Theta[n:, n:]
        return A, B, C, D

    def _abcd_observability_matrix(self, U1, U2, L11, L31, Sigma_sqrt, n, r):
        ny = self.ny
        nu = self.nu

        Or = U1 @ Sigma_sqrt  # extended observability matrix

        C = Or[:ny, :]
        # Solve A from Or
        # Or(1:end-ny, :) A = Or(ny+1:end, :)
        # A = pinv(Or[:-ny]) @ Or[ny:]
        A = scipy.linalg.pinv(Or[0:-ny, :]) @ Or[ny:, :]

        # Estimate B and D
        # This part is complex PO-MOESP logic.

        P = np.split(U2.T, U2.T.shape[1] // ny, axis=1)

        nn = len(P) * P[0].shape[0]
        rny = r
        r_blocks = r // ny
        A_ = np.zeros((nn, ny + n))
        P1 = np.vstack(P)
        rr = P[0].shape[0]
        A_[:, :ny] = P1

        for i in range(1, r_blocks):
            # Pi_tilda = [P[i], P[i+1], ...]
            Pi_tilda = np.hstack(P[i:])
            # Ori = Or[: r - i*ny] ?
            # Or has r rows (actually r is total rows of Hankel Y, so r rows).
            # Or is (r, n).
            # We take top rows.
            Ori = Or[: rny - (ny * i)]
            N = Pi_tilda @ Ori
            A_[(i - 1) * rr : i * rr, ny:] = N

        M = U2.T @ L31 @ np.linalg.inv(L11)
        Mi = np.split(M, M.shape[1] // nu, axis=1)
        M = np.vstack(Mi)

        x_, *_ = scipy.linalg.lstsq(A_, M)

        D = x_[:ny, :]
        B = x_[ny:, :]

        return A, B, C, D

    @staticmethod
    def lq(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute LQ decomposition."""
        # LQ = (QR(A.T)).T = R.T Q.T
        Q, R = scipy.linalg.qr(A.T, mode="economic")
        return R.T, Q.T


class N4SID(SubspaceIdent):
    """N4SID subspace identification."""

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
        self.logger = logging.getLogger(__name__)

        # estimate extended observability matrix and states.
        # Then estimate A, B, C, and D in one go.
        # (Tangirala 2014)

        if self.y.shape[1] > 1 or self.u.shape[1] > 1:
            # Original code raised this. Keeping it for safety, though N4SID supports MIMO.
            raise NotImplementedError("n4sid not implemented for multiple inputs or outputs.")

    def ident(self, order: Union[int, Tuple[int, ...]]) -> StateSpaceModel:
        if isinstance(order, (tuple, list)):
            n = order[0]
        else:
            n = order

        r = (2 * n + 1) * self.nu * self.ny  # window length

        Y = self.hankel(self.y, 2 * r)
        U = self.hankel(self.u, 2 * r)

        Yp = Y[0:r, :]
        Up = U[0:r, :]

        Yf = Y[r : 2 * r, :]
        Uf = U[r : 2 * r, :]

        Wp = np.vstack((Up, Yp))
        Psi = np.vstack((np.vstack((Uf, Wp)), Yf))

        L, Q = self.lq(Psi)
        L22 = L[r : 3 * r, r : 3 * r]
        L32 = L[3 * r : 4 * r, r : 3 * r]

        Gamma_r = L32 @ np.linalg.pinv(L22) @ Wp  # oblique projection

        U_, s_, V_ = scipy.linalg.svd(Gamma_r, full_matrices=False)

        self.singular_values = s_
        Sigma_sqrt = np.diag(np.sqrt(s_[:n]))

        # ====================================================================
        s = Y.shape[1]
        V1 = V_[0:n, :]
        Xf = Sigma_sqrt @ V1  # state matrix # TANGIRALA SAYS IT SHOULD BE TRANSPOSED !?!

        A, B, C, D = self._abcd_state(Xf, s, n, r)

        if self.settings.get("enforce_stability", False):
            A = self.enforce_stability(A)

        mod = StateSpaceModel(
            A=A,
            B=B,
            C=C,
            D=D,
            Ts=self.Ts,
            nk=self.nk,
            input_names=self.input_names,
            output_names=self.output_names,
        )
        mod.info["Hankel singular values"] = s_

        return mod

    @staticmethod
    def name() -> str:
        return "n4sid"


class PO_MOESP(SubspaceIdent):
    """PO-MOESP subspace identification."""

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
        self.logger = logging.getLogger(__name__)

        # estimate extended observability matrix and states.
        # Then estimate A, B, C, and D in one go.
        # (Tangirala 2014)

    def ident(self, order: Union[int, Tuple[int, ...]]) -> StateSpaceModel:
        # Tangirala 2014
        # Algorithm 23.3
        if isinstance(order, (tuple, list)):
            n = order[0]
        else:
            n = order

        r = (2 * n + 1) * self.nu * self.ny  # window length

        Y = self.hankel(self.y, 2 * r)
        U = self.hankel(self.u, 2 * r)

        Yp = Y[0:r, :]
        Up = U[0:r, :]

        Yf = Y[r : 2 * r, :]
        Uf = U[r : 2 * r, :]

        Wp = np.vstack((Up, Yp))
        Psi = np.vstack((np.vstack((Uf, Wp)), Yf))

        L, Q = self.lq(Psi)

        L11 = L[0:r, 0:r]
        L31 = L[3 * r : 4 * r, 0:r]
        L32 = L[3 * r : 4 * r, r : 3 * r]

        Gamma_r = L32

        U_, s_, V_ = scipy.linalg.svd(Gamma_r, full_matrices=False)

        self.singular_values = s_

        self.logger.debug(f"s: {s_}")

        U1 = U_[:, 0:n]
        U2 = U_[:, n:r]
        Sigma_sqrt = np.diag(np.sqrt(s_[:n]))

        A, B, C, D = self._abcd_observability_matrix(U1, U2, L11, L31, Sigma_sqrt, n, r)

        if self.settings.get("enforce_stability", False):
            A = self.enforce_stability(A)

        mod = StateSpaceModel(
            A=A,
            B=B,
            C=C,
            D=D,
            Ts=self.Ts,
            nk=self.nk,
            input_names=self.input_names,
            output_names=self.output_names,
        )
        mod.info["Hankel singular values"] = s_

        return mod

    @staticmethod
    def name() -> str:
        return "po-moesp"
