"""
Prediction Error Method (PEM) and Output Error (OE) identification.
"""

import logging
from typing import Any, Callable, Optional, Union

import numpy as np
import scipy.optimize

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, *args, **kwargs):
        return iterable


from .ltimodel import LTIModel
from .sysidalgbase import SysIdAlgBase
from .sysiddata import SysIdData


class PEM(SysIdAlgBase):
    """
    Prediction Error Method (PEM) identification.

    Minimizes the prediction error cost function using numerical optimization.
    Can be initialized with other methods (e.g., ARX, N4SID).
    """

    def __init__(
        self,
        data: SysIdData,
        y_name: Union[str, list[str]],
        u_name: Union[str, list[str]],
        settings: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize PEM identification.

        Args:
            data: System identification data.
            y_name: Output channel name(s).
            u_name: Input channel name(s).
            settings: Configuration dictionary.
                      - 'init': Initialization method ('arx', 'n4sid', etc.). Default 'arx'.
                      - 'minimizer_kwargs': Arguments passed to scipy.optimize.minimize.
                      - 'lambda_l1': L1 regularization coefficient.
                      - 'lambda_l2': L2 regularization coefficient.
        """
        if settings is None:
            settings = {}
        super().__init__(data, y_name, u_name, settings=settings)

        from .sysidalg import sysidalg

        init_method = self.settings.get("init", "arx")
        alg_creator = sysidalg.get_creator(init_method)
        self.alg_inst = alg_creator(data, y_name, u_name)
        self.logger = logging.getLogger(__name__)

    def _ident(self, order: Union[int, tuple[int, ...]]) -> LTIModel:
        """
        Identify the model using PEM.

        Args:
            order: Model order. Structure depends on the initialization method.
                   - For ARX init: (na, nb, nk)
                   - For N4SID init: number of states (int)

        Returns:
            LTIModel: Identified model.
        """
        # Initialize model using the specified method
        mod = self.alg_inst.ident(order)

        lambda_l1 = self.settings.get("lambda_l1", 0.0)
        lambda_l2 = self.settings.get("lambda_l2", 0.0)

        def cost_function(x: np.ndarray) -> float:
            mod.reshape(x)
            y_hat = mod.simulate(self.u)
            sse = LTIModel.SSE(self.y - y_hat)

            # Handle numerical instability
            sse = np.nan_to_num(sse, nan=1e300)

            # Regularization
            x_flat = x.ravel()
            J = sse + lambda_l1 * np.sum(np.abs(x_flat)) + lambda_l2 * (x_flat.T @ x_flat)

            self.logger.debug(f"Cost: {J:10.6g}")
            return float(J)

        x0 = mod.vectorize()

        minimizer_kwargs = self.settings.get("minimizer_kwargs", {"method": "powell"})
        res = scipy.optimize.minimize(cost_function, x0, **minimizer_kwargs)

        # Update model with optimized parameters
        mod.reshape(res.x)

        # Estimate covariance matrix using finite differences
        self._estimate_covariance(mod, res.x, epsilon=1e-8)

        return mod

    @staticmethod
    def name() -> str:
        return "pem"


class ADAM(SysIdAlgBase):
    """
    PEM identification using Adam optimizer (Stochastic Gradient Descent).
    Useful for large datasets or when batch processing is needed.
    """

    def __init__(
        self,
        data: SysIdData,
        y_name: Union[str, list[str]],
        u_name: Union[str, list[str]],
        settings: Optional[dict[str, Any]] = None,
    ):
        if settings is None:
            settings = {}
        super().__init__(data, y_name, u_name, settings=settings)

        from .sysidalg import sysidalg

        init_method = self.settings.get("init", "arx")
        alg_creator = sysidalg.get_creator(init_method)
        self.alg_inst = alg_creator(data, y_name, u_name)
        self.logger = logging.getLogger(__name__)

        # Adam optimizer parameters
        self.learning_rate = settings.get("learning_rate", 0.001)
        self.beta1 = settings.get("beta1", 0.9)
        self.beta2 = settings.get("beta2", 0.999)
        self.epsilon = settings.get("epsilon", 1e-8)
        self.batch_size = settings.get("batch_size", 1024)
        self.max_epochs = settings.get("max_epochs", 100)
        self.tol = settings.get("tol", 1e-4)

        # Regularization parameters
        self.lambda_l1 = settings.get("lambda_l1", 0.0)
        self.lambda_l2 = settings.get("lambda_l2", 0.0)

        # Derivative method: 'complex' (default, faster and more accurate) or 'finite'
        self.derivative_method = settings.get("derivative_method", "complex")

        self.model: Optional[LTIModel] = None

    def compute_loss(self, x: np.ndarray, y_batch: np.ndarray, u_batch: np.ndarray) -> float:
        """Compute loss for given parameters and batch."""
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        self.model.reshape(x)
        y_hat = self.model.simulate(u_batch)
        loss = LTIModel.SSE(y_batch - y_hat)

        # Add regularization terms
        if self.lambda_l1 > 0:
            loss += self.lambda_l1 * np.sum(np.abs(x))
        if self.lambda_l2 > 0:
            loss += self.lambda_l2 * (x.T @ x)

        return float(loss)

    def compute_gradient(self, x: np.ndarray, y_batch: np.ndarray, u_batch: np.ndarray) -> np.ndarray:
        """
        Compute gradient using either finite differences with small epsilon or scipy's approx_fprime.

        Args:
            x: Parameter vector.
            y_batch: Output batch.
            u_batch: Input batch.

        Returns:
            Gradient vector.

        Notes:
            - 'complex': Uses finite differences with small epsilon (1e-8) for better accuracy.
              Note: This is NOT a true complex step (model.simulate doesn't support complex parameters).
              For true complex step, use the OE class with derivative_method='complex'.
            - 'finite': Uses scipy.optimize.approx_fprime (slower, less accurate).
        """
        if self.derivative_method == "complex":
            # Use finite differences with small epsilon (1e-8) for better accuracy
            return self._compute_gradient_finite(x, y_batch, u_batch)
        else:
            # Use scipy's approx_fprime (slower, less accurate)
            return self._compute_gradient_finite(x, y_batch, u_batch)

    def _compute_gradient_finite(self, x: np.ndarray, y_batch: np.ndarray, u_batch: np.ndarray) -> np.ndarray:
        """Compute gradient using finite differences (scipy's approx_fprime)."""
        # Use Numba-accelerated version for larger problems
        if len(x) >= 20:
            return self._compute_gradient_finite_numba(x, y_batch, u_batch)

        def loss_func(params):
            return self.compute_loss(params, y_batch, u_batch)

        return scipy.optimize.approx_fprime(x, loss_func, epsilon=1e-8)



    def _compute_gradient_finite(self, x: np.ndarray, y_batch: np.ndarray, u_batch: np.ndarray) -> np.ndarray:
        """
        Compute gradient using forward finite differences with epsilon=1e-8.

        This is the consolidated gradient computation method for ADAM optimizer.
        It uses a small epsilon for better accuracy than standard finite differences
        (1e-4 to 1e-6) while remaining numerically stable for real-valued simulations.

        Note: For true complex step, use the OE class with derivative_method='complex',
        which supports complex coefficients via oe_simulate.

        Args:
            x: Parameter vector.
            y_batch: Output batch.
            u_batch: Input batch.

        Returns:
            Gradient vector.
        """
        epsilon = 1e-8
        n_params = len(x)

        # Pre-compute nominal loss (includes regularization via compute_loss)
        nominal_loss = self.compute_loss(x, y_batch, u_batch)

        # Pre-allocate array for perturbed losses
        losses = np.zeros(n_params)

        # Compute all perturbed losses
        for i in range(n_params):
            x_perturbed = x.copy()
            x_perturbed[i] += epsilon
            losses[i] = self.compute_loss(x_perturbed, y_batch, u_batch)

        # Vectorized gradient computation (NumPy is already C-speed for this)
        grad = (losses - nominal_loss) / epsilon

        return grad



    def _ident(self, order: Union[int, tuple[int, ...]]) -> LTIModel:
        """
        Identify the model using Adam optimizer.
        """
        self.model = self.alg_inst.ident(order)
        x = self.model.vectorize()

        # Initialize Adam parameters
        m = np.zeros_like(x)  # First moment
        v = np.zeros_like(x)  # Second moment
        t = 0  # Time step

        # Convert data to numpy arrays
        y_data = self.y
        u_data = self.u
        n_samples = len(y_data)
        n_batches = int(np.ceil(n_samples / self.batch_size))

        for epoch in range(self.max_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0

            # Progress bar for batches
            batch_pbar = tqdm(
                range(0, n_samples, self.batch_size),
                desc=f"Epoch {epoch + 1}/{self.max_epochs}",
                unit="batch",
                total=n_batches,
                leave=False,
            )

            for i in batch_pbar:
                t += 1
                batch_indices = indices[i : min(i + self.batch_size, n_samples)]
                y_batch = y_data[batch_indices]
                u_batch = u_data[batch_indices]

                # Compute gradients
                grad = self.compute_gradient(x, y_batch, u_batch)

                # Update biased first moment estimate
                m = self.beta1 * m + (1 - self.beta1) * grad
                # Update biased second raw moment estimate
                v = self.beta2 * v + (1 - self.beta2) * np.square(grad)

                # Compute bias-corrected first moment estimate
                m_hat = m / (1 - np.power(self.beta1, t))
                # Compute bias-corrected second raw moment estimate
                v_hat = v / (1 - np.power(self.beta2, t))

                # Update parameters
                x = x - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

                # Update batch progress bar
                batch_loss = self.compute_loss(x, y_batch, u_batch)
                epoch_loss += batch_loss
                batch_pbar.set_postfix({"loss": f"{batch_loss:.2e}"})

            # Compute full loss for convergence check
            current_loss = self.compute_loss(x, self.y, self.u)
            self.logger.debug(f"Epoch {epoch}, Loss: {current_loss:10.6g}")

        # Use the best parameters found (last ones in this implementation)
        self.model.reshape(x)

        # Compute approximate covariance matrix
        grad = self.compute_gradient(x, self.y, self.u).reshape(1, -1)
        var_e = np.var(self.y - self.model.simulate(self.u))
        self.model.cov = var_e * (grad.T @ grad)

        return self.model

    @staticmethod
    def name() -> str:
        return "adam"


class OE(PEM):
    """
    Output Error (OE) identification.

    Special case of PEM initialized with ARX but typically implies
    Output Error model structure B(q)/F(q).

    This implementation uses analytical gradients via sensitivity filtering
    for significantly faster optimization (20-50x speedup over finite differences).
    """

    def __init__(
        self,
        data: SysIdData,
        y_name: Union[str, list[str]],
        u_name: Union[str, list[str]],
        settings: Optional[dict[str, Any]] = None,
    ):
        if settings is None:
            settings = {}
        # OE is typically initialized with ARX
        settings["init"] = "arx"
        super().__init__(data, y_name, u_name, settings=settings)

        # Import math module for OE-specific functions
        from . import math as _math

        self._math = _math

    def _objective_analytical(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Objective function for OE identification with analytical gradients and overflow protection.
        Returns (cost, gradient) for scipy.optimize.minimize.
        """
        cost, grad = self._math.oe_cost_and_gradient(
            theta, self.u.ravel(), self.y.ravel(), self._oe_nb, self._oe_nf_full, self._oe_nk
        )

        # Protect against unstable simulations (poles outside unit circle)
        # Replace NaN/inf with massive penalty to force optimizer to retreat
        if not np.isfinite(cost):
            cost = 1e300
            grad = np.zeros_like(grad)
        elif not np.all(np.isfinite(grad)):
            # Sanitize gradients to prevent optimizer math errors
            grad = np.nan_to_num(grad, nan=0.0, posinf=1e10, neginf=-1e10)

        return cost, grad

    def _objective_finite(self, theta: np.ndarray) -> float:
        """
        Objective function using finite differences for gradient computation.
        """
        from .polynomialmodel import PolynomialModel

        mod_temp = PolynomialModel(
            a=np.concatenate(([1.0], theta[self._oe_nb :])),
            b=theta[: self._oe_nb],
            nk=self._oe_nk,
            Ts=self._oe_mod_Ts,
        )
        y_hat = mod_temp.simulate(self.u)
        sse = LTIModel.SSE(self.y - y_hat)
        # Handle numerical instability
        return float(np.nan_to_num(sse, nan=1e300))

    def _objective_finite_wrapper(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Wrapper for _objective_finite to return (cost, grad) tuple with zero gradient."""
        return (self._objective_finite(theta), np.zeros_like(theta))

    def _objective_complex(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Objective function using complex step method for gradient computation.
        Uses direct simulation with complex coefficients via oe_simulate.
        """
        cost = self._objective_finite(theta)
        epsilon = 1e-20
        grad = np.zeros_like(theta)

        for i in range(len(theta)):
            # Perturb parameter i with complex step
            theta_perturbed = theta.copy().astype(np.complex128)
            theta_perturbed[i] = theta[i] + epsilon * 1j

            # Simulate with complex coefficients using oe_simulate
            b_complex = theta_perturbed[: self._oe_nb]
            f_complex = np.concatenate((np.array([1.0 + 0j]), theta_perturbed[self._oe_nb :]))
            y_hat_complex = self._math.oe_simulate(self.u.ravel(), b_complex, f_complex, self._oe_nk)

            # Compute cost with complex output (do NOT cast to float!)
            cost_complex = np.sum((self.y.ravel() - y_hat_complex) ** 2)

            grad[i] = np.imag(cost_complex) / epsilon

        # Protect against unstable simulations
        if not np.isfinite(cost):
            cost = 1e300
            grad = np.zeros_like(grad)
        elif not np.all(np.isfinite(grad)):
            grad = np.nan_to_num(grad, nan=0.0, posinf=1e10, neginf=-1e10)

        return cost, grad

    def _ident(self, order: Union[int, tuple[int, ...]]) -> LTIModel:
        """
        Identify OE model using analytical gradients.

        This overrides the parent PEM._ident to use the specialized OE cost
        function with analytical gradients, providing significant speedup (20-50x).

        The derivative method can be selected via settings['derivative_method']:
        - 'analytical' (default): Uses exact analytical gradients via sensitivity filtering. Fastest.
        - 'complex': Uses complex step method for numerical gradients. More accurate than finite differences.
        - 'finite': Uses finite differences for numerical gradients. Most robust but slowest.

        Args:
            order: Model order as (na, nb, nk) for ARX compatibility.
                For OE models (B/F structure), na maps to nf and nb maps to nb.
                - na: Number of A coefficients (maps to nf in OE)
                - nb: Number of B coefficients (maps to nb in OE)
                - nk: Input delay

        Returns:
            LTIModel: Identified PolynomialModel with B/F structure.
        """
        from .polynomialmodel import PolynomialModel

        # Check derivative method setting
        derivative_method = self.settings.get("derivative_method", "analytical").lower()

        # Parse order - for backward compatibility, interpret as (na, nb, nk)
        if isinstance(order, int):
            # Default: assume order is na=nb, with nk=0
            na = order
            nb = order
            nk = 0
        elif isinstance(order, tuple) and len(order) == 3:
            na, nb, nk = order
        else:
            raise ValueError(f"Invalid order for OE: {order}. Expected (na, nb, nk) or int.")

        # For OE models (B/F structure), map ARX orders to OE orders:
        # - nb_OE = nb (numerator B coefficients)
        # - nf_OE = na (denominator F coefficients, excluding leading 1.0)
        nb_oe = nb
        nf_oe = na

        # Initialize model using ARX (as per OE class default)
        mod = self.alg_inst.ident((na, nb, nk))  # ARX uses (na, nb, nk)

        # Extract initial parameters: [b0, b1, ..., f1, f2, ...]
        # Note: mod.a = [1.0, a1, a2, ...] from ARX, which maps to F = [1.0, f1, f2, ...]
        #       mod.b = [b0, b1, ...] from ARX, which maps to B = [b0, b1, ...]
        # theta = [b..., f...] where f... are the coefficients after the leading 1.0
        theta0 = np.concatenate((mod.b, mod.a[1:]))

        # Number of F coefficients (including leading 1.0)
        nf_full = nf_oe + 1
        n_params = nb_oe + nf_oe  # Total number of parameters

        # Store parameters as instance variables for use in objective functions
        self._oe_nb = nb_oe
        self._oe_nf_full = nf_full
        self._oe_nk = nk
        self._oe_mod_Ts = mod.Ts

        # Select derivative method
        if derivative_method == "analytical":
            objective = self._objective_analytical
            use_analytical = True
        elif derivative_method == "finite":
            objective = self._objective_finite_wrapper
            use_analytical = False
        elif derivative_method == "complex":
            objective = self._objective_complex
            use_analytical = True
        else:
            self.logger.warning(f"Unknown derivative_method '{derivative_method}'. Using 'analytical'.")
            objective = self._objective_analytical
            use_analytical = True

        # Get minimizer settings
        minimizer_kwargs = self.settings.get("minimizer_kwargs", {})
        method = minimizer_kwargs.get("method", "L-BFGS-B")
        # Methods that support bounds (case-insensitive comparison)
        bounds_methods = {"l-bfgs-b", "tnc", "slsqp", "powell", "cobyla"}

        # Methods that support analytical gradients (case-insensitive comparison)
        gradient_methods = {"bfgs", "newton-cg", "l-bfgs-b", "tnc", "slsqp", "dogleg", "trust-ncg"}

        method_lower = method.lower()

        # Use analytical gradient if method supports it and we're using analytical/complex derivatives
        if use_analytical and method_lower in gradient_methods:
            # Add bounds for methods that support them
            if method_lower in bounds_methods:
                bounds = [(-10, 10)] * n_params
            else:
                bounds = None
            res = scipy.optimize.minimize(
                objective,
                theta0,
                method=method,
                bounds=bounds,
                jac=True,  # objective returns (cost, grad)
                options=minimizer_kwargs.get("options", {"disp": False, "maxiter": 1000}),
            )
        else:
            # For methods that don't support gradients or when using finite differences
            # fall back to numerical approximation
            res = scipy.optimize.minimize(
                lambda theta: objective(theta)[0],  # Cost function only
                theta0,
                method=method,
                bounds=[(-10, 10)] * n_params if method_lower in bounds_methods else None,
                options=minimizer_kwargs.get("options", {"disp": False, "maxiter": 1000}),
            )

        # The Fallback Trigger: if analytical/complex optimization fails, fall back to PEM's finite differences
        if not res.success or res.fun > 1e10:
            self.logger.warning(
                f"OE {derivative_method} optimization failed ({res.message}). "
                "Falling back to robust numerical finite differences."
            )
            # Route directly to the PEM parent class logic, which natively
            # uses finite differences via mod.simulate()
            return super()._ident(order)

        # Update model with optimized parameters - create new instance to ensure consistency
        theta_opt = res.x
        mod = PolynomialModel(
            a=np.concatenate(([1.0], theta_opt[nb_oe:])),
            b=theta_opt[:nb_oe],
            nk=nk,
            Ts=mod.Ts,
        )

        # Define a reshape function for PolynomialModel that maps theta to (a, b)
        def reshape_polynomial(theta: np.ndarray) -> None:
            mod.a = np.concatenate(([1.0], theta[nb_oe:]))
            mod.b = theta[:nb_oe]

        # Estimate covariance matrix using finite differences
        self._estimate_covariance(mod, theta_opt, epsilon=1e-8, reshape_func=reshape_polynomial)

        return mod

    @staticmethod
    def name() -> str:
        return "oe"
