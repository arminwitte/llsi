#!/usr/bin/env python3
"""
Created on Sun Apr  4 20:47:33 2021

@author: armin
"""

import logging

import numpy as np
import scipy.optimize
from tqdm.auto import tqdm

from .sysidalg import sysidalg
from .sysidalgbase import SysIdAlgBase


class PEM(SysIdAlgBase):
    def __init__(self, data, y_name, u_name, settings=None):
        if settings is None:
            settings = {}
        super().__init__(data, y_name, u_name, settings=settings)
        init = self.settings.get("init", "arx")
        alg = sysidalg.get_creator(init)
        # alg = sysidalg.get_creator('n4sid')
        self.alg_inst = alg(data, y_name, u_name)
        self.logger = logging.getLogger(__name__)

    def ident(self, order):
        mod = self.alg_inst.ident(order)
        # y_hat = mod.simulate(self.u)
        # sse0 = self._sse(self.y, y_hat)
        lambda_l1 = self.settings.get("lambda_l1", 0.0)
        lambda_l2 = self.settings.get("lambda_l2", 0.0)

        def fun(x):
            mod.reshape(x)
            y_hat = mod.simulate(self.u)
            sse = self._sse(self.y, y_hat)
            sse = np.nan_to_num(sse, nan=1e300)
            # print("{:10.6g}".format(sse / sse0))
            # return sse / sse0
            x_ = x.ravel()
            J = sse + lambda_l1 * np.sum(np.abs(x)) + lambda_l2 * x_.T @ x_
            self.logger.debug(f"{J:10.6g}")
            return J

        x0 = mod.vectorize()
        # method = self.settings.get("minimizer", "nelder-mead")
        # res = scipy.optimize.minimize(
        # fun, x0, method=method, options={"maxiter": 200, "maxfev": 200}
        # (
        # res = scipy.optimize.basinhopping(
        #     fun,
        #     x0,
        #     niter=1,
        #     minimizer_kwargs={"method": "BFGS", "options": {"maxiter": 20}},
        #     disp=True,
        # )
        # res = scipy.optimize.minimize(fun,x0,method='nelder-mead')
        # res = scipy.optimize.minimize(fun,res.x,method='BFGS',options={"gtol":1e-3})
        minimizer_kwargs = self.settings.get("minimizer_kwargs", {"method": "powell"})
        res = scipy.optimize.minimize(fun, x0, **minimizer_kwargs)
        mod.reshape(res.x)

        J = scipy.optimize.approx_fprime(res.x, fun).reshape(1, -1)
        var_e = np.var(self.y - mod.simulate(self.u))
        mod.cov = var_e * (J.T @ J)

        return mod

    @staticmethod
    def name():
        return "pem"


class ADAM(SysIdAlgBase):
    def __init__(self, data, y_name, u_name, settings=None):
        if settings is None:
            settings = {}
        super().__init__(data, y_name, u_name, settings=settings)
        init = self.settings.get("init", "arx")
        alg = sysidalg.get_creator(init)
        self.alg_inst = alg(data, y_name, u_name)
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

    def compute_loss(self, x, y_batch, u_batch):
        """Compute loss for given parameters and batch"""
        self.model.reshape(x)
        y_hat = self.model.simulate(u_batch)
        loss = self._sse(y_batch, y_hat)

        # Add regularization terms
        if self.lambda_l1 > 0:
            loss += self.lambda_l1 * np.sum(np.abs(x))
        if self.lambda_l2 > 0:
            loss += self.lambda_l2 * x.T @ x

        return loss

    def compute_gradient(self, x, y_batch, u_batch):
        """Compute gradient using scipy's approx_fprime"""

        def loss_func(params):
            return self.compute_loss(params, y_batch, u_batch)

        return scipy.optimize.approx_fprime(x, loss_func)

    def ident(self, order):
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

        # best_loss = float("inf")
        # best_x = x.copy()
        # patience_counter = 0

        for epoch in range(self.max_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            epoch_loss = 0

            # Progress bar for batches
            batch_pbar = tqdm(
                range(0, n_samples, self.batch_size),
                desc=f"Epoch {epoch + 1}/{self.max_epochs}",
                unit="batch",
                total=n_batches,
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
                batch_pbar.set_postfix({"batch_loss": f"{batch_loss:.4e}"})

            # Compute full loss for convergence check
            current_loss = self.compute_loss(x, self.y, self.u)
            self.logger.debug(f"Epoch {epoch}, Loss: {current_loss:10.6g}")

            # # Early stopping logic
            # if current_loss < best_loss - self.tol:
            #     best_loss = current_loss
            #     best_x = x.copy()
            #     patience_counter = 0
            # else:
            #     patience_counter += 1
            #     if patience_counter >= 10:
            #         break

        # Use the best parameters found
        self.model.reshape(x)

        # Compute approximate covariance matrix
        J = self.compute_gradient(x, self.y, self.u).reshape(1, -1)
        var_e = np.var(self.y - self.model.simulate(self.u))
        self.model.cov = var_e * (J.T @ J)

        return self.model

    @staticmethod
    def name():
        return "adam"


######################################################################################
# CONVENIENCE CLASSES
######################################################################################


class OE(PEM):
    def __init__(self, data, y_name, u_name, settings=None):
        if settings is None:
            settings = {}
        settings["init"] = "arx"
        super().__init__(data, y_name, u_name, settings=settings)

    @staticmethod
    def name():
        return "oe"
