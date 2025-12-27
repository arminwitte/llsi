import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .sysidalg import sysid
from .sysiddata import SysIdData


class LTIModel(BaseEstimator, RegressorMixin):
    """
    Scikit-learn wrapper for LTI system identification models.
    """

    def __init__(self, method="n4sid", order=1, settings=None, Ts=1.0):
        self.method = method
        self.order = order
        self.settings = settings
        self.Ts = Ts
        self.model_ = None

    def fit(self, X, y):
        """
        Fit the LTI model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input signals.
        y : array-like of shape (n_samples, n_targets)
            Output signals.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_inputs = X.shape
        n_samples_y, n_outputs = y.shape

        if n_samples != n_samples_y:
            raise ValueError("X and y must have the same number of samples")

        data = SysIdData(Ts=self.Ts)

        u_names = [f"u{i}" for i in range(n_inputs)]
        y_names = [f"y{i}" for i in range(n_outputs)]

        for i, name in enumerate(u_names):
            data.add_series(**{name: X[:, i]})

        for i, name in enumerate(y_names):
            data.add_series(**{name: y[:, i]})

        self.model_ = sysid(data, y_names, u_names, self.order, method=self.method, settings=self.settings)

        return self

    def predict(self, X):
        """
        Predict output using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input signals.

        Returns
        -------
        y_pred : array-like of shape (n_samples, n_targets)
            Predicted output signals.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted")

        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # simulate expects (nu, N)
        return self.model_.simulate(X.T)
