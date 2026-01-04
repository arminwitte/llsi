from .autoident import AutoIdentResult, autoident
from .figure import Figure
from .matlab import arx, compare, iddata, impulse, n4sid, oe, pem, step
from .polynomialmodel import PolynomialModel
from .statespacemodel import StateSpaceModel
from .sysidalg import sysid
from .sysiddata import SysIdData
from .utils import cv, load_model, save_model

try:
    from .sklearn import LTIModel
except ImportError:

    class LTIModel:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "scikit-learn is required to use LTIModel. Install it with 'pip install llsi[sklearn]'."
            ) from None


__all__ = [
    "Figure",
    "PolynomialModel",
    "LTIModel",
    "StateSpaceModel",
    "sysid",
    "SysIdData",
    "cv",
    "load_model",
    "save_model",
    "iddata",
    "arx",
    "n4sid",
    "oe",
    "pem",
    "compare",
    "step",
    "impulse",
    "autoident",
    "AutoIdentResult",
]
