from .figure import Figure
from .matlab import arx, compare, iddata, impulse, n4sid, oe, pem, step
from .polynomialmodel import PolynomialModel
from .statespacemodel import StateSpaceModel
from .sysidalg import sysid
from .sysiddata import SysIdData
from .utils import cv

try:
    from .sklearn import LTIModel
except ImportError:
    LTIModel = None

__all__ = [
    "Figure",
    "PolynomialModel",
    "LTIModel",
    "StateSpaceModel",
    "sysid",
    "SysIdData",
    "cv",
    "iddata",
    "arx",
    "n4sid",
    "oe",
    "pem",
    "compare",
    "step",
    "impulse",
]
