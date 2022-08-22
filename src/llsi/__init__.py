from .polynomialmodel import PolynomialModel
from .statespacemodel import StateSpaceModel
from .sysidalg import sysid
from .sysiddata import SysIdData

try:
    from .figure import Figure
except ImportError:

    class Figure:
        def __init__(self, *args, **kwargs):
            print("to use plotting capabilities please install matplotlib")
