"""
Factory for system identification algorithms and convenience function.
"""

from typing import Any, Dict, List, Optional, Type, Union

from .arx import ARX, FIR
from .firor import FIROR
from .ltimodel import LTIModel
from .pem import OE, PEM
from .subspace import N4SID, PO_MOESP
from .sysidalgbase import SysIdAlgBase
from .sysiddata import SysIdData


class SysIdAlgFactory:
    """Factory for registering and retrieving system identification algorithms."""

    def __init__(self) -> None:
        self.creators: Dict[str, Type[SysIdAlgBase]] = {}
        self.default_creator_name: Optional[str] = None

    def register_creator(self, creator: Type[SysIdAlgBase], default: bool = False) -> None:
        """
        Register a system identification algorithm class.

        Args:
            creator: The algorithm class (subclass of SysIdAlgBase).
            default: Whether this algorithm should be the default.
        """
        name = creator.name()
        if default:
            self.default_creator_name = name
        self.creators[name] = creator

    def get_creator(self, name: Optional[str] = None) -> Type[SysIdAlgBase]:
        """
        Get a system identification algorithm class by name.

        Args:
            name: The name of the algorithm. If None, returns the default.

        Returns:
            The algorithm class.

        Raises:
            KeyError: If the algorithm name is not found.
            ValueError: If no default algorithm is set and name is None.
        """
        if name:
            if name not in self.creators:
                raise KeyError(f"Unknown algorithm: {name}")
            c = self.creators[name]
        else:
            if self.default_creator_name is None:
                raise ValueError("No default algorithm set.")
            c = self.creators[self.default_creator_name]
        return c


sysidalg = SysIdAlgFactory()

sysidalg.register_creator(N4SID)
sysidalg.register_creator(PO_MOESP, default=True)
sysidalg.register_creator(PEM)
sysidalg.register_creator(ARX)
sysidalg.register_creator(FIROR)
sysidalg.register_creator(OE)
sysidalg.register_creator(FIR)


def sysid(
    data: SysIdData,
    y_name: Union[str, List[str]],
    u_name: Union[str, List[str]],
    order: Any,
    method: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> LTIModel:
    """
    Convenience function to perform system identification.

    Args:
        data: The system identification data.
        y_name: Name(s) of the output channel(s).
        u_name: Name(s) of the input channel(s).
        order: The order of the model (int or tuple/list depending on method).
        method: The name of the identification method (e.g., 'n4sid', 'po-moesp', 'pem', 'arx').
                If None, uses the default method.
        settings: Dictionary of settings for the algorithm.

    Returns:
        The identified LTI model.
    """
    if settings is None:
        settings = {}
    alg_cls = sysidalg.get_creator(method)
    alg_inst = alg_cls(data, y_name, u_name, settings=settings)
    mod = alg_inst.ident(order)
    return mod
