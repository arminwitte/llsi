"""
Finite Impulse Response Order Reduction (FIROR) method.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from .arx import ARX
from .statespacemodel import StateSpaceModel
from .sysidalgbase import SysIdAlgBase
from .sysiddata import SysIdData


class FIROR(SysIdAlgBase):
    """
    Finite Impulse Response Order Reduction (FIROR) identification method.

    This method first identifies a high-order Finite Impulse Response (FIR) model
    using ARX (with na=0) and then reduces it to a lower-order State-Space model.
    """

    def __init__(
        self,
        data: SysIdData,
        y_name: Union[str, List[str]],
        u_name: Union[str, List[str]],
        settings: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the FIROR identification.

        Args:
            data: The system identification data.
            y_name: Name(s) of the output channel(s).
            u_name: Name(s) of the input channel(s).
            settings: Configuration dictionary.
                      - 'lambda': Regularization parameter for FIR estimation. Default 1e-3.
                      - 'fir_order': Order of the intermediate FIR model. Default 100.
        """
        if settings is None:
            settings = {}
        super().__init__(data, y_name, u_name, settings=settings)

        lmb = self.settings.get("lambda", 1e-3)

        # Use ARX directly instead of going through sysidalg factory
        # This avoids circular imports and is more explicit
        self.alg_inst = ARX(data, y_name, u_name, settings={"lambda": lmb})
        self.logger = logging.getLogger(__name__)

    def ident(self, order: int) -> StateSpaceModel:
        """
        Identify the State-Space model.

        Args:
            order: The desired order of the reduced State-Space model.

        Returns:
            StateSpaceModel: The identified state-space model.
        """
        fir_order = self.settings.get("fir_order", 100)

        # Identify high-order FIR model: (na=0, nb=fir_order, nk=0)
        mod_fir = self.alg_inst.ident((0, fir_order, 0))

        # Convert FIR to StateSpace and reduce order
        red_mod = StateSpaceModel.from_fir(mod_fir)
        red_mod, s = red_mod.reduce_order(order)

        self.logger.debug(f"Hankel singular values (s): {s}")
        return red_mod

    @staticmethod
    def name() -> str:
        return "firor"
