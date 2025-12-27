#!/usr/bin/env python3
"""
Created on Fri Aug 19 23:10:09 2022

@author: armin
"""

import logging

from .statespacemodel import StateSpaceModel
from .sysidalg import sysidalg
from .sysidalgbase import SysIdAlgBase


class FIROR(SysIdAlgBase):
    def __init__(self, data, y_name, u_name, settings=None):
        if settings is None:
            settings = {}
        super().__init__(data, y_name, u_name, settings=settings)
        alg = sysidalg.get_creator("arx")
        lmb = self.settings.get("lambda", 1e-3)
        self.alg_inst = alg(data, y_name, u_name, settings={"lambda": lmb})
        self.logger = logging.getLogger(__name__)

    def ident(self, order):
        fir_order = self.settings.get("fir_order", 100)
        mod = self.alg_inst.ident((0, fir_order, 0))

        red_mod = StateSpaceModel.from_fir(mod)
        red_mod, s = red_mod.reduce_order(order)
        self.logger.debug(f"s: {s}")
        return red_mod

    @staticmethod
    def name():
        return "firor"
