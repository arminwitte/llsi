#!/usr/bin/env python3
"""
Created on Sun Apr  4 20:47:33 2021

@author: armin
"""


######################################################################################
# FACTORY
######################################################################################


class SysIdAlgFactory:
    def __init__(self):
        self.creators = {}
        self.default_creator_name = None

    def register_creator(self, creator, default=False):
        name = creator.name()
        if default:
            self.default_creator_name = name
        self.creators[name] = creator

    def get_creator(self, name=None):
        if name:
            c = self.creators[name]
        else:
            c = self.creators[self.default_creator_name]
        return c


sysidalg = SysIdAlgFactory()

from .arx import ARX, FIR  # noqa: E402
from .firor import FIROR  # noqa: E402
from .pem import OE, PEM  # noqa: E402
from .subspace import N4SID, PO_MOESP  # noqa: E402

sysidalg.register_creator(N4SID)
sysidalg.register_creator(PO_MOESP, default=True)
sysidalg.register_creator(PEM)
sysidalg.register_creator(ARX)
sysidalg.register_creator(FIROR)
sysidalg.register_creator(OE)
sysidalg.register_creator(FIR)

######################################################################################
# CONVENIENCE FUNCTION
######################################################################################


def sysid(data, y_name, u_name, order, method=None, settings=None):
    if settings is None:
        settings = {}
    alg = sysidalg.get_creator(method)
    alg_inst = alg(data, y_name, u_name, settings=settings)
    mod = alg_inst.ident(order)
    return mod
