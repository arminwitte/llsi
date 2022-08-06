#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 00:01:33 2022

@author: armin
"""

from llsi.ltimodel import LTIModel
from llsi.statespacemodel import StateSpaceModel

def test_init():
    ss = StateSpaceModel()
    assert isinstance(ss,LTIModel)
    assert isinstance(ss,StateSpaceModel)
    assert ss.Ts == 1.