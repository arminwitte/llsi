#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 00:01:33 2022

@author: armin
"""
import numpy as np
import pytest
import scipy.signal

from llsi.ltimodel import LTIModel


def test_NRMSE():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y_hat = np.array([1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 8.9, 9.1, 9.9])
    np.testing.assert_allclose(LTIModel.NRMSE(y, y_hat), 0.10444659357341872)
