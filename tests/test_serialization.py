import json

import numpy as np

from llsi.polynomialmodel import PolynomialModel
from llsi.statespacemodel import StateSpaceModel
from llsi.utils import load_model, save_model


def test_polynomial_model_serialization(tmp_path):
    # Create a PolynomialModel
    a = [1, -0.9]
    b = [0.5]
    mod = PolynomialModel(a=a, b=b, Ts=0.1, nk=2)
    mod.cov = np.array([[0.01]])
    mod.info = {"author": "tester"}

    # Save
    filename = tmp_path / "poly_model.json"
    save_model(mod, str(filename))

    # Load
    mod_loaded = load_model(str(filename))

    # Verify
    assert isinstance(mod_loaded, PolynomialModel)
    np.testing.assert_array_almost_equal(mod.a, mod_loaded.a)
    np.testing.assert_array_almost_equal(mod.b, mod_loaded.b)
    assert mod.Ts == mod_loaded.Ts
    assert mod.nk == mod_loaded.nk
    assert mod.na == mod_loaded.na
    assert mod.nb == mod_loaded.nb
    np.testing.assert_array_almost_equal(mod.cov, mod_loaded.cov)

    # Check info
    assert str(mod.info) == mod_loaded.info


def test_statespace_model_serialization(tmp_path):
    # Create a StateSpaceModel
    A = [[0.9, 0.1], [0, 0.8]]
    B = [[1], [0]]
    C = [[1, 0]]
    D = [[0]]
    mod = StateSpaceModel(A=A, B=B, C=C, D=D, Ts=0.1, nk=1)
    mod.x_init = np.array([[0.5], [-0.5]])
    mod.cov = np.eye(2) * 0.01
    mod.info = {"author": "tester"}

    # Save
    filename = tmp_path / "ss_model.json"
    save_model(mod, str(filename))

    # Load
    mod_loaded = load_model(str(filename))

    # Verify
    assert isinstance(mod_loaded, StateSpaceModel)
    np.testing.assert_array_almost_equal(mod.A, mod_loaded.A)
    np.testing.assert_array_almost_equal(mod.B, mod_loaded.B)
    np.testing.assert_array_almost_equal(mod.C, mod_loaded.C)
    np.testing.assert_array_almost_equal(mod.D, mod_loaded.D)
    assert mod.Ts == mod_loaded.Ts
    assert mod.nk == mod_loaded.nk
    np.testing.assert_array_almost_equal(mod.x_init, mod_loaded.x_init)
    np.testing.assert_array_almost_equal(mod.cov, mod_loaded.cov)

    # StateSpaceModel.to_json DOES save info.
    # But it saves it as str(self.info).
    assert str(mod.info) == mod_loaded.info


def test_save_load_model_utils(tmp_path):
    # Test that save_model adds the __class__ tag and load_model uses it
    mod = PolynomialModel(a=[1, -0.5], b=[1], Ts=1.0)
    filename = tmp_path / "test_utils.json"

    save_model(mod, str(filename))

    with open(filename) as f:
        data = json.load(f)

    assert "__class__" in data
    assert data["__class__"] == "PolynomialModel"

    loaded = load_model(str(filename))
    assert isinstance(loaded, PolynomialModel)
