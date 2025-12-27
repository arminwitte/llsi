import pytest
from numpy.polynomial import Polynomial
from numpy.testing import assert_array_almost_equal
from scipy.signal import TransferFunction

from llsi.controllerdesigner import (
    ControllerDesigner,
    NPZICDesigner,
    ZMETCDesigner,
    ZPETCDesigner,
    create_designer,
)


@pytest.fixture
def sample_system():
    """Create a sample discrete-time transfer function with non-minimum phase zeros."""
    num = [1, -1.5]  # z - 1.5 (non-minimum phase zero)
    den = [1, -0.5]  # z - 0.5 (stable pole)
    Ts = 0.1
    return TransferFunction(num, den, dt=Ts)


@pytest.fixture
def minimum_phase_system():
    """Create a sample discrete-time transfer function with only minimum phase zeros."""
    num = [1, 0.5]  # z + 0.5 (minimum phase zero)
    den = [1, -0.5]  # z - 0.5 (stable pole)
    Ts = 0.1
    return TransferFunction(num, den, dt=Ts)


class TestControllerDesigner:
    """Test suite for base ControllerDesigner class functionality."""

    def test_extract_roots_gain(self, sample_system):
        """Test the root extraction method."""
        designer = NPZICDesigner(sample_system)
        acceptable, unacceptable, poles, gain = designer._extract_roots_gain(acceptable_threshold=1.0)

        assert len(unacceptable) == 1
        assert abs(unacceptable[0] - 1.5) < 1e-10
        assert len(poles) == 1
        assert abs(poles[0] - 0.5) < 1e-10
        assert abs(gain - 1.0) < 1e-10

    def test_rel_deg_calculation(self):
        """Test relative degree calculation between polynomials."""
        A = Polynomial([1, 2, 3])  # degree 2
        B = Polynomial([1, 2])  # degree 1

        rel_deg = NPZICDesigner._rel_deg(A, B)
        assert rel_deg == 1

    def test_rel_deg_poly_creation(self):
        """Test creation of relative degree polynomial."""
        q = 2
        poly = NPZICDesigner._rel_deg_poly(q)

        assert poly.degree() == q
        assert_array_almost_equal(poly.coef, [0, 0, 1])

    def test_low_pass_filter(self, sample_system):
        """Test adding low-pass filter to designed controller."""
        designer = NPZICDesigner(sample_system)

        # Should raise error if no controller is designed yet
        with pytest.raises(ValueError):
            designer.add_low_pass()

        # Design controller first
        tf = designer.design()

        # Add low-pass filter
        filtered_tf = designer.add_low_pass(N=1, Wn=0.5)
        assert filtered_tf.dt == sample_system.dt
        assert len(filtered_tf.num) > len(tf.num)  # Order should increase


class TestNPZICDesigner:
    """Test suite for NPZIC Designer."""

    def test_design_result(self, sample_system):
        """Test NPZIC design method results."""
        designer = NPZICDesigner(sample_system)
        tf = designer.design()

        assert isinstance(tf, TransferFunction)
        assert tf.dt == sample_system.dt

    def test_minimum_phase_case(self, minimum_phase_system):
        """Test NPZIC design with minimum phase system."""
        designer = NPZICDesigner(minimum_phase_system)
        tf = designer.design()

        assert isinstance(tf, TransferFunction)
        assert tf.dt == minimum_phase_system.dt


class TestZPETCDesigner:
    """Test suite for ZPETC Designer."""

    def test_design_result(self, sample_system):
        """Test ZPETC design method results."""
        designer = ZPETCDesigner(sample_system)
        tf = designer.design()

        assert isinstance(tf, TransferFunction)
        assert tf.dt == sample_system.dt

        # ZPETC should have higher order numerator due to B_u_ast
        assert len(tf.num) > len(sample_system.num)


class TestZMETCDesigner:
    """Test suite for ZMETC Designer."""

    def test_not_implemented(self, sample_system):
        """Test that ZMETC raises NotImplementedError."""
        designer = ZMETCDesigner(sample_system)
        with pytest.raises(NotImplementedError):
            designer.design()


class TestFactoryFunction:
    """Test suite for designer factory function."""

    def test_valid_methods(self, sample_system):
        """Test factory function with all valid methods."""
        valid_methods = ["npzic", "zpetc"]

        for method in valid_methods:
            designer = create_designer(sample_system, method)
            assert isinstance(designer, ControllerDesigner)
            tf = designer.design()
            assert isinstance(tf, TransferFunction)

    def test_invalid_method(self, sample_system):
        """Test factory function with invalid method."""
        with pytest.raises(ValueError):
            create_designer(sample_system, "invalid_method")

    def test_case_insensitive(self, sample_system):
        """Test that method names are case insensitive."""
        designer1 = create_designer(sample_system, "NPZIC")
        designer2 = create_designer(sample_system, "npzic")

        tf1 = designer1.design()
        tf2 = designer2.design()

        assert_array_almost_equal(tf1.num, tf2.num)
        assert_array_almost_equal(tf1.den, tf2.den)


def test_end_to_end(sample_system):
    """End-to-end test of the design process."""
    # Create designer
    designer = create_designer(sample_system, "npzic")

    # Design controller
    tf = designer.design()
    assert isinstance(tf, TransferFunction)

    # Add low-pass filter
    filtered_tf = designer.add_low_pass(N=2, Wn=0.3)
    assert isinstance(filtered_tf, TransferFunction)
    assert len(filtered_tf.num) > len(tf.num)  # Order should increase due to filter
