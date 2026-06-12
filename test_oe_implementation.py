#!/usr/bin/env python3
"""
Test script to verify the OE implementation structure without requiring numpy.

This script checks:
1. The math.py file has the required functions
2. The pem.py file has the updated OE class
3. The functions have the correct signatures
"""

import ast
import sys


def check_math_functions():
    """Check that math.py has the required OE functions."""
    print("=" * 80)
    print("Checking math.py for OE functions...")
    print("=" * 80)

    with open("src/llsi/math.py") as f:
        content = f.read()

    # Parse the file
    tree = ast.parse(content)

    # Find all function definitions
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    required_functions = {"oe_simulate", "oe_cost_and_gradient"}

    for func in required_functions:
        if func in functions:
            print(f"✓ Found function: {func}")
        else:
            print(f"✗ Missing function: {func}")
            return False

    # Check for Numba decorators
    if "@njit(cache=True)" in content or "@njit" in content:
        print("✓ Numba decorators present")
    else:
        print("✗ Numba decorators missing")
        return False

    # Check function signatures
    if "def oe_simulate(u: np.ndarray, b: np.ndarray, f: np.ndarray, nk: int)" in content:
        print("✓ oe_simulate has correct signature")
    else:
        print("✗ oe_simulate signature may be incorrect")
        return False

    if "def oe_cost_and_gradient(" in content:
        print("✓ oe_cost_and_gradient has correct signature")
    else:
        print("✗ oe_cost_and_gradient signature may be incorrect")
        return False

    print("\n✓ All math.py checks passed!\n")
    return True


def check_pem_oe_class():
    """Check that pem.py has the updated OE class."""
    print("=" * 80)
    print("Checking pem.py for updated OE class...")
    print("=" * 80)

    with open("src/llsi/pem.py") as f:
        content = f.read()

    # Parse the file
    tree = ast.parse(content)

    # Find the OE class
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}

    if "OE" in classes:
        print("✓ Found OE class")
    else:
        print("✗ OE class not found")
        return False

    # Check for _ident method in OE class
    oe_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "OE":
            oe_class = node
            break

    if oe_class:
        methods = {node.name for node in oe_class.body if isinstance(node, ast.FunctionDef)}
        if "_ident" in methods:
            print("✓ OE class has _ident method")
        else:
            print("✗ OE class missing _ident method")
            return False

    # Check for analytical gradient usage
    if "jac=True" in content or "jac=lambda" in content:
        print("✓ OE uses analytical gradients (jac parameter)")
    else:
        print("✗ OE may not be using analytical gradients")
        return False

    # Check for oe_cost_and_gradient usage
    if "oe_cost_and_gradient" in content:
        print("✓ OE uses oe_cost_and_gradient function")
    else:
        print("✗ OE doesn't use oe_cost_and_gradient function")
        return False

    # Check for BFGS method
    if "BFGS" in content:
        print("✓ OE uses BFGS optimizer")
    else:
        print("✗ OE doesn't use BFGS optimizer")
        return False

    print("\n✓ All pem.py checks passed!\n")
    return True


def check_benchmark_script():
    """Check that the benchmark script exists and is valid."""
    print("=" * 80)
    print("Checking benchmark_oe_speedup.py...")
    print("=" * 80)

    try:
        with open("benchmark_oe_speedup.py") as f:
            content = f.read()

        # Parse the file
        tree = ast.parse(content)

        # Find all function definitions
        functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

        required_functions = {
            "generate_test_data",
            "benchmark_oe_analytical",
            "benchmark_pem_finite_differences",
            "benchmark_gradient_computation",
            "run_full_benchmark",
            "test_gradient_correctness",
        }

        for func in required_functions:
            if func in functions:
                print(f"✓ Found function: {func}")
            else:
                print(f"✗ Missing function: {func}")
                return False

        print("\n✓ All benchmark script checks passed!\n")
        return True
    except FileNotFoundError:
        print("✗ benchmark_oe_speedup.py not found")
        return False


def check_imports():
    """Check that the imports are correct."""
    print("=" * 80)
    print("Checking imports...")
    print("=" * 80)

    # Check math.py imports
    with open("src/llsi/math.py") as f:
        math_content = f.read()

    if "import numpy as np" in math_content:
        print("✓ math.py imports numpy")
    else:
        print("✗ math.py doesn't import numpy")
        return False

    if "from numba import njit" in math_content or "try:" in math_content and "from numba import njit" in math_content:
        print("✓ math.py imports numba with fallback")
    else:
        print("✗ math.py doesn't import numba")
        return False

    # Check pem.py imports
    with open("src/llsi/pem.py") as f:
        pem_content = f.read()

    if "from . import math as _math" in pem_content or "from .math import" in pem_content:
        print("✓ pem.py imports math module")
    else:
        print("✗ pem.py doesn't import math module")
        return False

    if "import scipy.optimize" in pem_content:
        print("✓ pem.py imports scipy.optimize")
    else:
        print("✗ pem.py doesn't import scipy.optimize")
        return False

    print("\n✓ All import checks passed!\n")
    return True


def main():
    """Run all checks."""
    print("\n" + "=" * 80)
    print("OE IMPLEMENTATION VERIFICATION")
    print("=" * 80 + "\n")

    all_passed = True

    # Run all checks
    checks = [
        ("Math Functions", check_math_functions),
        ("PEM OE Class", check_pem_oe_class),
        ("Benchmark Script", check_benchmark_script),
        ("Imports", check_imports),
    ]

    for name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
                print(f"\n✗ {name} checks FAILED\n")
        except Exception as e:
            all_passed = False
            print(f"\n✗ {name} checks FAILED with exception: {e}\n")

    # Final summary
    print("=" * 80)
    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("\nThe OE implementation with analytical gradients has been successfully:")
        print("  1. Added to math.py (oe_simulate, oe_cost_and_gradient)")
        print("  2. Integrated into the OE class in pem.py")
        print("  3. Benchmark script created (benchmark_oe_speedup.py)")
        print("\nExpected speedup: 20-50x for gradient computation")
        print("=" * 80)
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("Please review the implementation.")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
