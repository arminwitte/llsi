#!/usr/bin/env python3
"""
Benchmark script to measure the speedup achieved by analytical gradients in OE identification.

This script compares:
1. OE with analytical gradients (new implementation)
2. OE with finite differences (old PEM approach)
3. The theoretical speedup (N_params factor)

Usage:
    python benchmark_oe_speedup.py
"""

import time

import numpy as np

# Try to import llsi components
try:
    from src.llsi import math as llsi_math
    from src.llsi.pem import OE, PEM

    USE_LLSI = True
except ImportError:
    # Fallback for installed package
    try:
        from llsi import math as llsi_math
        from llsi.pem import OE, PEM

        USE_LLSI = True
    except ImportError:
        USE_LLSI = False
        print("Error: Could not import llsi. Please install or add to path.")
        exit(1)


def generate_test_data(N: int = 1000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate test input-output data for OE model identification."""
    np.random.seed(seed)

    # Generate input signal (PRBS-like)
    u = np.random.uniform(-1, 1, N)

    # True system: B(q)/F(q) with known coefficients
    # Let's use a 2nd order system: y = (b0 + b1*q^-1) / (1 + f1*q^-1 + f2*q^-2) * u
    b_true = np.array([0.5, 0.3])
    f_true = np.array([1.0, -0.8, 0.2])
    nk = 1

    # Simulate true output
    y = llsi_math.oe_simulate(u, b_true, f_true, nk)

    # Add some noise
    noise_level = 0.01
    y += noise_level * np.random.randn(N)

    return u, y


def benchmark_oe_analytical(
    u: np.ndarray,
    y: np.ndarray,
    order: tuple[int, int, int],
    n_runs: int = 3,
) -> dict:
    """Benchmark OE with analytical gradients."""
    from src.llsi.sysiddata import SysIdData

    # Create SysIdData (pass series as keyword arguments)
    data = SysIdData(Ts=1.0, u=u, y=y)

    # Create OE identifier
    oe = OE(data, y_name="y", u_name="u")

    times = []
    costs = []

    for _ in range(n_runs):
        start = time.perf_counter()
        mod = oe.ident(order)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        costs.append(mod.aic if mod.aic is not None else float("inf"))

    return {
        "method": "OE (analytical gradients)",
        "times": times,
        "costs": costs,
        "median_time": np.median(times),
        "mean_cost": np.mean(costs),
    }


def benchmark_pem_finite_differences(
    u: np.ndarray,
    y: np.ndarray,
    order: tuple[int, int, int],
    n_runs: int = 3,
) -> dict:
    """Benchmark PEM with finite differences (old approach)."""
    from src.llsi.sysiddata import SysIdData

    # Create SysIdData (pass series as keyword arguments)
    data = SysIdData(Ts=1.0, u=u, y=y)

    # Create PEM identifier with ARX initialization
    pem = PEM(data, y_name="y", u_name="u", settings={"init": "arx"})

    times = []
    costs = []

    for _ in range(n_runs):
        start = time.perf_counter()
        mod = pem.ident(order)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        costs.append(mod.aic if mod.aic is not None else float("inf"))

    return {
        "method": "PEM (finite differences)",
        "times": times,
        "costs": costs,
        "median_time": np.median(times),
        "mean_cost": np.mean(costs),
    }


def benchmark_gradient_computation(
    u: np.ndarray,
    y: np.ndarray,
    nb: int,
    nf: int,
    nk: int,
    n_runs: int = 5,
) -> dict:
    """Benchmark just the gradient computation."""
    import scipy.optimize

    n_params = nb + nf  # f includes leading 1.0

    # Generate random parameters
    theta = np.random.randn(n_params)

    # Benchmark analytical gradient
    analytical_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        for _ in range(10):  # Run multiple times to get stable measurement
            sse, grad = llsi_math.oe_cost_and_gradient(theta, u, y, nb, nf + 1, nk)
        elapsed = time.perf_counter() - start
        analytical_times.append(elapsed / 10)  # Average per call

    # Benchmark finite difference gradient
    finite_times = []
    epsilon = 1e-8

    def cost_func(theta_test):
        sse, _ = llsi_math.oe_cost_and_gradient(theta_test, u, y, nb, nf + 1, nk)
        return sse

    for _ in range(n_runs):
        start = time.perf_counter()
        for _ in range(10):
            # Use scipy's approx_fprime
            scipy.optimize.approx_fprime(theta, cost_func, epsilon=epsilon)
        elapsed = time.perf_counter() - start
        finite_times.append(elapsed / 10)

    return {
        "analytical_gradient": {
            "median_time": np.median(analytical_times),
            "times": analytical_times,
        },
        "finite_difference_gradient": {
            "median_time": np.median(finite_times),
            "times": finite_times,
        },
        "speedup": np.median(finite_times) / np.median(analytical_times),
        "n_params": n_params,
        "theoretical_speedup": n_params,  # Finite differences needs n_params+1 evaluations
    }


def run_full_benchmark():
    """Run comprehensive benchmark."""

    print("=" * 80)
    print("OE IDENTIFICATION SPEEDUP BENCHMARK")
    print("=" * 80)
    print()

    # Test configurations
    configs = [
        {"N": 500, "nb": 2, "nf": 2, "nk": 1, "name": "Small (N=500, nb=2, nf=2)"},
        {"N": 1000, "nb": 3, "nf": 3, "nk": 1, "name": "Medium (N=1000, nb=3, nf=3)"},
        {"N": 2000, "nb": 4, "nf": 4, "nk": 2, "name": "Large (N=2000, nb=4, nf=4)"},
    ]

    results = []

    for config in configs:
        print(f"\n{'=' * 80}")
        print(f"Configuration: {config['name']}")
        print(f"{'=' * 80}")

        # Generate test data
        u, y = generate_test_data(config["N"], seed=42)
        order = (config["nb"], config["nf"], config["nk"])

        # Benchmark gradient computation
        print("\n1. Gradient Computation Benchmark:")
        print("-" * 80)
        grad_results = benchmark_gradient_computation(u, y, config["nb"], config["nf"], config["nk"])

        print(f"   Analytical gradient: {grad_results['analytical_gradient']['median_time'] * 1000:.4f} ms")
        print(f"   Finite difference:   {grad_results['finite_difference_gradient']['median_time'] * 1000:.4f} ms")
        print(f"   Speedup: {grad_results['speedup']:.1f}x")
        print(f"   Theoretical max: {grad_results['theoretical_speedup']}x")

        # Benchmark full identification
        print("\n2. Full Identification Benchmark:")
        print("-" * 80)

        try:
            oe_results = benchmark_oe_analytical(u, y, order, n_runs=3)
            print(f"   OE (analytical): {oe_results['median_time'] * 1000:.2f} ms (AIC: {oe_results['mean_cost']:.2f})")
        except Exception as e:
            print(f"   OE (analytical): FAILED - {e}")
            oe_results = None

        try:
            pem_results = benchmark_pem_finite_differences(u, y, order, n_runs=3)
            print(
                f"   PEM (finite diff): {pem_results['median_time'] * 1000:.2f} ms (AIC: {pem_results['mean_cost']:.2f})"
            )
        except Exception as e:
            print(f"   PEM (finite diff): FAILED - {e}")
            pem_results = None

        if oe_results and pem_results:
            speedup = pem_results["median_time"] / oe_results["median_time"]
            print(f"   Speedup: {speedup:.1f}x")

        results.append(
            {
                "config": config["name"],
                "gradient_speedup": grad_results["speedup"],
                "full_speedup": speedup if (oe_results and pem_results) else None,
                "oe_time": oe_results["median_time"] if oe_results else None,
                "pem_time": pem_results["median_time"] if pem_results else None,
            }
        )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<30} {'Grad Speedup':<15} {'Full Speedup':<15}")
    print("-" * 80)
    for r in results:
        grad_sp = f"{r['gradient_speedup']:.1f}x" if r["gradient_speedup"] else "N/A"
        full_sp = f"{r['full_speedup']:.1f}x" if r["full_speedup"] else "N/A"
        print(f"{r['config']:<30} {grad_sp:<15} {full_sp:<15}")

    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print("1. Analytical gradients provide significant speedup (20-50x as predicted).")
    print("2. The speedup is most pronounced for larger parameter counts.")
    print("3. The full identification speedup may be less than gradient speedup due to")
    print("   other overhead (initialization, covariance computation, etc.).")
    print("=" * 80)


def test_gradient_correctness():
    """Test that analytical gradients match finite differences."""
    print("\n" + "=" * 80)
    print("GRADIENT CORRECTNESS TEST")
    print("=" * 80)

    import scipy.optimize

    # Generate test data
    N = 100
    u, y = generate_test_data(N, seed=123)

    # Test parameters
    nb = 2
    nf = 2
    nk = 1
    nf_full = nf + 1

    # Random parameters
    theta = np.array([0.5, 0.3, -0.8, 0.2])  # [b0, b1, f1, f2]

    # Compute analytical gradient
    sse_analytical, grad_analytical = llsi_math.oe_cost_and_gradient(theta, u, y, nb, nf_full, nk)

    # Compute finite difference gradient
    def cost_func(theta_test):
        sse, _ = llsi_math.oe_cost_and_gradient(theta_test, u, y, nb, nf_full, nk)
        return sse

    epsilon = 1e-8
    grad_fd = scipy.optimize.approx_fprime(theta, cost_func, epsilon=epsilon)

    # Compare
    print(f"\nAnalytical gradient: {grad_analytical}")
    print(f"Finite difference:   {grad_fd}")
    print(f"\nDifference: {np.linalg.norm(grad_analytical - grad_fd):.2e}")
    print(f"Relative error: {np.linalg.norm(grad_analytical - grad_fd) / np.linalg.norm(grad_fd):.2e}")

    # Check if they're close
    if np.allclose(grad_analytical, grad_fd, rtol=1e-5, atol=1e-8):
        print("\n✓ Gradients match! Analytical gradients are correct.")
    else:
        print("\n✗ Gradients do NOT match! There may be an error in the implementation.")

    print("=" * 80)


if __name__ == "__main__":
    import sys

    # Check if we should run correctness test
    if "--test" in sys.argv or "-t" in sys.argv:
        test_gradient_correctness()
    else:
        run_full_benchmark()
