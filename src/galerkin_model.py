"""Galerkin matrices for a simply supported axially moving web."""

import numpy as np

from parameters import (
    flexural_rigidity_nm2,
    initial_tension_n,
    linear_density_kg_per_m,
    span_length_m,
)


def modal_shape(x_m: float | np.ndarray, mode_number: int) -> float | np.ndarray:
    """Return phi_n(x) = sin(n*pi*x/L)."""
    return np.sin(mode_number * np.pi * np.asarray(x_m) / span_length_m)


def modal_shape_first_derivative(
    x_m: float | np.ndarray, mode_number: int
) -> float | np.ndarray:
    """Return the first spatial derivative of phi_n(x) [1/m]."""
    wave_number_per_m = mode_number * np.pi / span_length_m
    return wave_number_per_m * np.cos(wave_number_per_m * np.asarray(x_m))


def modal_shape_second_derivative(
    x_m: float | np.ndarray, mode_number: int
) -> float | np.ndarray:
    """Return the second spatial derivative of phi_n(x) [1/m^2]."""
    wave_number_per_m = mode_number * np.pi / span_length_m
    return -(wave_number_per_m**2) * np.sin(wave_number_per_m * np.asarray(x_m))


def assemble_galerkin_matrices(
    transport_speed_m_per_s: float,
    number_of_modes: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the analytical mass, gyroscopic, and effective stiffness matrices.

    The sine-mode integrals are orthogonal for M and K_eff. For C,
    integral(phi_i * d(phi_j)/dx) is zero for equal-parity modes and
    2*i*j/(i**2-j**2) otherwise.
    """
    if number_of_modes < 1:
        raise ValueError("number_of_modes must be a positive integer")
    if not np.isfinite(transport_speed_m_per_s):
        raise ValueError("transport_speed_m_per_s must be finite")

    mode_numbers = np.arange(1, number_of_modes + 1, dtype=float)
    mass_diagonal = linear_density_kg_per_m * span_length_m / 2.0 * np.ones(
        number_of_modes
    )
    mass_matrix = np.diag(mass_diagonal)

    coriolis_matrix = np.zeros((number_of_modes, number_of_modes), dtype=float)
    coriolis_factor = 2.0 * linear_density_kg_per_m * transport_speed_m_per_s
    for row, mode_i in enumerate(mode_numbers):
        for column, mode_j in enumerate(mode_numbers):
            if (int(mode_i) + int(mode_j)) % 2 == 1:
                overlap = 2.0 * mode_i * mode_j / (mode_i**2 - mode_j**2)
                coriolis_matrix[row, column] = coriolis_factor * overlap

    effective_tension_n = (
        initial_tension_n - linear_density_kg_per_m * transport_speed_m_per_s**2
    )
    bending_diagonal = (
        flexural_rigidity_nm2
        * mode_numbers**4
        * np.pi**4
        / (2.0 * span_length_m**3)
    )
    tension_diagonal = (
        effective_tension_n
        * mode_numbers**2
        * np.pi**2
        / (2.0 * span_length_m)
    )
    stiffness_matrix = np.diag(bending_diagonal + tension_diagonal)

    return mass_matrix, coriolis_matrix, stiffness_matrix


def _quadrature_reference_matrices(
    transport_speed_m_per_s: float,
    number_of_modes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independently evaluate the defining integrals with Gauss-Legendre points."""
    nodes, weights = np.polynomial.legendre.leggauss(128)
    x_m = (nodes + 1.0) * span_length_m / 2.0
    weights_m = weights * span_length_m / 2.0
    mass_matrix = np.zeros((number_of_modes, number_of_modes), dtype=float)
    coriolis_matrix = np.zeros_like(mass_matrix)
    stiffness_matrix = np.zeros_like(mass_matrix)
    effective_tension_n = (
        initial_tension_n - linear_density_kg_per_m * transport_speed_m_per_s**2
    )

    for row, mode_i in enumerate(range(1, number_of_modes + 1)):
        phi_i = modal_shape(x_m, mode_i)
        dphi_i = modal_shape_first_derivative(x_m, mode_i)
        d2phi_i = modal_shape_second_derivative(x_m, mode_i)
        for column, mode_j in enumerate(range(1, number_of_modes + 1)):
            phi_j = modal_shape(x_m, mode_j)
            dphi_j = modal_shape_first_derivative(x_m, mode_j)
            d2phi_j = modal_shape_second_derivative(x_m, mode_j)
            mass_matrix[row, column] = linear_density_kg_per_m * np.dot(
                weights_m, phi_i * phi_j
            )
            coriolis_matrix[row, column] = (
                2.0
                * linear_density_kg_per_m
                * transport_speed_m_per_s
                * np.dot(weights_m, phi_i * dphi_j)
            )
            stiffness_matrix[row, column] = (
                flexural_rigidity_nm2 * np.dot(weights_m, d2phi_i * d2phi_j)
                + effective_tension_n * np.dot(weights_m, dphi_i * dphi_j)
            )

    return mass_matrix, coriolis_matrix, stiffness_matrix


def _assert_positive_definite(matrix: np.ndarray, matrix_name: str) -> None:
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise AssertionError(f"{matrix_name} is not positive definite") from error


def _validate_matrices(
    transport_speed_m_per_s: float,
    number_of_modes: int,
    mass_matrix: np.ndarray,
    coriolis_matrix: np.ndarray,
    stiffness_matrix: np.ndarray,
) -> dict[str, bool]:
    expected_shape = (number_of_modes, number_of_modes)
    assert mass_matrix.shape == expected_shape
    assert coriolis_matrix.shape == expected_shape
    assert stiffness_matrix.shape == expected_shape
    assert np.isfinite(mass_matrix).all()
    assert np.isfinite(coriolis_matrix).all()
    assert np.isfinite(stiffness_matrix).all()

    checks = {
        "M symmetric": np.allclose(mass_matrix, mass_matrix.T),
        "C anti-symmetric": np.allclose(
            coriolis_matrix.T, -coriolis_matrix, rtol=1e-10, atol=1e-12
        ),
        "C diagonal zero": np.allclose(
            np.diag(coriolis_matrix), 0.0, rtol=0.0, atol=1e-12
        ),
        "K_eff symmetric": np.allclose(
            stiffness_matrix, stiffness_matrix.T, rtol=1e-10, atol=1e-12
        ),
    }
    assert all(checks.values()), f"Matrix symmetry check failed: {checks}"
    _assert_positive_definite(mass_matrix, "M")
    checks["M positive definite"] = True

    if transport_speed_m_per_s in (10.0, 45.0):
        _assert_positive_definite(stiffness_matrix, "K_eff")
        checks["K_eff positive definite"] = True

    reference_matrices = _quadrature_reference_matrices(
        transport_speed_m_per_s, number_of_modes
    )
    for matrix, reference in zip(
        (mass_matrix, coriolis_matrix, stiffness_matrix), reference_matrices
    ):
        assert np.allclose(matrix, reference, rtol=1e-10, atol=2e-10), (
            "Analytical matrix does not agree with independent quadrature"
        )
    checks["analytical integrals match quadrature"] = True

    return checks


def _validate_zero_speed(number_of_modes: int) -> dict[str, bool]:
    mass_matrix, coriolis_matrix, stiffness_matrix = assemble_galerkin_matrices(
        transport_speed_m_per_s=0.0,
        number_of_modes=number_of_modes,
    )
    assert np.allclose(coriolis_matrix, 0.0, atol=1e-12)

    mode_numbers = np.arange(1, number_of_modes + 1, dtype=float)
    bending_contribution = np.diag(
        flexural_rigidity_nm2
        * mode_numbers**4
        * np.pi**4
        / (2.0 * span_length_m**3)
    )
    initial_tension_contribution = np.diag(
        initial_tension_n
        * mode_numbers**2
        * np.pi**2
        / (2.0 * span_length_m)
    )
    assert np.allclose(
        stiffness_matrix,
        bending_contribution + initial_tension_contribution,
        rtol=1e-12,
        atol=1e-12,
    )
    _validate_matrices(
        0.0, number_of_modes, mass_matrix, coriolis_matrix, stiffness_matrix
    )
    return {
        "C is zero at V=0": True,
        "K_eff contains bending and initial-tension contributions at V=0": True,
    }


def _print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"{name} (shape {matrix.shape}):")
    print(np.array2string(matrix, precision=6, suppress_small=False))


def main() -> None:
    number_of_modes = 4
    matrices_by_speed: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for speed_m_per_s in (10.0, 45.0):
        matrices = assemble_galerkin_matrices(
            transport_speed_m_per_s=speed_m_per_s,
            number_of_modes=number_of_modes,
        )
        checks = _validate_matrices(speed_m_per_s, number_of_modes, *matrices)
        matrices_by_speed[speed_m_per_s] = matrices
        print(f"\nChecks at V = {speed_m_per_s:.1f} m/s:")
        for check_name, passed in checks.items():
            print(f"  {check_name}: {'PASS' if passed else 'FAIL'}")

    zero_speed_checks = _validate_zero_speed(number_of_modes)
    print("\nChecks at V = 0.0 m/s:")
    for check_name, passed in zero_speed_checks.items():
        print(f"  {check_name}: {'PASS' if passed else 'FAIL'}")

    mass_matrix, coriolis_matrix, stiffness_matrix = matrices_by_speed[10.0]
    print("\nGalerkin matrices at V = 10.0 m/s:")
    _print_matrix("M", mass_matrix)
    _print_matrix("C (Coriolis/gyroscopic only)", coriolis_matrix)
    _print_matrix("K_eff", stiffness_matrix)


if __name__ == "__main__":
    main()
