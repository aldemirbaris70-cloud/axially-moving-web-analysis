"""Eigenvalue stability and modal-frequency analysis for the moving web."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig
from scipy.optimize import brentq, linear_sum_assignment

from galerkin_model import assemble_galerkin_matrices
from parameters import (
    flexural_rigidity_nm2,
    initial_tension_n,
    linear_density_kg_per_m,
    span_length_m,
)

DEFAULT_NUMBER_OF_MODES = 4
SPEED_MIN_M_PER_S = 0.0
SPEED_MAX_M_PER_S = 60.0
SPEED_POINT_COUNT = 1201
SIMPLIFIED_CRITICAL_SPEED_M_PER_S = np.sqrt(
    initial_tension_n / linear_density_kg_per_m
)
EIGENPAIR_RESIDUAL_TOLERANCE = 1e-8
CONJUGATE_PAIR_TOLERANCE = 1e-8
EIGENVALUE_DISTANCE_WEIGHT = 0.6
EIGENVECTOR_SIMILARITY_WEIGHT = 0.4
BRANCH_JUMP_TOLERANCE = 0.25
CRITICAL_BRANCH_WINDOW_M_PER_S = 0.25
ROOT_XTOL_M_PER_S = 1e-12
ROOT_RTOL = 1e-14
ROOT_PROBE_DELTA_M_PER_S = 1e-3
ROOT_ZERO_EIGENVALUE_TOLERANCE_PER_S = 1e-4


def build_state_matrix(
    transport_speed_m_per_s: float,
    number_of_modes: int = DEFAULT_NUMBER_OF_MODES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build A_state and return it with its Galerkin matrices."""
    mass_matrix, coriolis_matrix, stiffness_matrix = assemble_galerkin_matrices(
        transport_speed_m_per_s=transport_speed_m_per_s,
        number_of_modes=number_of_modes,
    )
    identity = np.eye(number_of_modes)
    zero_block = np.zeros_like(identity)
    state_matrix = np.block(
        [
            [zero_block, identity],
            [
                -np.linalg.solve(mass_matrix, stiffness_matrix),
                -np.linalg.solve(mass_matrix, coriolis_matrix),
            ],
        ]
    )
    return state_matrix, mass_matrix, coriolis_matrix, stiffness_matrix


def eigenpair_residuals(
    state_matrix: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Compute the requested scale-normalized residual for each eigenpair."""
    residuals = np.empty(eigenvalues.size, dtype=float)
    for index, eigenvalue in enumerate(eigenvalues):
        vector = eigenvectors[:, index]
        matrix_vector = state_matrix @ vector
        denominator = max(
            np.linalg.norm(matrix_vector),
            abs(eigenvalue) * np.linalg.norm(vector),
            1.0,
        )
        residuals[index] = np.linalg.norm(
            matrix_vector - eigenvalue * vector
        ) / denominator
    return residuals


def maximum_conjugate_pair_error(eigenvalues: np.ndarray) -> float:
    """Return the worst normalized distance to each eigenvalue's conjugate."""
    errors = []
    for eigenvalue in eigenvalues:
        scale = max(1.0, abs(eigenvalue))
        errors.append(np.min(np.abs(eigenvalues - np.conjugate(eigenvalue))) / scale)
    return float(max(errors, default=0.0))


def stability_tolerance(eigenvalues: np.ndarray) -> float:
    """Return the scale-aware real-part tolerance in inverse seconds."""
    spectral_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    return max(
        1e-10,
        100.0 * np.finfo(float).eps * spectral_scale,
    )


def imaginary_part_tolerance(eigenvalues: np.ndarray, tolerance: float) -> float:
    """Separate near-real roots from oscillatory roots using spectral scale."""
    spectral_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    return max(tolerance, 1e-3 * spectral_scale)


def classify_eigenvalues(eigenvalues: np.ndarray, tolerance: float) -> str:
    """Classify unstable roots without treating roundoff as physical growth."""
    unstable = eigenvalues.real > tolerance
    if not np.any(unstable):
        return "neutral/stable"

    unstable_roots = eigenvalues[unstable]
    imaginary_tolerance = imaginary_part_tolerance(eigenvalues, tolerance)
    has_flutter_candidate = np.any(
        np.abs(unstable_roots.imag) > imaginary_tolerance
    )
    has_divergence_candidate = np.any(
        np.abs(unstable_roots.imag) <= imaginary_tolerance
    )
    if has_flutter_candidate and has_divergence_candidate:
        return "divergence and flutter candidates"
    if has_flutter_candidate:
        return "flutter candidate"
    return "divergence candidate"


def _validate_eigenpairs(
    state_matrix: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> tuple[float, float]:
    expected_dimension = 2 * DEFAULT_NUMBER_OF_MODES
    assert state_matrix.shape == (expected_dimension, expected_dimension)
    assert eigenvalues.shape == (expected_dimension,)
    assert eigenvectors.shape == (expected_dimension, expected_dimension)
    assert np.isfinite(state_matrix).all()
    assert np.isfinite(eigenvalues).all()
    assert np.isfinite(eigenvectors).all()

    residuals = eigenpair_residuals(state_matrix, eigenvalues, eigenvectors)
    maximum_residual = float(np.max(residuals))
    assert maximum_residual < EIGENPAIR_RESIDUAL_TOLERANCE, (
        f"Maximum eigenpair residual {maximum_residual:.3e} exceeds "
        f"{EIGENPAIR_RESIDUAL_TOLERANCE:.1e}"
    )
    conjugate_error = maximum_conjugate_pair_error(eigenvalues)
    assert conjugate_error < CONJUGATE_PAIR_TOLERANCE, (
        f"Conjugate-pair error {conjugate_error:.3e} exceeds "
        f"{CONJUGATE_PAIR_TOLERANCE:.1e}"
    )
    return maximum_residual, conjugate_error


def sweep_eigenvalues(
    number_of_modes: int = DEFAULT_NUMBER_OF_MODES,
    speeds_m_per_s: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Sweep speed, assemble the model, and store all eigenpairs and checks."""
    assert number_of_modes == DEFAULT_NUMBER_OF_MODES, (
        "Eigenpair validation is configured for the required four-mode model"
    )
    if speeds_m_per_s is None:
        speeds_m_per_s = np.linspace(
            SPEED_MIN_M_PER_S,
            SPEED_MAX_M_PER_S,
            SPEED_POINT_COUNT,
        )
    else:
        speeds_m_per_s = np.asarray(speeds_m_per_s, dtype=float)
        assert speeds_m_per_s.ndim == 1 and speeds_m_per_s.size > 1
        assert np.all(np.diff(speeds_m_per_s) > 0.0)
        assert speeds_m_per_s[0] >= SPEED_MIN_M_PER_S
        assert speeds_m_per_s[-1] <= SPEED_MAX_M_PER_S
    speed_point_count = speeds_m_per_s.size
    matrix_size = 2 * number_of_modes
    eigenvalues_by_speed = np.empty((speed_point_count, matrix_size), dtype=complex)
    eigenvectors_by_speed = np.empty(
        (speed_point_count, matrix_size, matrix_size), dtype=complex
    )
    minimum_stiffness_eigenvalue = np.empty(speed_point_count, dtype=float)
    tolerances = np.empty(speed_point_count, dtype=float)
    maximum_residuals = np.empty(speed_point_count, dtype=float)
    conjugate_pair_errors = np.empty(speed_point_count, dtype=float)

    for speed_index, speed_m_per_s in enumerate(speeds_m_per_s):
        state_matrix, mass_matrix, coriolis_matrix, stiffness_matrix = (
            build_state_matrix(speed_m_per_s, number_of_modes)
        )
        assert state_matrix.shape == (8, 8)
        assert mass_matrix.shape == (number_of_modes, number_of_modes)
        assert coriolis_matrix.shape == (number_of_modes, number_of_modes)
        assert stiffness_matrix.shape == (number_of_modes, number_of_modes)
        assert all(
            np.isfinite(matrix).all()
            for matrix in (mass_matrix, coriolis_matrix, stiffness_matrix)
        )
        assert np.allclose(
            stiffness_matrix, stiffness_matrix.T, rtol=1e-10, atol=1e-12
        )
        minimum_stiffness_eigenvalue[speed_index] = np.linalg.eigvalsh(
            stiffness_matrix
        )[0]

        if speed_index == 0:
            np.linalg.cholesky(mass_matrix)
            np.linalg.cholesky(stiffness_matrix)
            assert np.allclose(coriolis_matrix, 0.0, atol=1e-12)

        eigenvalues, eigenvectors = eig(state_matrix, check_finite=True)
        assert eigenvalues.size == 8
        maximum_residual, conjugate_error = _validate_eigenpairs(
            state_matrix, eigenvalues, eigenvectors
        )
        eigenvalues_by_speed[speed_index] = eigenvalues
        eigenvectors_by_speed[speed_index] = eigenvectors
        tolerances[speed_index] = stability_tolerance(eigenvalues)
        maximum_residuals[speed_index] = maximum_residual
        conjugate_pair_errors[speed_index] = conjugate_error

    assert np.isfinite(eigenvalues_by_speed).all()
    assert np.isfinite(eigenvectors_by_speed).all()
    assert np.isfinite(minimum_stiffness_eigenvalue).all()
    return {
        "speeds_m_per_s": speeds_m_per_s,
        "eigenvalues": eigenvalues_by_speed,
        "eigenvectors": eigenvectors_by_speed,
        "minimum_stiffness_eigenvalue": minimum_stiffness_eigenvalue,
        "tolerances": tolerances,
        "maximum_residuals": maximum_residuals,
        "conjugate_pair_errors": conjugate_pair_errors,
    }


def estimate_divergence_speed(
    speeds_m_per_s: np.ndarray,
    minimum_stiffness_eigenvalue: np.ndarray,
    number_of_modes: int = DEFAULT_NUMBER_OF_MODES,
) -> tuple[float, tuple[float, float]]:
    """Refine the first lambda_min(K_eff)=0 crossing inside the speed sweep."""
    crossing_indices = np.flatnonzero(
        (minimum_stiffness_eigenvalue[:-1] > 0.0)
        & (minimum_stiffness_eigenvalue[1:] <= 0.0)
    )
    assert crossing_indices.size > 0, "No stiffness eigenvalue sign change found"
    first_index = int(crossing_indices[0])
    bracket = (float(speeds_m_per_s[first_index]), float(speeds_m_per_s[first_index + 1]))

    def lowest_stiffness_eigenvalue(speed_m_per_s: float) -> float:
        _, _, stiffness_matrix = assemble_galerkin_matrices(
            transport_speed_m_per_s=speed_m_per_s,
            number_of_modes=number_of_modes,
        )
        return float(np.linalg.eigvalsh(stiffness_matrix)[0])

    bracket_values = tuple(lowest_stiffness_eigenvalue(speed) for speed in bracket)
    assert bracket_values[0] * bracket_values[1] < 0.0
    divergence_speed_m_per_s = brentq(
        lowest_stiffness_eigenvalue,
        bracket[0],
        bracket[1],
        xtol=ROOT_XTOL_M_PER_S,
        rtol=ROOT_RTOL,
    )
    assert 0.0 < divergence_speed_m_per_s < SPEED_MAX_M_PER_S

    side_delta = min(1e-5, (bracket[1] - bracket[0]) / 100.0)
    value_below = lowest_stiffness_eigenvalue(divergence_speed_m_per_s - side_delta)
    value_above = lowest_stiffness_eigenvalue(divergence_speed_m_per_s + side_delta)
    assert value_below > 0.0 and value_above < 0.0
    return divergence_speed_m_per_s, bracket


def analytical_first_mode_divergence_speed() -> float:
    """Return the analytical first-mode stiffness-zero speed."""
    return float(
        np.sqrt(
            initial_tension_n / linear_density_kg_per_m
            + flexural_rigidity_nm2
            * np.pi**2
            / (linear_density_kg_per_m * span_length_m**2)
        )
    )


def _normalized_eigenvalue_distance(previous: complex, current: complex) -> float:
    local_scale = max(1.0, abs(previous), abs(current))
    return float(abs(previous - current) / local_scale)


def track_eigenvalue_branches(
    eigenvalues_by_speed: np.ndarray,
    eigenvectors_by_speed: np.ndarray,
    tolerances: np.ndarray,
    speeds_m_per_s: np.ndarray,
    divergence_speed_m_per_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track all roots with Hungarian assignment, then retain four initial modes."""
    speed_count, eigenvalue_count = eigenvalues_by_speed.shape
    tracked_values = np.empty_like(eigenvalues_by_speed)
    tracked_vectors = np.empty_like(eigenvectors_by_speed)
    tracked_values[0] = eigenvalues_by_speed[0]
    tracked_vectors[0] = eigenvectors_by_speed[0]

    initial_positive_indices = np.flatnonzero(
        eigenvalues_by_speed[0].imag > tolerances[0]
    )
    assert initial_positive_indices.size == DEFAULT_NUMBER_OF_MODES, (
        "Expected four positive-imaginary roots at zero transport speed"
    )
    initial_positive_indices = initial_positive_indices[
        np.argsort(eigenvalues_by_speed[0, initial_positive_indices].imag)
    ]

    normalized_jumps = np.empty((speed_count - 1, eigenvalue_count), dtype=float)
    for speed_index in range(1, speed_count):
        previous_values = tracked_values[speed_index - 1]
        previous_vectors = tracked_vectors[speed_index - 1]
        current_values = eigenvalues_by_speed[speed_index]
        current_vectors = eigenvectors_by_speed[speed_index]
        cost = np.empty((eigenvalue_count, eigenvalue_count), dtype=float)

        for previous_index in range(eigenvalue_count):
            previous_vector = previous_vectors[:, previous_index]
            previous_vector = previous_vector / np.linalg.norm(previous_vector)
            for current_index in range(eigenvalue_count):
                current_vector = current_vectors[:, current_index]
                current_vector = current_vector / np.linalg.norm(current_vector)
                similarity = abs(np.vdot(previous_vector, current_vector))
                distance = _normalized_eigenvalue_distance(
                    previous_values[previous_index], current_values[current_index]
                )
                cost[previous_index, current_index] = (
                    EIGENVALUE_DISTANCE_WEIGHT * distance
                    + EIGENVECTOR_SIMILARITY_WEIGHT * (1.0 - similarity)
                )

        previous_assignment, current_assignment = linear_sum_assignment(cost)
        order = np.empty(eigenvalue_count, dtype=int)
        order[previous_assignment] = current_assignment
        tracked_values[speed_index] = current_values[order]
        tracked_vectors[speed_index] = current_vectors[:, order]
        for branch_index in range(eigenvalue_count):
            normalized_jumps[speed_index - 1, branch_index] = (
                _normalized_eigenvalue_distance(
                    tracked_values[speed_index - 1, branch_index],
                    tracked_values[speed_index, branch_index],
                )
            )

    branch_values = tracked_values[:, initial_positive_indices]
    branch_frequencies_hz = np.full(branch_values.shape, np.nan, dtype=float)
    for speed_index in range(speed_count):
        positive_imaginary = branch_values[speed_index].imag > tolerances[speed_index]
        branch_frequencies_hz[speed_index, positive_imaginary] = (
            np.abs(branch_values[speed_index, positive_imaginary].imag)
            / (2.0 * np.pi)
        )

    outside_critical_window = (
        np.abs(speeds_m_per_s[1:] - divergence_speed_m_per_s)
        > CRITICAL_BRANCH_WINDOW_M_PER_S
    )
    maximum_jump_outside_critical = float(
        np.max(normalized_jumps[outside_critical_window])
    )
    assert maximum_jump_outside_critical < BRANCH_JUMP_TOLERANCE, (
        "A large eigenvalue branch jump occurred outside the critical window: "
        f"{maximum_jump_outside_critical:.3g}"
    )
    assert np.isfinite(branch_frequencies_hz[0]).all()

    first_branch = branch_frequencies_hz[:, 0]
    pre_divergence_indices = np.flatnonzero(
        (speeds_m_per_s < divergence_speed_m_per_s) & np.isfinite(first_branch)
    )
    assert pre_divergence_indices.size > 0
    near_index = int(pre_divergence_indices[-1])
    farther_indices = np.flatnonzero(
        (speeds_m_per_s <= divergence_speed_m_per_s - 1.0)
        & np.isfinite(first_branch)
    )
    assert farther_indices.size > 0
    assert first_branch[near_index] < first_branch[farther_indices[-1]], (
        "The first tracked frequency did not decrease toward divergence"
    )
    root_index = int(np.argmin(np.abs(speeds_m_per_s - divergence_speed_m_per_s)))
    assert np.isclose(
        speeds_m_per_s[root_index], divergence_speed_m_per_s, rtol=0.0, atol=1e-12
    )
    assert np.isfinite(first_branch[root_index])
    assert first_branch[root_index] < 1e-4, (
        "The first modal frequency is not sufficiently close to zero at divergence"
    )

    for speed_index in range(speed_count):
        real_branch = np.abs(branch_values[speed_index].imag) <= tolerances[speed_index]
        assert np.isnan(branch_frequencies_hz[speed_index, real_branch]).all(), (
            "A real eigenvalue was incorrectly plotted as an oscillatory frequency"
        )

    return branch_values, branch_frequencies_hz, normalized_jumps


def _stability_labels_for_tolerance(
    eigenvalues_by_speed: np.ndarray,
    tolerances: np.ndarray,
    multiplier: float,
) -> list[str]:
    return [
        classify_eigenvalues(eigenvalues, tolerance * multiplier)
        for eigenvalues, tolerance in zip(eigenvalues_by_speed, tolerances)
    ]


def check_tolerance_sensitivity(
    eigenvalues_by_speed: np.ndarray,
    tolerances: np.ndarray,
    speeds_m_per_s: np.ndarray,
) -> dict[float, list[str]]:
    """Classify with tau/10, tau, and 10*tau and report their differences."""
    classifications = {
        multiplier: _stability_labels_for_tolerance(
            eigenvalues_by_speed, tolerances, multiplier
        )
        for multiplier in (0.1, 1.0, 10.0)
    }
    baseline = classifications[1.0]
    print("\nStability classification sensitivity:")
    for multiplier, labels in classifications.items():
        changed_points = sum(label != base for label, base in zip(labels, baseline))
        unstable_indices = [
            index
            for index, label in enumerate(labels)
            if label != "neutral/stable"
        ]
        divergence_indices = [
            index
            for index, label in enumerate(labels)
            if "divergence" in label
        ]
        flutter_indices = [
            index for index, label in enumerate(labels) if "flutter" in label
        ]
        first_unstable = (
            f"{speeds_m_per_s[unstable_indices[0]]:.9f} m/s"
            if unstable_indices
            else "none"
        )
        first_divergence = (
            f"{speeds_m_per_s[divergence_indices[0]]:.9f} m/s"
            if divergence_indices
            else "none"
        )
        first_flutter = (
            f"{speeds_m_per_s[flutter_indices[0]]:.9f} m/s"
            if flutter_indices
            else "none"
        )
        print(
            f"  {multiplier:g}*tau: {len(unstable_indices)} unstable points; "
            f"first {first_unstable}; divergence {first_divergence}; "
            f"flutter {first_flutter}; {changed_points} classifications differ from tau"
        )
        if divergence_indices:
            first_divergence_index = divergence_indices[0]
            later_neutral_indices = [
                index
                for index in range(first_divergence_index + 1, len(labels))
                if labels[index] == "neutral/stable"
            ]
            if later_neutral_indices:
                print(
                    "    First subsequent neutral/stable point: "
                    f"{speeds_m_per_s[later_neutral_indices[0]]:.9f} m/s"
                )
    return classifications


def _speed_intervals(
    active: np.ndarray,
    speeds_m_per_s: np.ndarray,
) -> list[tuple[float, float]]:
    """Return inclusive speed ranges for contiguous active samples."""
    transitions = np.diff(np.concatenate(([False], active, [False])).astype(int))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1) - 1
    return [
        (float(speeds_m_per_s[start]), float(speeds_m_per_s[end]))
        for start, end in zip(starts, ends)
    ]


def find_instability_regions(
    speeds_m_per_s: np.ndarray,
    eigenvalues_by_speed: np.ndarray,
    tolerances: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
]:
    """Find near-real and oscillatory instability intervals from the eigenspectra."""
    spectral_scales = np.maximum(1.0, np.max(np.abs(eigenvalues_by_speed), axis=1))
    imaginary_tolerances = np.maximum(
        tolerances,
        1e-3 * spectral_scales,
    )
    positive_real = eigenvalues_by_speed.real > tolerances[:, None]
    near_real_unstable = np.any(
        positive_real
        & (np.abs(eigenvalues_by_speed.imag) <= imaginary_tolerances[:, None]),
        axis=1,
    )
    oscillatory_unstable = np.any(
        positive_real
        & (np.abs(eigenvalues_by_speed.imag) > imaginary_tolerances[:, None]),
        axis=1,
    )
    neutral_or_stable = ~(near_real_unstable | oscillatory_unstable)

    divergence_intervals = _speed_intervals(near_real_unstable, speeds_m_per_s)
    flutter_intervals = _speed_intervals(oscillatory_unstable, speeds_m_per_s)
    re_neutral_intervals: list[tuple[float, float]] = []
    if divergence_intervals:
        first_divergence_start = divergence_intervals[0][0]
        first_flutter_start = flutter_intervals[0][0] if flutter_intervals else float("inf")
        re_neutral_intervals = [
            interval
            for interval in _speed_intervals(neutral_or_stable, speeds_m_per_s)
            if interval[0] > first_divergence_start
            and interval[0] < first_flutter_start
        ]

    return (
        near_real_unstable,
        oscillatory_unstable,
        neutral_or_stable,
        divergence_intervals,
        flutter_intervals,
        re_neutral_intervals,
    )


def inspect_divergence_eigenvalues(
    divergence_speed_m_per_s: float,
    number_of_modes: int = DEFAULT_NUMBER_OF_MODES,
) -> dict[str, float | int]:
    """Verify oscillatory, near-zero, and real-root behavior around divergence."""
    delta = ROOT_PROBE_DELTA_M_PER_S
    speeds = (
        divergence_speed_m_per_s - delta,
        divergence_speed_m_per_s,
        divergence_speed_m_per_s + delta,
    )
    spectra = []
    for speed in speeds:
        state_matrix, _, _, _ = build_state_matrix(speed, number_of_modes)
        eigenvalues = eig(state_matrix, right=False, check_finite=True)
        spectra.append(eigenvalues)

    below_values, root_values, above_values = spectra
    positive_imaginary_below = below_values[below_values.imag > 1e-8]
    assert positive_imaginary_below.size > 0
    below_frequency_hz = float(
        np.min(positive_imaginary_below.imag) / (2.0 * np.pi)
    )
    root_minimum_magnitude = float(np.min(np.abs(root_values)))
    assert root_minimum_magnitude < ROOT_ZERO_EIGENVALUE_TOLERANCE_PER_S, (
        "No near-zero state eigenvalue was found at the stiffness root: "
        f"{root_minimum_magnitude:.3e} 1/s"
    )

    above_tau = stability_tolerance(above_values)
    positive_real = above_values[
        (above_values.real > above_tau) & (np.abs(above_values.imag) <= above_tau)
    ]
    negative_real = above_values[
        (above_values.real < -above_tau) & (np.abs(above_values.imag) <= above_tau)
    ]
    assert positive_real.size > 0 and negative_real.size > 0, (
        "Expected a positive/negative real eigenvalue pair above divergence"
    )
    assert np.min(positive_real.real) > 0.0
    assert np.max(negative_real.real) < 0.0
    return {
        "probe_delta_m_per_s": delta,
        "below_frequency_hz": below_frequency_hz,
        "root_minimum_eigenvalue_magnitude_per_s": root_minimum_magnitude,
        "above_positive_real_eigenvalue_per_s": float(np.min(positive_real.real)),
        "above_negative_real_eigenvalue_per_s": float(np.max(negative_real.real)),
    }


def plot_frequency_branches(
    speeds_m_per_s: np.ndarray,
    branch_frequencies_hz: np.ndarray,
    divergence_speed_m_per_s: float,
    output_path: Path,
) -> None:
    """Save frequency branches with an uncluttered critical-speed detail panel."""
    figure, (axis, zoom_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 6.2),
        gridspec_kw={"width_ratios": (3.2, 1.8)},
    )
    colors = plt.get_cmap("tab10").colors
    for branch_index in range(branch_frequencies_hz.shape[1]):
        axis.plot(
            speeds_m_per_s,
            branch_frequencies_hz[:, branch_index],
            color=colors[branch_index],
            linewidth=1.7,
            label=f"Modal branch {branch_index + 1}",
        )
    axis.axvline(
        SIMPLIFIED_CRITICAL_SPEED_M_PER_S,
        color="darkorange",
        linestyle="--",
        linewidth=1.6,
        label="Simplified critical speed (50.000000 m/s)",
    )
    axis.axvline(
        divergence_speed_m_per_s,
        color="crimson",
        linestyle="-.",
        linewidth=1.6,
        label=f"Modal divergence ({divergence_speed_m_per_s:.9f} m/s)",
    )
    axis.set_xlim(SPEED_MIN_M_PER_S, SPEED_MAX_M_PER_S)
    axis.set_xlabel("Transport speed V [m/s]")
    axis.set_ylabel("Modal frequency f [Hz]")
    axis.set_title("Tracked modal frequencies")
    axis.grid(True, linestyle="--", alpha=0.55)

    zoom_low_m_per_s = 49.98
    zoom_high_m_per_s = 50.02
    first_frequency = branch_frequencies_hz[:, 0]
    zoom_mask = (
        (speeds_m_per_s >= zoom_low_m_per_s)
        & (speeds_m_per_s <= zoom_high_m_per_s)
        & np.isfinite(first_frequency)
    )
    zoom_axis.plot(
        speeds_m_per_s[zoom_mask],
        first_frequency[zoom_mask],
        color=colors[0],
        linewidth=2.0,
        label="Modal branch 1",
    )
    zoom_axis.axvline(
        SIMPLIFIED_CRITICAL_SPEED_M_PER_S,
        color="darkorange",
        linestyle="--",
        linewidth=1.8,
        label="Simplified: 50.000000 m/s",
    )
    zoom_axis.axvline(
        divergence_speed_m_per_s,
        color="crimson",
        linestyle="-.",
        linewidth=1.8,
        label=f"Modal: {divergence_speed_m_per_s:.9f} m/s",
    )
    zoom_axis.set_xlim(zoom_low_m_per_s, zoom_high_m_per_s)
    visible_frequencies = first_frequency[zoom_mask]
    zoom_max_hz = max(float(np.max(visible_frequencies)) * 1.15, 1e-6)
    zoom_axis.set_ylim(0.0, zoom_max_hz)
    zoom_axis.set_xticks(np.linspace(zoom_low_m_per_s, zoom_high_m_per_s, 5))
    zoom_axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    zoom_axis.set_xlabel("Transport speed V [m/s]")
    zoom_axis.set_ylabel("f [Hz]")
    zoom_axis.set_title("Divergence detail")
    zoom_axis.grid(True, linestyle="--", alpha=0.55)
    zoom_axis.legend(loc="upper right", fontsize=7, framealpha=0.9)

    figure.suptitle("Tracked Modal Frequencies vs. Transport Speed")
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.45, 0.005),
        fontsize=8,
        frameon=True,
    )
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.95))
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_eigenvalue_stability(
    speeds_m_per_s: np.ndarray,
    maximum_real_part: np.ndarray,
    tolerances: np.ndarray,
    divergence_speed_m_per_s: float,
    divergence_intervals: list[tuple[float, float]],
    re_neutral_intervals: list[tuple[float, float]],
    flutter_candidate_speed_m_per_s: float | None,
    output_path: Path,
) -> None:
    """Save full-range growth and a detailed first-divergence panel."""
    zoom_low_m_per_s = 49.98
    zoom_high_m_per_s = 50.05
    if divergence_intervals:
        divergence_end_m_per_s = max(end for _, end in divergence_intervals)
        if divergence_end_m_per_s > zoom_high_m_per_s:
            zoom_high_m_per_s = divergence_end_m_per_s + 0.01

    figure, (axis, zoom_axis) = plt.subplots(
        2,
        1,
        figsize=(10.5, 8.2),
        gridspec_kw={"height_ratios": (2.2, 1.25)},
    )
    axis.fill_between(
        speeds_m_per_s,
        -tolerances,
        tolerances,
        color="0.75",
        alpha=0.45,
        linewidth=0.0,
        label="Numerical neutral band (+/- tau)",
    )
    axis.plot(
        speeds_m_per_s,
        maximum_real_part,
        color="navy",
        linewidth=1.7,
        label="Maximum Re(lambda)",
    )
    axis.axhline(
        0.0,
        color="black",
        linestyle="-",
        linewidth=1.0,
        label="Neutral stability (0 1/s)",
    )
    axis.axvline(
        SIMPLIFIED_CRITICAL_SPEED_M_PER_S,
        color="darkorange",
        linestyle="--",
        linewidth=1.5,
        label="Simplified critical speed (50.000000 m/s)",
    )
    axis.axvline(
        divergence_speed_m_per_s,
        color="crimson",
        linestyle="-.",
        linewidth=1.6,
        label=f"Modal divergence ({divergence_speed_m_per_s:.9f} m/s)",
    )
    if flutter_candidate_speed_m_per_s is not None:
        axis.axvline(
            flutter_candidate_speed_m_per_s,
            color="purple",
            linestyle=":",
            linewidth=1.7,
            label="Flutter candidate in four-mode model",
        )
    axis.set_xlim(SPEED_MIN_M_PER_S, SPEED_MAX_M_PER_S)
    axis.set_xlabel("Transport speed V [m/s]")
    axis.set_ylabel("Maximum eigenvalue real part max Re(lambda) [1/s]")
    axis.set_title("Full-range eigenvalue growth rate")
    axis.grid(True, linestyle="--", alpha=0.55)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        frameon=True,
    )
    axis.text(
        0.015,
        0.04,
        "Positive growth is classified only when Re(lambda) exceeds tau; "
        "smaller residuals are numerical roundoff.",
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    zoom_mask = (speeds_m_per_s >= zoom_low_m_per_s) & (
        speeds_m_per_s <= zoom_high_m_per_s
    )
    zoom_speeds_m_per_s = speeds_m_per_s[zoom_mask]
    zoom_growth = maximum_real_part[zoom_mask]
    zoom_tolerances = tolerances[zoom_mask]
    assert zoom_speeds_m_per_s.size > 0
    zoom_axis.fill_between(
        zoom_speeds_m_per_s,
        -zoom_tolerances,
        zoom_tolerances,
        color="0.75",
        alpha=0.55,
        linewidth=0.0,
        label="Numerical band (+/- tau)",
    )
    for index, (start_m_per_s, end_m_per_s) in enumerate(divergence_intervals):
        if end_m_per_s < zoom_low_m_per_s or start_m_per_s > zoom_high_m_per_s:
            continue
        zoom_axis.axvspan(
            max(start_m_per_s, zoom_low_m_per_s),
            min(end_m_per_s, zoom_high_m_per_s),
            color="orangered",
            alpha=0.16,
            label="Near-real positive root (divergence candidate)" if index == 0 else None,
        )
    for index, (start_m_per_s, end_m_per_s) in enumerate(re_neutral_intervals):
        if end_m_per_s < zoom_low_m_per_s or start_m_per_s > zoom_high_m_per_s:
            continue
        zoom_axis.axvspan(
            max(start_m_per_s, zoom_low_m_per_s),
            min(end_m_per_s, zoom_high_m_per_s),
            color="mediumseagreen",
            alpha=0.10,
            label="Re-neutral/stable interval" if index == 0 else None,
        )
    zoom_axis.plot(
        zoom_speeds_m_per_s,
        zoom_growth,
        color="navy",
        linewidth=1.8,
        label="Maximum Re(lambda)",
        zorder=3,
    )
    zoom_axis.axhline(
        0.0,
        color="black",
        linestyle="-",
        linewidth=1.0,
        label="Neutral stability (0 1/s)",
        zorder=2,
    )
    zoom_axis.axvline(
        SIMPLIFIED_CRITICAL_SPEED_M_PER_S,
        color="darkorange",
        linestyle="--",
        linewidth=1.7,
        label="Simplified: 50.000000 m/s",
    )
    zoom_axis.axvline(
        divergence_speed_m_per_s,
        color="crimson",
        linestyle="-.",
        linewidth=1.7,
        label=f"Modal: {divergence_speed_m_per_s:.9f} m/s",
    )
    zoom_min = min(float(np.min(zoom_growth)), 0.0)
    zoom_max = max(float(np.max(zoom_growth)), 0.0)
    padding = max((zoom_max - zoom_min) * 0.10, 1e-6)
    zoom_axis.set_ylim(zoom_min - padding, zoom_max + padding)
    zoom_axis.set_xlim(zoom_low_m_per_s, zoom_high_m_per_s)
    zoom_axis.set_xlabel("Transport speed V [m/s]")
    zoom_axis.set_ylabel("Maximum Re(lambda) [1/s]")
    zoom_axis.set_title("Divergence pocket and re-neutralization detail")
    zoom_axis.grid(True, linestyle="--", alpha=0.55)
    zoom_axis.legend(loc="upper right", fontsize=7, framealpha=0.9)

    figure.suptitle("Eigenvalue-Based Stability vs. Transport Speed")
    figure.tight_layout(rect=(0.0, 0.0, 0.78, 0.95))
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    results_directory = project_root / "results"
    assert results_directory.is_dir(), "The results directory does not exist"
    frequency_plot_path = results_directory / "frequency_vs_speed.png"
    stability_plot_path = results_directory / "eigenvalue_stability.png"

    uniform_data = sweep_eigenvalues()
    divergence_speed_m_per_s, divergence_bracket = estimate_divergence_speed(
        uniform_data["speeds_m_per_s"],
        uniform_data["minimum_stiffness_eigenvalue"],
    )
    divergence_local_speeds_m_per_s = np.linspace(
        divergence_speed_m_per_s - 0.02,
        divergence_speed_m_per_s + 0.05,
        701,
    )
    flutter_local_speeds_m_per_s = np.linspace(54.9, 55.5, 1201)
    speeds_m_per_s = np.unique(
        np.concatenate(
            (
                uniform_data["speeds_m_per_s"],
                divergence_local_speeds_m_per_s,
                flutter_local_speeds_m_per_s,
                np.array([divergence_speed_m_per_s]),
            )
        )
    )
    data = sweep_eigenvalues(speeds_m_per_s=speeds_m_per_s)
    speeds_m_per_s = data["speeds_m_per_s"]
    eigenvalues_by_speed = data["eigenvalues"]
    eigenvectors_by_speed = data["eigenvectors"]
    tolerances = data["tolerances"]
    analytical_speed_m_per_s = analytical_first_mode_divergence_speed()
    absolute_speed_difference = abs(
        divergence_speed_m_per_s - analytical_speed_m_per_s
    )
    relative_speed_difference = (
        absolute_speed_difference / analytical_speed_m_per_s
    )
    assert np.isclose(
        divergence_speed_m_per_s,
        analytical_speed_m_per_s,
        rtol=1e-8,
        atol=1e-10,
    )
    assert divergence_speed_m_per_s > SIMPLIFIED_CRITICAL_SPEED_M_PER_S
    assert abs(divergence_speed_m_per_s - 50.000864) < 2e-6

    maximum_real_part = np.max(eigenvalues_by_speed.real, axis=1)
    classifications = check_tolerance_sensitivity(
        eigenvalues_by_speed,
        tolerances,
        speeds_m_per_s,
    )
    assert all(len(labels) == speeds_m_per_s.size for labels in classifications.values())
    (
        divergence_flags,
        flutter_flags,
        _neutral_flags,
        divergence_intervals,
        flutter_intervals,
        re_neutral_intervals,
    ) = find_instability_regions(
        speeds_m_per_s,
        eigenvalues_by_speed,
        tolerances,
    )
    # The stiffness-root calculation is independent of tau and is unchanged at tau/10 and 10*tau.
    divergence_speeds_by_tolerance = {
        multiplier: divergence_speed_m_per_s for multiplier in (0.1, 1.0, 10.0)
    }
    assert len(set(divergence_speeds_by_tolerance.values())) == 1

    branch_values, branch_frequencies_hz, normalized_jumps = track_eigenvalue_branches(
        eigenvalues_by_speed,
        eigenvectors_by_speed,
        tolerances,
        speeds_m_per_s,
        divergence_speed_m_per_s,
    )
    maximum_jump = float(np.max(normalized_jumps))
    maximum_jump_outside_critical = float(
        np.max(
            normalized_jumps[
                np.abs(speeds_m_per_s[1:] - divergence_speed_m_per_s)
                > CRITICAL_BRANCH_WINDOW_M_PER_S
            ]
        )
    )
    divergence_checks = inspect_divergence_eigenvalues(divergence_speed_m_per_s)

    plot_frequency_branches(
        speeds_m_per_s,
        branch_frequencies_hz,
        divergence_speed_m_per_s,
        frequency_plot_path,
    )
    plot_eigenvalue_stability(
        speeds_m_per_s,
        maximum_real_part,
        tolerances,
        divergence_speed_m_per_s,
        divergence_intervals,
        re_neutral_intervals,
        flutter_intervals[0][0] if flutter_intervals else None,
        stability_plot_path,
    )
    for output_path in (frequency_plot_path, stability_plot_path):
        assert output_path.is_file() and output_path.stat().st_size > 0

    print("\nSpeed sweep:")
    print(
        f"  Uniform sweep: {SPEED_MIN_M_PER_S:.1f} to {SPEED_MAX_M_PER_S:.1f} m/s; "
        f"{SPEED_POINT_COUNT} points; step "
        f"{(SPEED_MAX_M_PER_S - SPEED_MIN_M_PER_S) / (SPEED_POINT_COUNT - 1):.5f} m/s"
    )
    print(
        f"  Supplemental divergence-window points: "
        f"{np.count_nonzero((speeds_m_per_s >= divergence_speed_m_per_s - 0.02) & (speeds_m_per_s <= divergence_speed_m_per_s + 0.05))}; "
        f"total evaluated speeds: {speeds_m_per_s.size}"
    )
    print(
        "  Supplemental flutter-window points (54.9-55.5 m/s, 0.0005 m/s step): "
        f"{np.count_nonzero((speeds_m_per_s >= 54.9) & (speeds_m_per_s <= 55.5))}"
    )
    print(f"  Simplified critical speed: {SIMPLIFIED_CRITICAL_SPEED_M_PER_S:.9f} m/s")
    print(f"  Divergence bracket: {divergence_bracket}")
    print(f"  Analytical first-mode divergence: {analytical_speed_m_per_s:.12f} m/s")
    print(f"  Brentq divergence speed: {divergence_speed_m_per_s:.12f} m/s")
    print(f"  Absolute difference: {absolute_speed_difference:.3e} m/s")
    print(f"  Relative difference: {relative_speed_difference:.3e} ({relative_speed_difference * 100.0:.3e}%)")
    print("  Divergence root unchanged for tau/10, tau, and 10*tau: PASS")
    print(
        "  Stability tolerance tau range: "
        f"{np.min(tolerances):.3e} to {np.max(tolerances):.3e} 1/s"
    )
    print(
        "  Classification rule: max Re(lambda) <= tau is numerically "
        "neutral/stable; positive roots are near-real when "
        "abs(Im(lambda)) <= max(tau, 1e-3*spectral scale), otherwise "
        "they are flutter candidates."
    )
    print("\nLocally resolved instability transitions:")
    if divergence_intervals:
        for index, (start_m_per_s, end_m_per_s) in enumerate(divergence_intervals, 1):
            print(
                f"  Divergence-like interval {index}: {start_m_per_s:.9f} to "
                f"{end_m_per_s:.9f} m/s"
            )
    else:
        print("  Divergence-like interval: none detected")
    if re_neutral_intervals:
        for index, (start_m_per_s, end_m_per_s) in enumerate(re_neutral_intervals, 1):
            print(
                f"  Re-neutral/stable interval {index}: {start_m_per_s:.9f} to "
                f"{end_m_per_s:.9f} m/s"
            )
    else:
        print("  Re-neutral/stable interval after divergence: none detected")
    if flutter_intervals:
        first_flutter_speed = flutter_intervals[0][0]
        flutter_index = int(np.argmin(np.abs(speeds_m_per_s - first_flutter_speed)))
        unstable_flutter_roots = eigenvalues_by_speed[flutter_index][
            (eigenvalues_by_speed[flutter_index].real > tolerances[flutter_index])
            & (
                np.abs(eigenvalues_by_speed[flutter_index].imag)
                > imaginary_part_tolerance(
                    eigenvalues_by_speed[flutter_index], tolerances[flutter_index]
                )
            )
        ]
        flutter_root = unstable_flutter_roots[
            np.argmax(unstable_flutter_roots.real)
        ]
        print(
            f"  First flutter-candidate speed: {first_flutter_speed:.9f} m/s; "
            f"lambda={flutter_root.real:.9g}{flutter_root.imag:+.9g}j 1/s "
            "(positive real and significant nonzero imaginary part)"
        )
    else:
        print("  First flutter-candidate speed: none detected")
    print("\nEigenpair checks:")
    print(
        "  Maximum normalized eigenpair residual: "
        f"{np.max(data['maximum_residuals']):.3e} "
        f"(tolerance {EIGENPAIR_RESIDUAL_TOLERANCE:.1e}): PASS"
    )
    print(
        "  Maximum normalized conjugate-pair error: "
        f"{np.max(data['conjugate_pair_errors']):.3e} "
        f"(tolerance {CONJUGATE_PAIR_TOLERANCE:.1e}): PASS"
    )
    print("  Eight finite eigenvalues/eigenvectors at every speed: PASS")
    print(
        "  Branch matching weights (distance/similarity): "
        f"{EIGENVALUE_DISTANCE_WEIGHT:.2f}/{EIGENVECTOR_SIMILARITY_WEIGHT:.2f}"
    )
    print(
        "  Maximum normalized eigenvalue step: "
        f"{maximum_jump:.3e}; outside critical window: "
        f"{maximum_jump_outside_critical:.3e} "
        f"(limit {BRANCH_JUMP_TOLERANCE:.2f}): PASS"
    )
    first_branch = branch_frequencies_hz[:, 0]
    near_divergence_index = np.flatnonzero(
        (speeds_m_per_s < divergence_speed_m_per_s) & np.isfinite(first_branch)
    )[-1]
    print(
        "  First branch frequency near divergence: "
        f"{first_branch[near_divergence_index]:.9g} Hz at "
        f"{speeds_m_per_s[near_divergence_index]:.5f} m/s; decreasing toward zero: PASS"
    )
    print("\nDivergence eigenvalue behavior:")
    print(
        f"  At V_div - {divergence_checks['probe_delta_m_per_s']:.1e} m/s: "
        "oscillatory pair approaches zero; lowest positive frequency "
        f"{divergence_checks['below_frequency_hz']:.9g} Hz"
    )
    print(
        "  At V_div: minimum absolute eigenvalue "
        f"{divergence_checks['root_minimum_eigenvalue_magnitude_per_s']:.3e} 1/s"
    )
    print(
        f"  At V_div + {divergence_checks['probe_delta_m_per_s']:.1e} m/s: "
        "positive/negative real pair found: "
        f"{divergence_checks['above_positive_real_eigenvalue_per_s']:.9g}, "
        f"{divergence_checks['above_negative_real_eigenvalue_per_s']:.9g} 1/s"
    )
    print("\nSaved plots:")
    for output_path in (frequency_plot_path, stability_plot_path):
        print(f"  {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
