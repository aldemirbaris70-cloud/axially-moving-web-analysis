import numpy as np
import pytest

from stability_analysis import (
    analytical_first_mode_divergence_speed,
    classify_eigenvalues,
    estimate_divergence_speed,
    sweep_eigenvalues,
)


@pytest.mark.parametrize("number_of_modes", [1, 2, 4, 8])
def test_sweep_supports_variable_mode_count(number_of_modes: int) -> None:
    result = sweep_eigenvalues(number_of_modes, np.array([0.0, 1.0]))

    state_dimension = 2 * number_of_modes
    assert result["eigenvalues"].shape == (2, state_dimension)
    assert result["eigenvectors"].shape == (2, state_dimension, state_dimension)
    assert result["maximum_residuals"].max() < 1e-8


@pytest.mark.parametrize("speeds", [[0.0], [1.0, 0.0], [0.0, np.nan]])
def test_sweep_rejects_invalid_speed_grid(speeds: list[float]) -> None:
    with pytest.raises(ValueError):
        sweep_eigenvalues(2, np.asarray(speeds))


def test_sweep_rejects_invalid_mode_count() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        sweep_eigenvalues(0, np.array([0.0, 1.0]))
    with pytest.raises(TypeError, match="must be an integer"):
        sweep_eigenvalues(2.5, np.array([0.0, 1.0]))


def test_divergence_root_matches_analytical_first_mode() -> None:
    result = sweep_eigenvalues(2, np.array([0.0, 52.0]))
    numerical_speed, _ = estimate_divergence_speed(
        result["speeds_m_per_s"],
        result["minimum_stiffness_eigenvalue"],
        number_of_modes=2,
    )

    assert numerical_speed == pytest.approx(
        analytical_first_mode_divergence_speed(), abs=1e-9
    )


@pytest.mark.parametrize(
    ("eigenvalues", "expected"),
    [
        (np.array([-1.0 + 2.0j, -1.0 - 2.0j]), "neutral/stable"),
        (np.array([0.1 + 0.0j, -0.1 + 0.0j]), "divergence candidate"),
        (np.array([0.1 + 5.0j, 0.1 - 5.0j]), "flutter candidate"),
    ],
)
def test_classifies_eigenvalue_behavior(
    eigenvalues: np.ndarray, expected: str
) -> None:
    assert classify_eigenvalues(eigenvalues, tolerance=1e-8) == expected