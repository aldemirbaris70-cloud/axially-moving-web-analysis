import numpy as np
import pytest

from galerkin_model import assemble_galerkin_matrices
from parameters import linear_density_kg_per_m, span_length_m


@pytest.mark.parametrize("speed_m_per_s", [0.0, 10.0, 45.0])
def test_galerkin_matrix_structure(speed_m_per_s: float) -> None:
    mass, coriolis, stiffness = assemble_galerkin_matrices(speed_m_per_s, 4)

    assert mass.shape == (4, 4)
    assert np.allclose(mass, mass.T)
    assert np.allclose(coriolis, -coriolis.T)
    assert np.allclose(stiffness, stiffness.T)
    assert np.all(np.linalg.eigvalsh(mass) > 0.0)
    assert np.all(np.linalg.eigvalsh(stiffness) > 0.0)
    assert np.allclose(
        mass,
        np.eye(4) * linear_density_kg_per_m * span_length_m / 2.0,
    )


@pytest.mark.parametrize("number_of_modes", [0, -1])
def test_galerkin_rejects_nonpositive_mode_count(number_of_modes: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        assemble_galerkin_matrices(0.0, number_of_modes)


def test_galerkin_rejects_noninteger_mode_count() -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        assemble_galerkin_matrices(0.0, 1.5)


def test_galerkin_rejects_nonfinite_speed() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        assemble_galerkin_matrices(float("nan"))