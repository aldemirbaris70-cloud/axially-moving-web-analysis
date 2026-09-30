import numpy as np

from galerkin_model import assemble_galerkin_matrices
from simulation import build_state_matrix, simulate_free_vibration


def test_state_matrix_has_first_order_block_form() -> None:
    matrices = assemble_galerkin_matrices(10.0, number_of_modes=2)
    state_matrix = build_state_matrix(*matrices)

    assert state_matrix.shape == (4, 4)
    assert np.allclose(state_matrix[:2, :2], 0.0)
    assert np.allclose(state_matrix[:2, 2:], np.eye(2))


def test_free_response_is_finite_and_starts_at_requested_displacement() -> None:
    solution, state_matrix, initial_state, midpoint_displacement = (
        simulate_free_vibration(10.0, number_of_modes=2)
    )

    assert solution.success
    assert state_matrix.shape == (4, 4)
    assert initial_state.shape == (4,)
    assert np.isfinite(solution.y).all()
    assert midpoint_displacement[0] == 0.001