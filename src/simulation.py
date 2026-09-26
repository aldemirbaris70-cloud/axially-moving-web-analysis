"""Time-domain free-vibration simulation for an axially moving web."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from galerkin_model import assemble_galerkin_matrices
from parameters import span_length_m

NUMBER_OF_MODES = 4
TRANSPORT_SPEEDS_M_PER_S = (10.0, 45.0)
T_START_SECONDS = 0.0
T_END_SECONDS = 2.0
OUTPUT_POINTS = 4001
RELATIVE_TOLERANCE = 1e-8
ABSOLUTE_TOLERANCE = 1e-10


def build_state_matrix(
    mass_matrix: np.ndarray,
    coriolis_matrix: np.ndarray,
    stiffness_matrix: np.ndarray,
) -> np.ndarray:
    """Build the first-order state matrix using linear solves, not an inverse."""
    number_of_modes = mass_matrix.shape[0]
    identity = np.eye(number_of_modes)
    zero_block = np.zeros((number_of_modes, number_of_modes))
    return np.block(
        [
            [zero_block, identity],
            [
                -np.linalg.solve(mass_matrix, stiffness_matrix),
                -np.linalg.solve(mass_matrix, coriolis_matrix),
            ],
        ]
    )


def simulate_free_vibration(
    transport_speed_m_per_s: float,
    number_of_modes: int = NUMBER_OF_MODES,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    """Run the undamped free response and return solution and midpoint data."""
    mass_matrix, coriolis_matrix, stiffness_matrix = assemble_galerkin_matrices(
        transport_speed_m_per_s=transport_speed_m_per_s,
        number_of_modes=number_of_modes,
    )
    state_matrix = build_state_matrix(
        mass_matrix,
        coriolis_matrix,
        stiffness_matrix,
    )

    initial_displacements_m = np.zeros(number_of_modes)
    # The specified initial shape is exactly the first sine mode, so only q_1(0) is nonzero.
    initial_displacements_m[0] = 0.001
    initial_modal_velocities_m_per_s = np.zeros(number_of_modes)
    initial_state = np.concatenate(
        (initial_displacements_m, initial_modal_velocities_m_per_s)
    )

    expected_state_shape = (2 * number_of_modes, 2 * number_of_modes)
    assert state_matrix.shape == expected_state_shape
    assert initial_state.shape == (2 * number_of_modes,)

    time_values_s = np.linspace(T_START_SECONDS, T_END_SECONDS, OUTPUT_POINTS)
    solution = solve_ivp(
        fun=lambda _time_s, state: state_matrix @ state,
        t_span=(T_START_SECONDS, T_END_SECONDS),
        y0=initial_state,
        method="RK45",
        t_eval=time_values_s,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )
    assert solution.success, (
        f"solve_ivp failed at V={transport_speed_m_per_s:g} m/s: "
        f"{solution.message}"
    )
    assert np.isfinite(solution.t).all()
    assert np.isfinite(solution.y).all()
    assert np.array_equal(solution.t, time_values_s)

    mode_numbers = np.arange(1, number_of_modes + 1)
    midpoint_shape_values = np.sin(mode_numbers * np.pi / 2.0)
    midpoint_displacement_m = midpoint_shape_values @ solution.y[:number_of_modes]
    initial_midpoint_displacement_m = midpoint_displacement_m[0]
    assert np.isclose(initial_midpoint_displacement_m, 0.001, rtol=0.0, atol=1e-12)

    if number_of_modes == 4:
        assert np.allclose(
            midpoint_shape_values,
            np.array([1.0, 0.0, -1.0, 0.0]),
            rtol=0.0,
            atol=1e-15,
        )
        assert np.allclose(
            midpoint_displacement_m,
            solution.y[0] - solution.y[2],
            rtol=1e-12,
            atol=1e-15,
        )

    return solution, state_matrix, initial_state, midpoint_displacement_m


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "results" / "time_response.png"
    assert output_path.parent.is_dir(), "The results directory does not exist"

    time_series_by_speed: dict[float, np.ndarray] = {}
    displacement_by_speed_m: dict[float, np.ndarray] = {}

    for speed_m_per_s in TRANSPORT_SPEEDS_M_PER_S:
        solution, state_matrix, initial_state, displacement_m = (
            simulate_free_vibration(speed_m_per_s)
        )
        assert state_matrix.shape == (8, 8)
        assert initial_state.shape == (8,)

        time_series_by_speed[speed_m_per_s] = solution.t
        displacement_by_speed_m[speed_m_per_s] = displacement_m
        minimum_m = float(np.min(displacement_m))
        maximum_m = float(np.max(displacement_m))
        initial_m = float(displacement_m[0])
        print(f"V = {speed_m_per_s:.1f} m/s: solve_ivp success = {solution.success}")
        print(f"  State matrix shape: {state_matrix.shape}")
        print(f"  Initial state shape: {initial_state.shape}")
        print(
            "  Initial midpoint displacement: "
            f"{initial_m:.9g} m ({initial_m * 1000.0:.9g} mm)"
        )
        print(
            "  Midpoint displacement range: "
            f"{minimum_m:.9g} to {maximum_m:.9g} m; "
            f"{minimum_m * 1000.0:.9g} to {maximum_m * 1000.0:.9g} mm"
        )

    reference_times_s = time_series_by_speed[TRANSPORT_SPEEDS_M_PER_S[0]]
    assert np.array_equal(
        reference_times_s, time_series_by_speed[TRANSPORT_SPEEDS_M_PER_S[1]]
    )

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for speed_m_per_s in TRANSPORT_SPEEDS_M_PER_S:
        displacement_mm = displacement_by_speed_m[speed_m_per_s] * 1000.0
        axis.plot(
            time_series_by_speed[speed_m_per_s],
            displacement_mm,
            label=f"V = {speed_m_per_s:.0f} m/s",
        )
    axis.set_xlabel("Time t [s]")
    axis.set_ylabel("Midpoint displacement w(L/2,t) [mm]")
    axis.set_title("Free Vibration Response of an Axially Moving Web")
    axis.grid(True, linestyle="--", alpha=0.6)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    assert output_path.is_file() and output_path.stat().st_size > 0
    print(f"Time-response plot saved to: {output_path}")


if __name__ == "__main__":
    main()
