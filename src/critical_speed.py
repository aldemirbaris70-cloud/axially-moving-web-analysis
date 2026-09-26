"""Calculate and plot the simplified critical speed versus web tension."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from parameters import initial_tension_n, linear_density_kg_per_m


def critical_speed_m_per_s(tension_n: float) -> float:
    """Return the simplified critical speed for a given tension [N]."""
    return float(np.sqrt(tension_n / linear_density_kg_per_m))


def main() -> None:
    nominal_speed_m_per_s = critical_speed_m_per_s(initial_tension_n)
    print(
        f"Nominal critical speed at T0 = {initial_tension_n:.1f} N: "
        f"{nominal_speed_m_per_s:.2f} m/s"
    )

    tension_values_n = np.linspace(50.0, 500.0, 451)
    critical_speed_values_m_per_s = np.sqrt(
        tension_values_n / linear_density_kg_per_m
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(tension_values_n, critical_speed_values_m_per_s, label="Critical speed")
    axis.scatter(
        initial_tension_n,
        nominal_speed_m_per_s,
        color="crimson",
        marker="o",
        s=70,
        zorder=3,
        label=f"Nominal T0 = {initial_tension_n:.0f} N",
    )
    axis.set_xlabel("Tension T [N]")
    axis.set_ylabel("Critical speed V_cr [m/s]")
    axis.set_title("Simplified Critical Speed vs. Web Tension")
    axis.grid(True, linestyle="--", alpha=0.6)
    axis.legend()
    figure.tight_layout()

    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "results" / "critical_speed.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"Critical speed plot saved to: {output_path}")


if __name__ == "__main__":
    main()
