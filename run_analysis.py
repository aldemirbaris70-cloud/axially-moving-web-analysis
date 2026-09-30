"""Run every analysis script from the repository root."""

import subprocess
import sys
from pathlib import Path

ANALYSIS_SCRIPTS = (
    "critical_speed.py",
    "galerkin_model.py",
    "simulation.py",
    "stability_analysis.py",
)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    for script_name in ANALYSIS_SCRIPTS:
        script_path = project_root / "src" / script_name
        print(f"\n=== {script_name} ===", flush=True)
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=project_root,
            check=True,
        )


if __name__ == "__main__":
    main()