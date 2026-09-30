# Axially Moving Web: Vibration and Stability

This repository studies the transverse response of a tensioned, simply supported flexible web moving at constant axial speed. An Euler-Bernoulli model is reduced with a sine-basis Galerkin approximation. The scripts calculate free response, a simplified critical-speed reference, and eigenvalue-based stability indicators.

This is a small research and portfolio model, not a validated design tool. Its four-mode flutter and re-neutralization observations are candidates that require modal convergence and experimental or higher-fidelity validation.

## Model and limits

- Linear, small-displacement Euler-Bernoulli behavior with constant tension and transport speed.
- Simply supported ends and four sine modes in the reported stability sweep.
- No physical or viscous damping and no applied forcing; the time response is free vibration.
- Linear mass density is specified directly as `0.08 kg/m`; no material density is inferred.
- The simplified reference speed is $\sqrt{T_0/(\rho A)}$. The finite-bending first-mode divergence is evaluated separately.
- A finite-duration bounded response is not proof of overall stability. No mesh, mode-convergence, uncertainty, or experimental validation is included.

## Quick start

Python 3.12 or 3.14 is covered by CI. From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

Generate all reported outputs from the repository root:

```powershell
\.venv\Scripts\python.exe run_analysis.py
```

The runner stops on the first failed analysis. Individual scripts can also be invoked directly from the repository root.

The numerical regression suite checks matrix structure and input validation, state-space construction, finite free-response output, eigenvalue classification, and agreement between numerical and analytical first-mode divergence.

## Outputs

- [Simplified critical speed](results/critical_speed.png)
- [Free-vibration response](results/time_response.png)
- [Tracked modal frequencies](results/frequency_vs_speed.png)
- [Eigenvalue-based stability indicators](results/eigenvalue_stability.png)

The detailed model derivation, nominal parameters, and reported results are documented in [Turkish](README.md).

## Project layout

```text
src/       Model, simulation, critical-speed, and stability scripts
tests/     Numerical regression tests
results/   Generated plots
```