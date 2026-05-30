# asrl-work

Astrodynamics research trial for the Astrodynamics and Space Research Laboratory (ASRL).
Advisor: Dr. Rohan Sood, University of Alabama.

This repository contains the trial modules building toward a generalized trajectory
optimization framework for multi-body gravity assist mission design. Each module pairs
a hand-built Python or MATLAB implementation against the ASSET (Astrodynamics Software
and Science Enabling Toolkit) framework to benchmark accuracy and conservation.

## Modules

### Module 2: Two-Body Integration (`module2_twobody/`)
Two-body equations of motion integrated three ways and compared on identical initial
conditions.

- `two_body.py`: Euler and RK4 integrators built from scratch.
- `asset_twobody_earth.py`: same problem run in ASSET, nondimensionalized, exported to CSV.
- `two_body_module2.m`: MATLAB script that overlays Euler, RK4, and ASSET, and plots the
  RK4 vs ASSET position delta in kilometers.

Initial conditions: r0 = [8000, 0, 6000] km, v0 = [0, 7, 0] km/s.
RK4 and ASSET converge to the same min/max altitude (3621.88 km / ~9572 km), confirming
both are at the resolution limit for the chosen physical constants. The delta plot shows
the residual difference in physical units, since a gap that looks flat on an orbit plot
can still be hundreds of kilometers.

### Module 3: CR3BP and Lagrange Points (`module3_cr3bp/`)
Circular restricted three-body problem for the Earth-Moon system.

- `cr3bp.py`: CR3BP dynamics, all five Lagrange points, Jacobi constant conservation.
- `cr3bp_asset.py`: same dynamics in ASSET, exported for MATLAB comparison.

Mass parameter mu = 0.012150585609624 (Pavlak thesis). ASSET produced two orders of
magnitude tighter Jacobi constant conservation than the hand-built integrator.

### Module 4: Lyapunov Orbit Families (`module4_lyapunov/`)
Periodic Lyapunov orbit families about L1, L2, and L3.

- Single shooting differential corrector with state transition matrix propagation.
- Convergence tolerance 1e-12.
- Known working L1 initial condition: x0 = 0.823.

### Module 5: Invariant Manifolds (`module5_manifolds/`)
Stable and unstable invariant manifolds computed from the Lyapunov families.

### Module 6: Trajectory Optimization (next)
Trajectory optimization problem implemented in ASSET's native framework.

## Environment

- WSL2 Ubuntu
- Python virtual environment at `~/asrl_env` with numpy, scipy, matplotlib, asset_asrl
- Working directory `~/asrl_work`

## Workflow

Python scripts run inside the virtual environment:

```bash
source ~/asrl_env/bin/activate
cd ~/asrl_work/module2_twobody
python asset_twobody_earth.py
```

ASSET scripts export trajectory data to CSV. The CSV is copied to Windows and read by the
matching MATLAB script for comparison plotting:

```bash
cp ~/asrl_work/module2_twobody/asset_orbit.csv /mnt/c/Users/oconn/Downloads/
```

## Research motivation

Designing a multi-body gravity assist trajectory from Earth to Mars via Venus using
heterogeneous CR3BP phase chaining.
