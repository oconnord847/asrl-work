"""
CR3BP L1 Lyapunov Family via ASSET
Module 3/4 | Declan O'Connor | ASRL

Five verified periodic L1 Lyapunov orbits of increasing amplitude,
Earth-Moon system. Each initial condition is pre-corrected (perpendicular
y=0 crossing); ASSET propagates each for exactly one period.

Run in WSL:
  source ~/asrl_env/bin/activate
  python cr3bp_family_asset.py

Outputs: cr3bp_family.csv (columns: orbit, t, x, y, z, vx, vy, vz)
"""

import numpy as np
import asset_asrl as ast
import asset_asrl.OptimalControl as oc
import asset_asrl.VectorFunctions as vf

mu = 0.01215

class CR3BP(oc.ODEBase):
    def __init__(self, mu):
        args = oc.ODEArguments(6, 0)
        xv = args.XVec()
        x, y, z    = xv[0], xv[1], xv[2]
        vx, vy, vz = xv[3], xv[4], xv[5]
        xE = -mu; xM = 1.0 - mu
        r1c = ((x - xE)*(x - xE) + y*y + z*z) ** 1.5
        r2c = ((x - xM)*(x - xM) + y*y + z*z) ** 1.5
        ax =  2*vy + x - (1-mu)*(x - xE)/r1c - mu*(x - xM)/r2c
        ay = -2*vx + y - (1-mu)*y/r1c        - mu*y/r2c
        az =            -(1-mu)*z/r1c        - mu*z/r2c
        super().__init__(vf.stack([vx, vy, vz, ax, ay, az]), 6, 0)

ode = CR3BP(mu)
integ = ode.integrator(1e-12)
integ.setStepSizes(0.001, 1e-9, 0.01)

# Verified periodic initial conditions: (x0, vy0, period)
ICS = [
    (0.834, 0.0249758646, 2.693446),
    (0.823, 0.1304364210, 2.746632),
    (0.812, 0.2511052120, 2.933921),
    (0.802, 0.3434538745, 3.252248),
    (0.792, 0.4020731858, 3.605290),
]

rows = []
for k, (x0, vy0, T) in enumerate(ICS):
    X0 = np.array([x0, 0, 0, 0, vy0, 0, 0.0]).reshape(7, 1)
    traj = integ.integrate_dense(X0, T)
    arr = np.array([p.flatten() for p in traj])
    for r in arr:
        rows.append([k, r[6], r[0], r[1], r[2], r[3], r[4], r[5]])
    print(f"orbit {k}: x0={x0}  {len(traj)} pts")

rows = np.array(rows)
np.savetxt("cr3bp_family.csv", rows, delimiter=",",
           header="orbit,t,x,y,z,vx,vy,vz", comments="")
print(f"Saved cr3bp_family.csv ({len(rows)} rows, {len(ICS)} orbits)")
