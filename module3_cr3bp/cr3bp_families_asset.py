"""
CR3BP L1 + L2 Lyapunov Families via ASSET
Module 3/4 | Declan O'Connor | ASRL

Five L1, five L2 Lyapunov orbits plus five DROs around the Moon, Earth-Moon system.
Each initial condition is pre-corrected; ASSET propagates one period each.

Run in WSL:
  source ~/asrl_env/bin/activate
  python cr3bp_families_asset.py

Outputs: cr3bp_families.csv (columns: family, orbit, t, x, y, z, vx, vy, vz)
         family 0 = L1, family 1 = L2, family 2 = DRO (around Moon)
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

L1_FAMILY = [
    (0.834, 0.0249758646, 2.693446),
    (0.823, 0.1304364210, 2.746632),
    (0.812, 0.2511052120, 2.933921),
    (0.802, 0.3434538745, 3.252248),
    (0.792, 0.4020731858, 3.605290),
]
L2_FAMILY = [
    (1.172, -0.0953155958, 3.387480),
    (1.180, -0.1493587714, 3.411570),
    (1.190, -0.2285404169, 3.482100),
    (1.200, -0.3193244021, 3.669320),
    (1.208, -0.3768158823, 3.920960),
]
DRO_FAMILY = [
    (1.02785, -0.5929669218, 0.427010),
    (1.04785, -0.5148229550, 0.749760),
    (1.06785, -0.4789232134, 1.102930),
    (1.08785, -0.4634413074, 1.474450),
    (1.10785, -0.4598901804, 1.856480),
]

rows = []
for fam, fam_ics in [(0, L1_FAMILY), (1, L2_FAMILY), (2, DRO_FAMILY)]:
    for k, (x0, vy0, T) in enumerate(fam_ics):
        X0 = np.array([x0, 0, 0, 0, vy0, 0, 0.0]).reshape(7, 1)
        traj = integ.integrate_dense(X0, T)
        arr = np.array([p.flatten() for p in traj])
        for r in arr:
            rows.append([fam, k, r[6], r[0], r[1], r[2], r[3], r[4], r[5]])
    print(f"family {fam}: {len(fam_ics)} orbits done")

rows = np.array(rows)
np.savetxt("cr3bp_families.csv", rows, delimiter=",",
           header="family,orbit,t,x,y,z,vx,vy,vz", comments="")
print(f"Saved cr3bp_families.csv ({len(rows)} rows)")
