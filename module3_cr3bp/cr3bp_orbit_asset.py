"""
CR3BP L1 Lyapunov Orbit via ASSET
Module 3 | Declan O'Connor | ASRL

Earth-Moon system, original L1 Lyapunov initial condition.
Y0 = [0.9870, 0, 0, 0, -1.0042, 0], tf = 2.746632

Run in WSL:
  source ~/asrl_env/bin/activate
  python cr3bp_orbit_asset.py

Outputs: cr3bp_orbit.csv (columns: t, x, y, z, vx, vy, vz)  nondimensional
"""

import numpy as np
import asset_asrl as ast
import asset_asrl.OptimalControl as oc
import asset_asrl.VectorFunctions as vf

mu = 0.01215     # Earth-Moon mass parameter (matches original module 3)

class CR3BP(oc.ODEBase):
    def __init__(self, mu):
        args = oc.ODEArguments(6, 0)
        xv = args.XVec()
        x, y, z    = xv[0], xv[1], xv[2]
        vx, vy, vz = xv[3], xv[4], xv[5]
        xE = -mu
        xM = 1.0 - mu
        r1c = ((x - xE)*(x - xE) + y*y + z*z) ** 1.5
        r2c = ((x - xM)*(x - xM) + y*y + z*z) ** 1.5
        ax =  2*vy + x - (1-mu)*(x - xE)/r1c - mu*(x - xM)/r2c
        ay = -2*vx + y - (1-mu)*y/r1c        - mu*y/r2c
        az =            -(1-mu)*z/r1c        - mu*z/r2c
        ode = vf.stack([vx, vy, vz, ax, ay, az])
        super().__init__(ode, 6, 0)

ode = CR3BP(mu)

# Original L1 Lyapunov initial condition
r0 = np.array([0.823, 0.0, 0.0])
v0 = np.array([0.0, 0.1304364210, 0.0])
X0 = np.concatenate([r0, v0, [0.0]]).reshape(7, 1)
tf = 2.746632

integ = ode.integrator(1e-12)
integ.setStepSizes(0.001, 1e-9, 0.01)
traj = integ.integrate_dense(X0, tf)

traj_arr = np.array([p.flatten() for p in traj])
t  = traj_arr[:,6]
x  = traj_arr[:,0]; y  = traj_arr[:,1]; z  = traj_arr[:,2]
vx = traj_arr[:,3]; vy = traj_arr[:,4]; vz = traj_arr[:,5]

np.savetxt("cr3bp_orbit.csv", np.column_stack([t, x, y, z, vx, vy, vz]),
           delimiter=",", header="t,x,y,z,vx,vy,vz", comments="")

print(f"Saved {len(traj)} points")

xE, xM = -mu, 1-mu
lstar = 384400.0
r1 = np.sqrt((x-xE)**2 + y**2 + z**2)
r2 = np.sqrt((x-xM)**2 + y**2 + z**2)
print(f"Min Earth altitude: {r1.min()*lstar - 6378.0:.0f} km")
print(f"Min Moon altitude:  {r2.min()*lstar - 1737.4:.0f} km")
