"""
CR3BP Trajectory in ASSET
Uses asset_asrl.Astro.AstroModels.CR3BP - the built-in class
Same initial conditions as the Python module so trajectories match

Run:
  source ~/asrl_env/bin/activate
  python cr3bp_asset.py
"""

import numpy as np
import asset_asrl as ast
from asset_asrl.Astro.AstroModels import CR3BP
import asset_asrl.Astro.Constants as c

# CR3BP model
mu1   = c.MuEarth          # m^3/s^2
mu2   = c.MuMoon
lstar = 385000 * 1000.0    # m

ode = CR3BP(mu1, mu2, lstar)

# Same initial conditions as Python module
X0 = np.array([1.2, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]).reshape(7, 1)
tf = 12.0   # nondimensional

# Integrate
integ = ode.integrator(1e-12)
integ.setStepSizes(0.001, 1e-7, 0.01)
traj = integ.integrate_dense(X0, tf)

# Save CSV
traj_arr = np.array([p.flatten() for p in traj])
np.savetxt("cr3bp_asset.csv",
           traj_arr[:, [6, 0, 1, 2]],
           delimiter=",",
           header="t,x,y,z",
           comments="")

print(f"Saved {len(traj)} points to cr3bp_asset.csv")
print(f"Time range: 0 to {traj_arr[-1,6]:.4f} nondim")
print(f"X range: [{traj_arr[:,0].min():.4f}, {traj_arr[:,0].max():.4f}]")
print(f"Y range: [{traj_arr[:,1].min():.4f}, {traj_arr[:,1].max():.4f}]")
