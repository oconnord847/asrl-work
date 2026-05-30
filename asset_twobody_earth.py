import numpy as np
import asset_asrl.OptimalControl as oc
import asset_asrl.VectorFunctions as vf

G   = 6.67430e-20
m1  = 5.97219e24
mu  = G * m1
R_E = 6378.12

lstar = 10000.0
tstar = np.sqrt(lstar**3 / mu)
vstar = lstar / tstar
mu_nd = 1.0

class TwoBodyODE(oc.ODEBase):
    def __init__(self, mu):
        args = oc.ODEArguments(6, 0)
        xv = args.XVec()
        x, y, z   = xv[0], xv[1], xv[2]
        xd, yd, zd = xv[3], xv[4], xv[5]
        r3 = (x*x + y*y + z*z) ** 1.5
        ode = vf.stack([xd, yd, zd, -mu*x/r3, -mu*y/r3, -mu*z/r3])
        super().__init__(ode, 6, 0)

r0 = np.array([8000.0, 0.0, 6000.0]) / lstar
v0 = np.array([0.0, 7.0, 0.0])       / vstar
X0 = np.concatenate([r0, v0, [0.0]]).reshape(7, 1)
tf_nd = 14709.0 / tstar

ode   = TwoBodyODE(mu_nd)
integ = ode.integrator(1e-12)
integ.setStepSizes(0.001, 1e-7, 0.01)   # default, min, max

traj = integ.integrate_dense(X0, tf_nd)

traj_arr = np.array([p.flatten() for p in traj])
t = traj_arr[:,6] * tstar
x = traj_arr[:,0] * lstar
y = traj_arr[:,1] * lstar
z = traj_arr[:,2] * lstar

np.savetxt("asset_orbit.csv", np.column_stack([t, x, y, z]),
           delimiter=",", header="t_s,x_km,y_km,z_km", comments="")

alt = np.sqrt(x**2 + y**2 + z**2) - R_E
print(f"Saved {len(traj)} points")
print(f"Min alt: {np.min(alt):.4f} km  (textbook 3621.88)")
print(f"Max alt: {np.max(alt):.4f} km  (textbook 9431.85)")
