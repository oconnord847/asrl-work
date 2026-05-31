"""
CR3BP Invariant Manifold of an L1 Lyapunov Orbit, globalized around Earth
Module 5 | Declan O'Connor | ASRL

ASSET propagates every manifold trajectory. The unstable direction comes from
the monodromy matrix (one-period STM); that small linear-algebra step uses the
variational equations, then ASSET integrates each manifold leg forward in time
until it wraps around the Earth.

Run in WSL:
  source ~/asrl_env/bin/activate
  python manifolds_L1.py

Outputs: manifolds_L1.csv (columns: branch, leg, x, y)  branch 0=orbit, 1=unstable, 2=stable
"""

import numpy as np
from scipy.integrate import solve_ivp
import asset_asrl as ast
import asset_asrl.OptimalControl as oc
import asset_asrl.VectorFunctions as vf

mu = 0.01215; xE, xM = -mu, 1-mu; lstar = 384400.0; Re = 6378.0/lstar

# L1 Lyapunov orbit
x0, vy0, T = 0.812, 0.2511052120, 2.933921
N_SEED = 14
D_PERT = 70.0/lstar
T_MAN  = 4.5           # one clean sweep around toward Earth
Rm = 1737.4/lstar

# ---- ASSET CR3BP for propagation ----
class CR3BP(oc.ODEBase):
    def __init__(self, mu):
        args = oc.ODEArguments(6, 0)
        xv = args.XVec()
        x, y, z    = xv[0], xv[1], xv[2]
        vx, vy, vz = xv[3], xv[4], xv[5]
        r1c = ((x + mu)*(x + mu) + y*y + z*z) ** 1.5
        r2c = ((x - 1 + mu)*(x - 1 + mu) + y*y + z*z) ** 1.5
        ax =  2*vy + x - (1-mu)*(x + mu)/r1c - mu*(x - 1 + mu)/r2c
        ay = -2*vx + y - (1-mu)*y/r1c        - mu*y/r2c
        az =            -(1-mu)*z/r1c        - mu*z/r2c
        super().__init__(vf.stack([vx, vy, vz, ax, ay, az]), 6, 0)

ode = CR3BP(mu)
integ = ode.integrator(1e-11)
integ.setStepSizes(0.001, 1e-9, 0.05)

# ---- monodromy / unstable direction (variational equations) ----
def accel(s):
    x,y,z,vx,vy,vz = s
    s1 = ((x-xE)**2+y**2+z**2)**1.5; s2 = ((x-xM)**2+y**2+z**2)**1.5
    return np.array([vx,vy,vz, 2*vy+x-(1-mu)*(x-xE)/s1-mu*(x-xM)/s2,
                     -2*vx+y-(1-mu)*y/s1-mu*y/s2, -(1-mu)*z/s1-mu*z/s2])
def Amat(s):
    x,y,z=s[:3]; r1=np.sqrt((x-xE)**2+y**2+z**2); r2=np.sqrt((x-xM)**2+y**2+z**2)
    Uxx=1-(1-mu)/r1**3-mu/r2**3+3*(1-mu)*(x-xE)**2/r1**5+3*mu*(x-xM)**2/r2**5
    Uyy=1-(1-mu)/r1**3-mu/r2**3+3*(1-mu)*y**2/r1**5+3*mu*y**2/r2**5
    Uxy=3*(1-mu)*(x-xE)*y/r1**5+3*mu*(x-xM)*y/r2**5
    A=np.zeros((6,6)); A[0,3]=A[1,4]=A[2,5]=1; A[3,0]=Uxx;A[3,1]=Uxy;A[3,4]=2
    A[4,0]=Uxy;A[4,1]=Uyy;A[4,3]=-2; A[5,2]=-(1-mu)/r1**3-mu/r2**3; return A
def rhs(t,Y): s=Y[:6];P=Y[6:].reshape(6,6); return np.concatenate([accel(s),(Amat(s)@P).flatten()])

Y0 = np.concatenate([[x0,0,0,0,vy0,0], np.eye(6).flatten()])
vsol = solve_ivp(rhs,[0,T],Y0,rtol=1e-12,atol=1e-12,max_step=0.02,dense_output=True)
M = vsol.y[6:,-1].reshape(6,6); w,V = np.linalg.eig(M)
iu = np.argmax(np.abs(w)); isb = np.argmin(np.abs(w))
vu = np.real(V[:,iu]); vs = np.real(V[:,isb])
print(f"Unstable eigenvalue = {np.abs(w[iu]):.1f}")

def trim(arr):
    # cut a leg before it breaches the Earth (1.2 Re) or Moon (1.4 Rm)
    r1 = np.sqrt((arr[:,0]-xE)**2 + arr[:,1]**2)
    r2 = np.sqrt((arr[:,0]-xM)**2 + arr[:,1]**2)
    bad = (r1 < 1.2*Re) | (r2 < 1.4*Rm)
    return np.argmax(bad) if np.any(bad) else len(arr)

rows = []
for t in np.linspace(0,T,400):
    s = vsol.sol(t)[:6]; rows.append([0,0,s[0],s[1]])

tg = np.linspace(0,T,N_SEED,endpoint=False); leg = 0
minMoon = 1e9; minEarth = 1e9
for t in tg:
    Phi = vsol.sol(t)[6:].reshape(6,6); s = vsol.sol(t)[:6]
    du = Phi@vu; du /= np.linalg.norm(du[:3])
    ds = Phi@vs; ds /= np.linalg.norm(ds[:3])
    for sign in (+1,-1):
        leg += 1
        # unstable: forward in time
        Xu = np.concatenate([s+sign*D_PERT*du,[0.0]]).reshape(7,1)
        au = np.array([p.flatten() for p in integ.integrate_dense(Xu,  T_MAN)])
        cu = trim(au); cu = cu if cu>0 else len(au)
        for k in range(cu): rows.append([1, leg, au[k,0], au[k,1]])
        # stable: backward in time
        Xs = np.concatenate([s+sign*D_PERT*ds,[0.0]]).reshape(7,1)
        as_ = np.array([p.flatten() for p in integ.integrate_dense(Xs, -T_MAN)])
        cs = trim(as_); cs = cs if cs>0 else len(as_)
        for k in range(cs): rows.append([2, leg, as_[k,0], as_[k,1]])
        for arr,c in [(au,cu),(as_,cs)]:
            r2 = np.sqrt((arr[:c,0]-xM)**2 + arr[:c,1]**2)
            r1 = np.sqrt((arr[:c,0]-xE)**2 + arr[:c,1]**2)
            minMoon  = min(minMoon,  r2.min()*lstar - 1737.4)
            minEarth = min(minEarth, r1.min()*lstar - 6378.0)

rows = np.array(rows)
np.savetxt("manifolds_L1.csv", rows, delimiter=",", header="branch,leg,x,y", comments="")
print(f"Saved manifolds_L1.csv ({len(rows)} rows, {leg} legs)")
print(f"PROOF no intersection -> min lunar altitude {minMoon:.0f} km, min Earth altitude {minEarth:.0f} km")
