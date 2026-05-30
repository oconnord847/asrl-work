import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt

mu = 0.012150585609624
L3x = -1.00506265

def cr3bp(t, Y):
    x,y,z=Y[0],Y[1],Y[2]; xd,yd,zd=Y[3],Y[4],Y[5]
    s=np.sqrt((x+mu)**2+y**2+z**2); p=np.sqrt((x-1+mu)**2+y**2+z**2)
    return [xd,yd,zd,
            2*yd+x-(1-mu)/s**3*(x+mu)-mu/p**3*(x-1+mu),
            -2*xd+y-(1-mu)/s**3*y-mu/p**3*y,
            -(1-mu)/s**3*z-mu/p**3*z]

def A_matrix(x,y,z):
    s=np.sqrt((x+mu)**2+y**2+z**2); p=np.sqrt((x-1+mu)**2+y**2+z**2)
    s3=s**3;s5=s**5;p3=p**3;p5=p**5
    Uxx=1-(1-mu)/s3+3*(1-mu)*(x+mu)**2/s5-mu/p3+3*mu*(x-1+mu)**2/p5
    Uyy=1-(1-mu)/s3+3*(1-mu)*y**2/s5-mu/p3+3*mu*y**2/p5
    Uzz=-(1-mu)/s3+3*(1-mu)*z**2/s5-mu/p3+3*mu*z**2/p5
    Uxy=3*(1-mu)*(x+mu)*y/s5+3*mu*(x-1+mu)*y/p5
    Uxz=3*(1-mu)*(x+mu)*z/s5+3*mu*(x-1+mu)*z/p5
    Uyz=3*(1-mu)*y*z/s5+3*mu*y*z/p5
    A=np.zeros((6,6))
    A[0,3]=1;A[1,4]=1;A[2,5]=1
    A[3,0]=Uxx;A[3,1]=Uxy;A[3,2]=Uxz;A[3,4]=2
    A[4,0]=Uxy;A[4,1]=Uyy;A[4,2]=Uyz;A[4,3]=-2
    A[5,0]=Uxz;A[5,1]=Uyz;A[5,2]=Uzz
    return A

def cr3bp_stm(t,state):
    Y=state[:6];Phi=state[6:].reshape(6,6)
    dY=cr3bp(t,Y);dPhi=A_matrix(Y[0],Y[1],Y[2])@Phi
    return np.concatenate([dY,dPhi.flatten()])

def find_crossing_down(x0,yd0,t_max=15):
    Y0=np.array([x0,0,0,0,yd0,0])
    state0=np.concatenate([Y0,np.eye(6).flatten()])
    sol=solve_ivp(cr3bp_stm,[0,t_max],state0,
                  rtol=1e-12,atol=1e-12,dense_output=True,max_step=0.005)
    y=sol.y[1]
    for i in range(1,len(y)):
        if sol.t[i]>0.5 and y[i-1]>=0 and y[i]<0:
            tc=brentq(lambda t: sol.sol(t)[1],sol.t[i-1],sol.t[i],xtol=1e-12)
            return tc,sol.sol(tc)
    return None,None

def shoot_x0_brentq(yd0_fixed,x0_lo,x0_hi):
    def f(x0):
        _,sf=find_crossing_down(x0,yd0_fixed)
        if sf is None: return np.nan
        return sf[3]
    try:
        return brentq(f,x0_lo,x0_hi,xtol=1e-10,maxiter=50)
    except:
        return None

def get_monodromy(x0c, yd0c):
    # Integrate one full period and return STM = monodromy matrix
    tc,_=find_crossing_down(x0c,yd0c)
    if tc is None: return None, None
    t_period=tc*2
    Y0=np.array([x0c,0,0,0,yd0c,0])
    state0=np.concatenate([Y0,np.eye(6).flatten()])
    sol=solve_ivp(cr3bp_stm,[0,t_period],state0,
                  rtol=1e-12,atol=1e-12,dense_output=True,max_step=0.005)
    sf=sol.sol(t_period)
    mono=sf[6:].reshape(6,6)
    return mono, t_period

# L3 family cases we already converged
L3_cases=[
    (0.05,-1.0301,-1.0201),
    (0.10,-1.0901,-1.0801),
    (0.20,-1.1101,-1.1001),
    (0.25,-1.1401,-1.1301),
    (0.30,-1.1601,-1.1501),
]

print("Computing monodromy eigenvalues along L3 family...\n")
print(f"{'yd0':>6} {'x0':>10} {'T':>8}  eigenvalues (real parts)")
print("-"*70)

results = []
for yd0,xlo,xhi in L3_cases:
    x0c=shoot_x0_brentq(yd0,xlo,xhi)
    if x0c is None: continue
    mono, T = get_monodromy(x0c, yd0)
    if mono is None: continue
    eigs = np.linalg.eigvals(mono)
    eigs_sorted = sorted(eigs, key=lambda e: abs(e.real))
    real_parts = [f"{e.real:+.4f}{'+' if e.imag>=0 else ''}{e.imag:.4f}i" for e in eigs]
    print(f"{yd0:>6.2f} {x0c:>10.6f} {T:>8.4f}  {[f'{e.real:+.3f}' for e in eigs]}")
    results.append((yd0, x0c, T, eigs))

print("\nLooking for eigenvalue pair near +1 (bifurcation indicator)...")
for yd0, x0c, T, eigs in results:
    close_to_1 = [e for e in eigs if abs(e-1.0)<0.3]
    print(f"  yd0={yd0:.2f} x0={x0c:.4f}: eigs near +1 = {[f'{e.real:.4f}+{e.imag:.4f}i' for e in close_to_1]}")
