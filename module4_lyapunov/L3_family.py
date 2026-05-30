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

def get_xdotf_stm(x0, yd0, t_max=15):
    Y0=np.array([x0,0,0,0,yd0,0])
    state0=np.concatenate([Y0,np.eye(6).flatten()])
    sol=solve_ivp(cr3bp_stm,[0,t_max],state0,
                  rtol=1e-12,atol=1e-12,dense_output=True,max_step=0.005)
    y=sol.y[1]
    for i in range(1,len(y)):
        if sol.t[i]>0.5 and y[i-1]>=0 and y[i]<0:
            tc=brentq(lambda t: sol.sol(t)[1],sol.t[i-1],sol.t[i],xtol=1e-12)
            sf=sol.sol(tc)
            return tc, sf[3], sf[6:].reshape(6,6)
    return None, None, None

def shoot_x0(yd0_fixed, x0_lo, x0_hi, tol=1e-12, max_iter=50):
    # Use brentq on x0 to find xdot_f=0
    def f(x0):
        _, xdf, _ = get_xdotf_stm(x0, yd0_fixed)
        if xdf is None: return np.nan
        return xdf
    try:
        x0_sol = brentq(f, x0_lo, x0_hi, xtol=1e-10, maxiter=max_iter)
        return x0_sol
    except:
        return None

# Known brackets from scan
cases = [
    (0.05, -1.0301, -1.0201),
    (0.10, -1.0901, -1.0801),
    (0.20, -1.1101, -1.1001),
    (0.25, -1.1401, -1.1301),
    (0.30, -1.1601, -1.1501),
]

fig, ax = plt.subplots(figsize=(8,8))
colors = plt.cm.Greens(np.linspace(0.3, 0.95, len(cases)))

for idx, (yd0, xlo, xhi) in enumerate(cases):
    print(f"Solving L3 yd0={yd0:.2f}...")
    x0c = shoot_x0(yd0, xlo, xhi)
    if x0c is None: print("  skipped"); continue
    tc, xdf, _ = get_xdotf_stm(x0c, yd0)
    print(f"  OK x0={x0c:.6f} yd0={yd0:.4f} xdot_f={xdf:.3e} t_half={tc:.4f}")
    t_period = tc * 2
    sf = solve_ivp(cr3bp, [0,t_period], np.array([x0c,0,0,0,yd0,0]),
                   t_eval=np.linspace(0,t_period,3000), rtol=1e-12, atol=1e-12)
    ax.plot(sf.y[0], sf.y[1], color=colors[idx], linewidth=1.8)

ax.plot(-mu,0,'bo',markersize=12,zorder=5,label='Earth')
ax.plot(1-mu,0,'o',color='gray',markersize=7,zorder=5,label='Moon')
ax.plot(L3x,0,'g*',markersize=14,zorder=5,label='L3')
ax.axhline(0,color='k',linewidth=0.4,alpha=0.3)
ax.set_xlabel('x* (non-dimensional)',fontsize=12)
ax.set_ylabel('y* (non-dimensional)',fontsize=12)
ax.set_title('L3 Lyapunov Orbit Family\nSingle Shooting - Earth-Moon CR3BP',fontsize=13)
ax.legend(fontsize=11)
ax.set_aspect('equal')
ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('L3_family.png',dpi=150)
print("Saved L3_family.png")
