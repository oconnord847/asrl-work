import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import shutil

mu = 0.012150585609624
L1x = 0.83691513
L2x = 1.15568217
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

def find_crossing_up(x0,yd0,t_max=15):
    Y0=np.array([x0,0,0,0,yd0,0])
    state0=np.concatenate([Y0,np.eye(6).flatten()])
    sol=solve_ivp(cr3bp_stm,[0,t_max],state0,
                  rtol=1e-12,atol=1e-12,dense_output=True,max_step=0.005)
    y=sol.y[1]
    for i in range(1,len(y)):
        if sol.t[i]>0.5 and y[i-1]<=0 and y[i]>0:
            tc=brentq(lambda t: sol.sol(t)[1],sol.t[i-1],sol.t[i],xtol=1e-12)
            return tc,sol.sol(tc)
    return None,None

def shoot_ydot(x0,ydot0,find_fn,tol=1e-12,max_iter=50):
    for i in range(max_iter):
        tc,sf=find_fn(x0,ydot0)
        if tc is None: return None,None
        Phi=sf[6:].reshape(6,6); xdf=sf[3]
        if abs(xdf)<tol: return x0,ydot0
        denom=Phi[3,4]
        if abs(denom)<1e-14: return None,None
        ydot0-=xdf/denom
    return None,None

def shoot_x0_brentq(yd0_fixed,x0_lo,x0_hi,find_fn,tol=1e-10):
    def f(x0):
        _,sf=find_fn(x0,yd0_fixed)
        if sf is None: return np.nan
        return sf[3]
    try:
        x0_sol=brentq(f,x0_lo,x0_hi,xtol=tol,maxiter=50)
        return x0_sol
    except:
        return None

def integrate_full(x0c,yd0c,find_fn):
    tc,_=find_fn(x0c,yd0c)
    if tc is None: return None,None
    sf=solve_ivp(cr3bp,[0,tc*2],np.array([x0c,0,0,0,yd0c,0]),
                 t_eval=np.linspace(0,tc*2,3000),rtol=1e-12,atol=1e-12)
    return sf.y[0],sf.y[1]

fig,ax=plt.subplots(figsize=(13,9))

# ── L1 family ──
print("Computing L1 family...")
L1_cases=[(0.820,0.51),(0.800,0.53),(0.780,0.55),(0.760,0.59),(0.740,0.55),(0.720,0.65),(0.700,0.69)]
cl1=plt.cm.Blues(np.linspace(0.35,0.95,len(L1_cases)))
for idx,(x0,yd0) in enumerate(L1_cases):
    x0c,yd0c=shoot_ydot(x0,yd0,find_crossing_down)
    if x0c is None: print(f"  L1 x0={x0} skipped"); continue
    print(f"  OK x0={x0c:.4f} yd0={yd0c:.4f}")
    xs,ys=integrate_full(x0c,yd0c,find_crossing_down)
    if xs is not None: ax.plot(xs,ys,color=cl1[idx],linewidth=1.5)

# ── L2 family ──
print("Computing L2 family...")
L2_cases=[(1.16,-0.05),(1.17,-0.10),(1.18,-0.15),(1.19,-0.25),(1.21,-0.40),(1.23,-0.45)]
cl2=plt.cm.Oranges(np.linspace(0.35,0.95,len(L2_cases)))
for idx,(x0,yd0) in enumerate(L2_cases):
    x0c,yd0c=shoot_ydot(x0,yd0,find_crossing_up)
    if x0c is None: print(f"  L2 x0={x0} skipped"); continue
    print(f"  OK x0={x0c:.4f} yd0={yd0c:.4f}")
    xs,ys=integrate_full(x0c,yd0c,find_crossing_up)
    if xs is not None: ax.plot(xs,ys,color=cl2[idx],linewidth=1.5)

# ── L3 family ──
print("Computing L3 family...")
L3_cases=[
    (0.05,-1.0301,-1.0201),
    (0.10,-1.0901,-1.0801),
    (0.20,-1.1101,-1.1001),
    (0.25,-1.1401,-1.1301),
    (0.30,-1.1601,-1.1501),
]
cl3=plt.cm.Greens(np.linspace(0.35,0.95,len(L3_cases)))
for idx,(yd0,xlo,xhi) in enumerate(L3_cases):
    x0c=shoot_x0_brentq(yd0,xlo,xhi,find_crossing_down)
    if x0c is None: print(f"  L3 yd0={yd0} skipped"); continue
    tc,sf=find_crossing_down(x0c,yd0)
    if tc is None: continue
    xdf=sf[3]
    print(f"  OK x0={x0c:.4f} yd0={yd0:.4f} xdot_f={xdf:.2e}")
    xs,ys=integrate_full(x0c,yd0,find_crossing_down)
    if xs is not None: ax.plot(xs,ys,color=cl3[idx],linewidth=1.5)

# ── Labels ──
ax.plot(-mu,0,'bo',markersize=13,zorder=5)
ax.plot(1-mu,0,'o',color='gray',markersize=8,zorder=5)
ax.plot(L1x,0,'r*',markersize=13,zorder=5)
ax.plot(L2x,0,'m*',markersize=13,zorder=5)
ax.plot(L3x,0,'g*',markersize=13,zorder=5)
ax.axhline(0,color='k',linewidth=0.4,alpha=0.3)

legend_elements=[
    Line2D([0],[0],color=plt.cm.Blues(0.7),lw=2,label='L1 Lyapunov Family'),
    Line2D([0],[0],color=plt.cm.Oranges(0.7),lw=2,label='L2 Lyapunov Family'),
    Line2D([0],[0],color=plt.cm.Greens(0.7),lw=2,label='L3 Lyapunov Family'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='b',ms=10,label='Earth'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='gray',ms=8,label='Moon'),
    Line2D([0],[0],marker='*',color='w',markerfacecolor='r',ms=12,label='L1'),
    Line2D([0],[0],marker='*',color='w',markerfacecolor='m',ms=12,label='L2'),
    Line2D([0],[0],marker='*',color='w',markerfacecolor='g',ms=12,label='L3'),
]
ax.legend(handles=legend_elements,fontsize=10,loc='lower left', bbox_to_anchor=(0.28, 0.02))
ax.set_xlabel('x* (non-dimensional)',fontsize=12)
ax.set_ylabel('y* (non-dimensional)',fontsize=12)
ax.set_title('Module 4: L1, L2, L3 Lyapunov Orbit Families\nSingle Shooting - Earth-Moon CR3BP',fontsize=13)
ax.set_aspect('equal')
ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('lyapunov_all.png',dpi=150)
shutil.copy('lyapunov_all.png', '/mnt/c/Users/oconn/Downloads/lyapunov_all.png')
print("Saved lyapunov_all.png")

# ── STM Newton equation image ──
fig_eq, ax_eq = plt.subplots(figsize=(5, 2))
ax_eq.axis('off')
ax_eq.text(0.5, 0.65,
           r'$\dot{y}_0^{j+1} = \dot{y}_0^{j} - \dfrac{\dot{x}_c}{\Phi_{34}}$',
           fontsize=32, ha='center', va='center', transform=ax_eq.transAxes)
ax_eq.text(0.5, 0.18,
           r'$\Phi_{34} = \dfrac{\partial \dot{x}_c}{\partial \dot{y}_0}$',
           fontsize=20, ha='center', va='center', transform=ax_eq.transAxes, color='gray')
plt.tight_layout()
plt.savefig('stm_equation.png', dpi=150, bbox_inches='tight', transparent=True)
shutil.copy('stm_equation.png', '/mnt/c/Users/oconn/Downloads/stm_equation.png')
print("Saved stm_equation.png")

# ── STM matrix image ──
fig_stm, ax_stm = plt.subplots(figsize=(9, 5.5))
ax_stm.axis('off')
ax_stm.set_facecolor('white')
fig_stm.patch.set_facecolor('white')

ax_stm.text(0.5, 0.97, r'State Transition Matrix $\Phi(t, t_0)$ — 6×6',
            fontsize=13, ha='center', va='top', transform=ax_stm.transAxes, fontweight='bold')

state_vars = [r'$x$', r'$y$', r'$z$', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$']
init_vars  = [r'$x_0$', r'$y_0$', r'$z_0$', r'$\dot{x}_0$', r'$\dot{y}_0$', r'$\dot{z}_0$']

x_start = 0.20
y_start = 0.82
dx = 0.125
dy = 0.14

# Column headers
for j, lbl in enumerate(init_vars):
    ax_stm.text(x_start + j*dx, y_start + 0.07,
                r'$\partial$/' + r'$\partial$' + lbl,
                fontsize=8, ha='center', va='center', transform=ax_stm.transAxes, color='dimgray')

# Row headers and cells
for i, row_lbl in enumerate(state_vars):
    ax_stm.text(x_start - 0.10, y_start - i*dy,
                r'$\partial$' + row_lbl,
                fontsize=8, ha='center', va='center', transform=ax_stm.transAxes, color='dimgray')
    for j in range(6):
        highlighted = (i == 3 and j == 4)
        entry = r'$\Phi_{' + str(i) + str(j) + r'}$'
        if highlighted:
            bbox = dict(boxstyle='round,pad=0.4', facecolor='steelblue', alpha=0.85, edgecolor='steelblue')
            color = 'white'
            fw = 'bold'
            fs = 12
        else:
            bbox = dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', alpha=0.9, edgecolor='#cccccc')
            color = '#333333'
            fw = 'normal'
            fs = 10
        ax_stm.text(x_start + j*dx, y_start - i*dy, entry,
                    fontsize=fs, ha='center', va='center', transform=ax_stm.transAxes,
                    color=color, fontweight=fw, bbox=bbox)

ax_stm.text(0.5, 0.04,
            r'$\Phi_{34}$ (highlighted) $= \partial\dot{x}_c/\partial\dot{y}_0$ — sensitivity used in Newton corrector',
            fontsize=10, ha='center', va='bottom', transform=ax_stm.transAxes, color='steelblue')

plt.tight_layout()
plt.savefig('stm_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
shutil.copy('stm_matrix.png', '/mnt/c/Users/oconn/Downloads/stm_matrix.png')
print("Saved stm_matrix.png")
