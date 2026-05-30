import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mu = 0.012150585609624

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

def make_event(direction,min_t=0.5):
    def ev(t,state):
        if t<min_t: return 1.0
        return state[1]
    ev.terminal=False; ev.direction=direction
    return ev

def single_shooter(x0,ydot0,tol=1e-12,max_iter=50,direction=-1,min_t=0.5):
    ev=make_event(direction,min_t)
    for i in range(max_iter):
        Y0=np.array([x0,0,0,0,ydot0,0])
        state0=np.concatenate([Y0,np.eye(6).flatten()])
        sol=solve_ivp(cr3bp_stm,[0,40],state0,events=ev,
                      rtol=1e-12,atol=1e-12,dense_output=True,max_step=0.01)
        valid=[tc for tc in sol.t_events[0] if tc>min_t]
        if not valid: return None,None
        t_half=valid[0];sf=sol.sol(t_half)
        Phi_f=sf[6:].reshape(6,6);xdot_f=sf[3]
        if abs(xdot_f)<tol: return x0,ydot0
        denom=Phi_f[3,4]
        if abs(denom)<1e-14: return None,None
        ydot0-=xdot_f/denom
    return None,None

def integrate_orbit(x0c,yd0c,direction=-1,min_t=0.5):
    ev=make_event(direction,min_t)
    Y0=np.array([x0c,0,0,0,yd0c,0])
    state0=np.concatenate([Y0,np.eye(6).flatten()])
    sol=solve_ivp(cr3bp_stm,[0,40],state0,events=ev,
                  rtol=1e-12,atol=1e-12,dense_output=True,max_step=0.01)
    valid=[tc for tc in sol.t_events[0] if tc>min_t]
    if not valid: return None,None
    t_period=valid[0]*2
    sf=solve_ivp(cr3bp,[0,t_period],np.array([x0c,0,0,0,yd0c,0]),
                 t_eval=np.linspace(0,t_period,3000),rtol=1e-12,atol=1e-12)
    return sf.y[0],sf.y[1]

L1_cases=[(0.820,0.51),(0.800,0.53),(0.780,0.55)]
L2_cases=[(0.90,-0.40)]
L3_cases=[(-1.02,0.05),(-1.05,0.10)]

fig,ax=plt.subplots(figsize=(13,10))
L1x=0.83691513; L2x=1.15568217; L3x=-1.00506265

print("Computing L1 family...")
cl1=plt.cm.Blues(np.linspace(0.4,0.95,len(L1_cases)))
for idx,(x0,yd0) in enumerate(L1_cases):
    x0c,yd0c=single_shooter(x0,yd0,direction=-1)
    if x0c is None: print(f"  L1 x0={x0} skipped"); continue
    print(f"  OK x0={x0c:.4f} ydot0={yd0c:.4f}")
    xs,ys=integrate_orbit(x0c,yd0c,direction=-1)
    if xs is not None: ax.plot(xs,ys,color=cl1[idx],linewidth=1.5)

print("Computing L2 family...")
cl2=plt.cm.Oranges(np.linspace(0.4,0.95,len(L2_cases)))
for idx,(x0,yd0) in enumerate(L2_cases):
    x0c,yd0c=single_shooter(x0,yd0,direction=1)
    if x0c is None: print(f"  L2 x0={x0} skipped"); continue
    print(f"  OK x0={x0c:.4f} ydot0={yd0c:.4f}")
    xs,ys=integrate_orbit(x0c,yd0c,direction=1)
    if xs is not None: ax.plot(xs,ys,color=cl2[idx],linewidth=1.5)

print("Computing L3 family...")
cl3=plt.cm.Greens(np.linspace(0.4,0.95,len(L3_cases)))
for idx,(x0,yd0) in enumerate(L3_cases):
    x0c,yd0c=single_shooter(x0,yd0,direction=-1,min_t=1.0)
    if x0c is None: print(f"  L3 x0={x0} skipped"); continue
    print(f"  OK x0={x0c:.4f} ydot0={yd0c:.4f}")
    xs,ys=integrate_orbit(x0c,yd0c,direction=-1,min_t=1.0)
    if xs is not None: ax.plot(xs,ys,color=cl3[idx],linewidth=1.5)

ax.plot(-mu,0,'bo',markersize=14,zorder=5)
ax.plot(1-mu,0,'o',color='gray',markersize=8,zorder=5)
ax.plot(L1x,0,'r*',markersize=14,zorder=5)
ax.plot(L2x,0,'m*',markersize=14,zorder=5)
ax.plot(L3x,0,'g*',markersize=14,zorder=5)
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
ax.legend(handles=legend_elements,fontsize=10,loc='upper left')
ax.set_xlabel('x* (non-dimensional)',fontsize=12)
ax.set_ylabel('y* (non-dimensional)',fontsize=12)
ax.set_title('Module 4: L1, L2, L3 Lyapunov Orbit Families\nSingle Shooting - Earth-Moon CR3BP',fontsize=13)
ax.set_aspect('equal')
ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('lyapunov_families.png',dpi=150)
print("Saved lyapunov_families.png")
