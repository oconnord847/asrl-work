import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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

# L2 is at x=1.1557 — use x0 values on far side (x0 > L2x)
# crossing direction is upward (+1)
def y_cross_up(t,state):
    if t<0.5: return -1.0
    return state[1]
y_cross_up.terminal=False;y_cross_up.direction=1

def single_shooter_L2(x0,ydot0,tol=1e-12,max_iter=50):
    for i in range(max_iter):
        Y0=np.array([x0,0,0,0,ydot0,0])
        state0=np.concatenate([Y0,np.eye(6).flatten()])
        sol=solve_ivp(cr3bp_stm,[0,20],state0,events=y_cross_up,
                      rtol=1e-12,atol=1e-12,dense_output=True,max_step=0.01)
        valid=[tc for tc in sol.t_events[0] if tc>0.5]
        if not valid: return None,None
        t_half=valid[0];sf=sol.sol(t_half)
        Phi_f=sf[6:].reshape(6,6);xdot_f=sf[3]
        if abs(xdot_f)<tol: return x0,ydot0
        denom=Phi_f[3,4]
        if abs(denom)<1e-14: return None,None
        ydot0-=xdot_f/denom
    return None,None

# x0 > L2x = 1.1557, negative ydot0
cases=[(1.18,-0.25),(1.20,-0.30),(1.22,-0.35),(1.25,-0.40),(1.28,-0.45),(1.32,-0.50),(1.36,-0.55)]
L2x=1.15568217

fig,ax=plt.subplots(figsize=(8,8))
colors=plt.cm.Oranges(np.linspace(0.35,0.95,len(cases)))

for idx,(x0,yd0) in enumerate(cases):
    x0c,yd0c=single_shooter_L2(x0,yd0)
    if x0c is None: print(f"skipped x0={x0}"); continue
    print(f"OK x0={x0c:.4f} ydot0={yd0c:.4f}")
    Y0=np.array([x0c,0,0,0,yd0c,0])
    state0=np.concatenate([Y0,np.eye(6).flatten()])
    sol=solve_ivp(cr3bp_stm,[0,20],state0,events=y_cross_up,
                  rtol=1e-12,atol=1e-12,dense_output=True,max_step=0.01)
    valid=[tc for tc in sol.t_events[0] if tc>0.5]
    if not valid: continue
    t_period=valid[0]*2
    sf=solve_ivp(cr3bp,[0,t_period],np.array([x0c,0,0,0,yd0c,0]),
                 t_eval=np.linspace(0,t_period,3000),rtol=1e-12,atol=1e-12)
    ax.plot(sf.y[0],sf.y[1],color=colors[idx],linewidth=1.8)

ax.plot(-mu,0,'bo',markersize=12,zorder=5,label='Earth')
ax.plot(1-mu,0,'o',color='gray',markersize=7,zorder=5,label='Moon')
ax.plot(L2x,0,'m*',markersize=14,zorder=5,label='L2')
ax.axhline(0,color='k',linewidth=0.4,alpha=0.3)
ax.set_xlabel('x* (non-dimensional)',fontsize=12)
ax.set_ylabel('y* (non-dimensional)',fontsize=12)
ax.set_title('L2 Lyapunov Orbit Family\nSingle Shooting — Earth-Moon CR3BP',fontsize=13)
ax.legend(fontsize=11)
ax.set_aspect('equal')
ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('L2_family.png',dpi=150)
print("Saved L2_family.png")
