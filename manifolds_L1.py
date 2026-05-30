import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import shutil

mu = 0.012150585609624
L1x = 0.83691513

def cr3bp_stm(t, state):
    state = np.array(state)
    x,y,z=state[0],state[1],state[2]
    xd,yd,zd=state[3],state[4],state[5]
    s=np.sqrt((x+mu)**2+y**2+z**2); p=np.sqrt((x-1+mu)**2+y**2+z**2)
    s3=s**3;s5=s**5;p3=p**3;p5=p**5
    xdd=2*yd+x-(1-mu)/s3*(x+mu)-mu/p3*(x-1+mu)
    ydd=-2*xd+y-(1-mu)/s3*y-mu/p3*y
    zdd=-(1-mu)/s3*z-mu/p3*z
    Phi=state[6:].reshape(6,6)
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
    dPhi=A@Phi
    return [xd,yd,zd,xdd,ydd,zdd]+list(dPhi.flatten())

def cr3bp(t, Y):
    return cr3bp_stm(t, list(Y)+[0]*36)[:6]

x0c  = 0.823
yd0c = 0.130407
T    = 2.746598

print(f"Using L1 orbit: x0={x0c} ydot0={yd0c} T={T}")

Y0     = np.array([x0c,0,0,0,yd0c,0])
state0 = np.concatenate([Y0, np.eye(6).flatten()])
sol_full = solve_ivp(cr3bp_stm, [0,T], state0,
                     t_eval=np.linspace(0,T,300),
                     rtol=1e-12, atol=1e-12, max_step=0.005)

orbit_states = sol_full.y[:6,:].T
orbit_phis   = sol_full.y[6:,:].T

mono = sol_full.y[6:,-1].reshape(6,6)
eigs, vecs = np.linalg.eig(mono)
mags = np.abs(eigs)

print(f"Eigenvalue magnitudes: {np.sort(mags)}")

idx_unstable = np.argmax(mags)
idx_stable   = np.argmin(mags)

eig_u = eigs[idx_unstable]
eig_s = eigs[idx_stable]
mag_u = mags[idx_unstable]
mag_s = mags[idx_stable]

print(f"Unstable eigenvalue: {eig_u:.4f} mag={mag_u:.2f}")
print(f"Stable   eigenvalue: {eig_s:.4f}  mag={mag_s:.6f}")

nu_u = np.real(vecs[:,idx_unstable])
nu_s = np.real(vecs[:,idx_stable])
nu_u /= np.linalg.norm(nu_u[:3])
nu_s /= np.linalg.norm(nu_s[:3])

# ── Monodromy matrix visual ──
fig_mono, ax_mono = plt.subplots(figsize=(10, 6))
ax_mono.axis('off')
fig_mono.patch.set_facecolor('white')

ax_mono.text(0.5, 0.97, r'Monodromy Matrix $M = \Phi(T, t_0)$ — L1 Lyapunov Orbit (x$_0$ = 0.823)',
             fontsize=13, ha='center', va='top', transform=ax_mono.transAxes, fontweight='bold')

state_vars = [r'$x$', r'$y$', r'$z$', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$']
init_vars  = [r'$x_0$', r'$y_0$', r'$z_0$', r'$\dot{x}_0$', r'$\dot{y}_0$', r'$\dot{z}_0$']

x_start = 0.18
y_start = 0.80
dx = 0.128
dy = 0.13

# Column headers
for j, lbl in enumerate(init_vars):
    ax_mono.text(x_start + j*dx, y_start + 0.07, lbl,
                 fontsize=9, ha='center', va='center', transform=ax_mono.transAxes, color='dimgray')

# Matrix entries with actual values
for i in range(6):
    ax_mono.text(x_start - 0.10, y_start - i*dy, state_vars[i],
                 fontsize=9, ha='center', va='center', transform=ax_mono.transAxes, color='dimgray')
    for j in range(6):
        val = mono[i, j]
        # Format value
        if abs(val) < 0.001:
            txt = f'{val:.1e}'
        elif abs(val) > 999:
            txt = f'{val:.1f}'
        else:
            txt = f'{val:.4f}'

        # Highlight unstable/stable eigenvector columns
        if j == idx_unstable:
            bg = dict(boxstyle='round,pad=0.3', facecolor='#d32f2f', alpha=0.75, edgecolor='#b71c1c')
            fc = 'white'; fw = 'bold'; fs = 8
        elif j == idx_stable:
            bg = dict(boxstyle='round,pad=0.3', facecolor='#1565c0', alpha=0.75, edgecolor='#0d47a1')
            fc = 'white'; fw = 'bold'; fs = 8
        else:
            bg = dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', alpha=0.9, edgecolor='#cccccc')
            fc = '#333333'; fw = 'normal'; fs = 8
        ax_mono.text(x_start + j*dx, y_start - i*dy, txt,
                     fontsize=fs, ha='center', va='center', transform=ax_mono.transAxes,
                     color=fc, fontweight=fw, bbox=bg)

# Eigenvalue summary
ax_mono.text(0.5, 0.08,
             f'Unstable eigenvalue (red column): |λ| = {mag_u:.1f}  —  Stable eigenvalue (blue column): |λ| = {mag_s:.4f}',
             fontsize=10, ha='center', va='bottom', transform=ax_mono.transAxes,
             color='#333333')
ax_mono.text(0.5, 0.02,
             r'|λ| $\gg$ 1 confirms strong hyperbolic instability — trajectories diverge exponentially from the L1 orbit',
             fontsize=10, ha='center', va='bottom', transform=ax_mono.transAxes,
             color='#d32f2f')

plt.tight_layout()
plt.savefig('monodromy_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
shutil.copy('monodromy_matrix.png', '/mnt/c/Users/oconn/Downloads/monodromy_matrix.png')
print("Saved monodromy_matrix.png")

# ── Manifolds plot ──
d        = 1e-4
n_points = 60
indices  = np.linspace(0, len(orbit_states)-2, n_points, dtype=int)
t_fwd    = 6.0
t_bwd    = 6.0

fig, ax = plt.subplots(figsize=(12,10))
print(f"Computing manifolds for {n_points} points...")

u_labeled = False
s_labeled = False

for idx in indices:
    xi    = orbit_states[idx]
    Phi_i = np.array(orbit_phis[idx]).reshape(6,6)

    nu_ui = Phi_i @ nu_u; nu_ui /= np.linalg.norm(nu_ui[:3])
    nu_si = Phi_i @ nu_s; nu_si /= np.linalg.norm(nu_si[:3])

    for sign in [+1, -1]:
        xp = xi + sign * d * nu_ui
        sol = solve_ivp(cr3bp, [0, t_fwd], xp,
                        rtol=1e-10, atol=1e-10, max_step=0.01)
        lbl = 'Unstable Manifold ($W^u$)' if not u_labeled else None
        ax.plot(sol.y[0], sol.y[1], 'r-', lw=0.5, alpha=0.6, label=lbl)
        u_labeled = True

    for sign in [+1, -1]:
        xp = xi + sign * d * nu_si
        sol = solve_ivp(cr3bp, [0, -t_bwd], xp,
                        rtol=1e-10, atol=1e-10, max_step=0.01)
        lbl = 'Stable Manifold ($W^s$)' if not s_labeled else None
        ax.plot(sol.y[0], sol.y[1], 'b-', lw=0.5, alpha=0.6, label=lbl)
        s_labeled = True

ax.plot(orbit_states[:,0], orbit_states[:,1], 'k-', lw=2.5, zorder=5)
ax.plot(-mu,  0, 'bo', ms=12, zorder=6)
ax.plot(1-mu, 0, 'o',  color='gray', ms=7, zorder=6)
ax.plot(L1x,  0, 'r*', ms=14, zorder=6)

legend_elements=[
    Line2D([0],[0],color='r',lw=1.5,label='Unstable Manifold ($W^u$)'),
    Line2D([0],[0],color='b',lw=1.5,label='Stable Manifold ($W^s$)'),
    Line2D([0],[0],color='k',lw=2,label='L1 Lyapunov Orbit'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='b',ms=10,label='Earth'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='gray',ms=7,label='Moon'),
    Line2D([0],[0],marker='*',color='w',markerfacecolor='r',ms=12,label='L1'),
]
ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
ax.set_xlabel('x* (non-dimensional)', fontsize=12)
ax.set_ylabel('y* (non-dimensional)', fontsize=12)
ax.set_title('Module 5: Invariant Manifolds — L1 Lyapunov Orbit\nEarth-Moon CR3BP', fontsize=13)
ax.set_xlim(-1.5, 1.8)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('manifolds_L1.png', dpi=150)
shutil.copy('manifolds_L1.png', '/mnt/c/Users/oconn/Downloads/manifolds_L1.png')
print("Saved manifolds_L1.png")
