import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt

pi2 = 0.01215

def cr3bp(t, Y, pi2):
    x, y, z = Y[0], Y[1], Y[2]
    xdot, ydot, zdot = Y[3], Y[4], Y[5]
    sigma = np.sqrt((x + pi2)**2 + y**2 + z**2)
    psi   = np.sqrt((x - 1 + pi2)**2 + y**2 + z**2)
    xddot = (2*ydot + x - (1-pi2)/sigma**3 * (x + pi2) - pi2/psi**3 * (x - 1 + pi2))
    yddot = (-2*xdot + y - (1-pi2)/sigma**3 * y - pi2/psi**3 * y)
    zddot = (-(1-pi2)/sigma**3 * z - pi2/psi**3 * z)
    return [xdot, ydot, zdot, xddot, yddot, zddot]

def jacobi(Y, pi2):
    x, y, z = Y[0], Y[1], Y[2]
    xdot, ydot, zdot = Y[3], Y[4], Y[5]
    sigma = np.sqrt((x + pi2)**2 + y**2 + z**2)
    psi   = np.sqrt((x - 1 + pi2)**2 + y**2 + z**2)
    C = (x**2 + y**2
         + 2*(1-pi2)/sigma
         + 2*pi2/psi
         - (xdot**2 + ydot**2 + zdot**2))
    return C

Y0 = [1.2, 0.0, 0.0, 0.0, -1.0, 0.0]
t0 = 0
tf = 12.0
t_points = np.linspace(t0, tf, 5000)

sol = solve_ivp(lambda t, Y: cr3bp(t, Y, pi2), [t0, tf], Y0, t_eval=t_points, rtol=1e-10, atol=1e-10)

x_sol = sol.y[0]
y_sol = sol.y[1]

C_vals = np.array([jacobi(sol.y[:,i], pi2) for i in range(len(t_points))])
C0 = C_vals[0]
dC = np.abs(C_vals - C0)

print(f"Initial Jacobi constant C0 = {C0:.6f}")
print(f"Max drift |dC| = {np.max(dC):.2e}")

def lagrange_eq(x, pi2):
    r1 = abs(x + pi2)
    r2 = abs(x - 1 + pi2)
    return x - (1-pi2)*(x+pi2)/r1**3 - pi2*(x-1+pi2)/r2**3

L1x = brentq(lagrange_eq, 0.80, 0.85, args=(pi2,))
L2x = brentq(lagrange_eq, 1.01, 1.3,  args=(pi2,))
L3x = brentq(lagrange_eq, -1.3, -0.9, args=(pi2,))
L4x, L4y = 0.5 - pi2,  np.sqrt(3)/2
L5x, L5y = 0.5 - pi2, -np.sqrt(3)/2

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
ax.plot(x_sol, y_sol, 'r-', linewidth=0.8, label='Spacecraft')
ax.plot(-pi2, 0, 'bo', markersize=12, label='Earth (m1)')
ax.plot(1-pi2, 0, 'ko', markersize=6, label='Moon (m2)')
ax.plot(0, 0, 'g+', markersize=10, label='Barycenter')
ax.plot(L1x, 0,   'r*', markersize=15, label=f'L1 ({L1x:.3f}, 0)')
ax.plot(L2x, 0,   'g*', markersize=15, label=f'L2 ({L2x:.3f}, 0)')
ax.plot(L3x, 0,   'b*', markersize=15, label=f'L3 ({L3x:.3f}, 0)')
ax.plot(L4x, L4y, 'm*', markersize=15, label=f'L4 ({L4x:.3f}, {L4y:.3f})')
ax.plot(L5x, L5y, 'c*', markersize=15, label=f'L5 ({L5x:.3f}, {L5y:.3f})')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel('x* (non-dimensional)')
ax.set_ylabel('y* (non-dimensional)')
ax.set_title('CR3BP: Earth-Moon System with Lagrange Points')
ax.legend(loc='upper right', fontsize=8)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.semilogy(t_points, dC + 1e-16, 'b-', linewidth=1.2)
ax2.set_xlabel('Time (non-dimensional)')
ax2.set_ylabel('|ΔC| = |C(t) - C₀|')
ax2.set_title(f'Jacobi Constant Drift (solve_ivp)\nC₀ = {C0:.4f}  Max drift = {np.max(dC):.2e}')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cr3bp_orbit.png', dpi=150)
print("Plot saved as cr3bp_orbit.png")
