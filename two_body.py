import numpy as np
from scipy.integrate import solve_ivp

G = 6.67430e-20
m_1 = 5.97219e24
m_2 = 1000
mu = G * m_1
R_E = 6378.12

r_0 = np.array([8000, 0, 6000])
v_0 = np.array([0, 7, 0])
Y_0 = np.hstack((r_0, v_0))

def relative_motion(t, Y):
    r_vec = Y[:3]
    Ydot = np.zeros_like(Y)
    Ydot[:3] = Y[3:]
    r = np.sqrt(np.sum(np.square(r_vec)))
    Ydot[3:] = -mu * r_vec / r**3
    return Ydot

t_0 = 0
t_f = 14709
t_points = np.linspace(t_0, t_f, 1000)
sol = solve_ivp(relative_motion, [t_0, t_f], Y_0, t_eval=t_points)

Y = sol.y.T
r = Y[:, :3]
v = Y[:, 3:]

r_mag = np.sqrt(np.sum(np.square(r), axis=1))
altitude = r_mag - R_E
speed = np.sqrt(np.sum(np.square(v), axis=1))

min_alt = np.min(altitude)
max_alt = np.max(altitude)
i_min = np.argmin(altitude)
i_max = np.argmax(altitude)

print(f"Min altitude: {min_alt:.2f} km")
print(f"Speed at min altitude: {speed[i_min]:.2f} km/s")
print(f"Max altitude: {max_alt:.2f} km")
print(f"Speed at max altitude: {speed[i_max]:.4f} km/s")
