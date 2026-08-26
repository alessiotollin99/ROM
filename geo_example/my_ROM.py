import numpy as np
import matplotlib.pyplot as plt
import time

# ==================================================================
# Data loading
# ==================================================================
data = np.load(r"C:\Users\tollale15722\Desktop\Projects\ROM\base_case_sol\fdm_results_20260826_170226.npz")
my_sol = data["base_system"]
A = data["A"]
n_steps = data["n_steps"]
n_mesh = data["n_mesh"]

# ==================================================================
# main data
# ==================================================================

k = 1.83
cp = 1600
rho = 1400

a = k / (rho * cp)

dt = 3600

r0 = 0.015
r_max = 10
n_mesh = 200

Tg = 13
Tstart = Tg
q = 50
L_he = 100

time_vec = np.linspace(0, n_steps * dt, n_steps + 1, dtype=np.float64)
dr = (r_max - r0) / n_mesh

r_i = np.arange(r0 + dr / 2, r_max, dr, dtype=np.float64)

# ==================================================================
# SVD
# ==================================================================

U, sigma, Vt = np.linalg.svd(my_sol)   # my_sol è (n_mesh, n_steps+1): righe = spazio -> U contiene i modi spaziali, ok così

fig, ax = plt.subplots(figsize=(4, 3))
plt.semilogy(sigma, marker="o")
plt.tight_layout()
plt.show()
r = int(input("Now that you were shown the singular values you can decide how many orders to keep: "))
V = U[:, :r]

sol_reduced = np.zeros((r, n_steps + 1), dtype=np.float64)
A_r = V.T @ A @ V
sol_0 = V.T @ my_sol[:, 0]          # era my_sol[0, :]
sol_reduced[:, 0] = sol_0            # era sol_reduced[0, :]

b0 = q * dr / (2 * np.pi * r0 * k) * (1 / dr**2 - 1 / (r_i[0] * 2 * dr))
b_last = 26 * (1 / dr**2 + 1 / (r_i[-1] * 2 * dr))

tic1 = time.time()

for t in range(n_steps):
    sol_full = V @ sol_reduced
    b = np.full(n_mesh, (-sol_full[:, t] / (a * dt)), dtype=np.float64)   # era sol_full[t, :]
    b[0] += -b0
    b[-1] += -b_last
    b_r = V.T @ b
    sol_reduced[:, t + 1] = np.linalg.solve(A_r, b_r)   # era sol_reduced[t+1, :]

toc1 = time.time()

print(f"The calculation time for the reduced-order model was: {toc1-tic1: .2f} s")

sol_new = V @ sol_reduced

# ==================================================================
# Plot
# ==================================================================

from matplotlib.widgets import Slider

fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(bottom=0.25)

line1, = ax.plot(r_i, my_sol[:, 0])
line2, = ax.plot(r_i, sol_new[:, 0])
ax.set_xlabel('r [m]')
ax.set_ylabel('Temperature [°C]')
ax.set_ylim(my_sol.min(), my_sol.max())
title = ax.set_title('t = 0 h')

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'step', 0, n_steps, valinit=0, valstep=1)

def update(val):
    step = int(slider.val)
    line1.set_ydata(my_sol[:, step])
    line2.set_ydata(sol_new[:, step])
    title.set_text(f't = {time_vec[step]/3600:.0f} h')
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()

# Computational time: xx.xx s