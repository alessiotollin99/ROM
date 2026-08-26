import numpy as np
from scipy.sparse import diags
from pathlib import Path
from datetime import datetime
import time

k = 1.83  # ground thermal conductivity [W / m K]
cp = 1600  # ground specific thermal capacity [J / kg K]
rho = 1400  # ground density [kg / m3]

a = k / (rho * cp)  # ground thermal diffusivity [m2 / s]

n_steps = 8760  # n° simulation steps
dt = 3600  # timestep [s]

r0 = 0.015  # borehole radius [m]
r_max = 10  # maximum radius [m]
n_mesh = 200  # radial mesh [-]

Tg = 13  # undisturbed ground temeprature [°C]
Tstart = Tg  # staring point [°C]
q = 50  # heat injected [W / m]
L_he = 100 # heat exchanger length [m]

time_vec = np.linspace(0, n_steps * dt, n_steps + 1, dtype=np.float64)
dr = (r_max - r0) / n_mesh

# ==================================================================
# Matrix construction
# ==================================================================
r_i = np.arange(r0 + dr / 2, r_max, dr, dtype=np.float64)

d = np.full(n_mesh, (-1 / (a * dt) - 2 / dr**2), dtype=np.float64)
d[0] += 1 / dr**2 - 1 / (2 * dr * r_i[0])
d[-1] += - (1 / dr**2 + 1 / (r_i[-1] * 2 * dr))

d_u = 1 / dr**2 + 1 / (2 * dr * r_i[:-1])
d_l = 1 / dr**2 - 1 / (2 * dr * r_i[1:])

diagonals = [d, d_l, d_u]
A = diags(diagonals, [0, -1, 1]).toarray()

# ==================================================================
# Inizialization
# ==================================================================

x0 = np.full(n_mesh, Tstart, dtype = np.float64)
my_sol = np.zeros((n_mesh, n_steps + 1), dtype = np.float64)
my_sol[:, 0] = x0

# ==================================================================
# Cycle
# ==================================================================
b0 = q * dr / (2 * np.pi * r0 * k) * (1 / dr**2 - 1 / (r_i[0] * 2 * dr))
b_last = 26 * (1 / dr**2 + 1 / (r_i[-1] * 2 * dr))

tic1 = time.time()

for t in range(n_steps):
    b = np.full(n_mesh, (-my_sol[:, t] / (a * dt)), dtype = np.float64)
    b[0] += - b0
    b[-1] += - b_last
    my_sol[:, t + 1] = np.linalg.solve(A, b)

toc1 = time.time()

print(f"The calculation time for the whole problem was: {toc1-tic1: .2f} s")

arrays = {"base_system": my_sol, "n_mesh": n_mesh, "n_steps": n_steps, "A": A}
output_dir = Path("base_case_sol")
output_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = output_dir / f"fdm_results_{timestamp}"

np.savez_compressed(output_path, **arrays)
print(f"Results saved to {output_path}.npz")
# ==================================================================
# Plot
# ==================================================================

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(bottom=0.25)

line, = ax.plot(r_i, my_sol[:, 0])
ax.set_xlabel('r [m]')
ax.set_ylabel('Temperature [°C]')
ax.set_ylim(my_sol.min(), my_sol.max())
title = ax.set_title('t = 0 h')

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'step', 0, n_steps, valinit=0, valstep=1)

def update(val):
    step = int(slider.val)
    line.set_ydata(my_sol[:, step])
    title.set_text(f't = {time_vec[step]/3600:.0f} h')
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()

# Computational time: 139.37 s