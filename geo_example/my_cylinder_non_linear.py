import numpy as np
from scipy.sparse import diags
from pathlib import Path
from datetime import datetime
import time

k_dry = 1.83  # ground thermal conductivity [W / m K]
cp_dry = 1600  # ground specific thermal capacity [J / kg K]
rho_dry = 1400  # ground density [kg / m3]

a_dry = k_dry / (rho_dry * cp_dry)
beta = 1e-3 # coefficient to account for non linearity

n_steps = 8760  # n° simulation steps
dt = 3600  # timestep [s]

r0 = 0.015  # borehole radius [m]
r_max = 10  # maximum radius [m]
n_mesh = 200  # radial mesh [-]

Tg = 13  # undisturbed ground temeprature [°C]
Tstart = Tg  # staring point [°C]
q = 50  # heat injected [W / m]
L_he = 100  # heat exchanger length [m]

time_vec = np.linspace(0, n_steps * dt, n_steps + 1, dtype=np.float64)
dr = (r_max - r0) / n_mesh

T_rif = Tg
f_k = lambda T: k_dry * (1 + beta * (T - T_rif)) # non linearity function
f_a = lambda k: k / (rho_dry * cp_dry) # to evaluate thermal diffusivity at each iteration

# ==================================================================
# Matrix construction
# ==================================================================
r_i = np.arange(r0 + dr / 2, r_max, dr, dtype=np.float64)

d_0 = 1 / dr**2 - 1 / (2 * dr * r_i[0])
d_last = -(1 / dr**2 + 1 / (r_i[-1] * 2 * dr))

d_u = 1 / dr**2 + 1 / (2 * dr * r_i[:-1])
d_l = 1 / dr**2 - 1 / (2 * dr * r_i[1:])

# ==================================================================
# Inizialization
# ==================================================================

x0 = np.full(n_mesh, Tstart, dtype=np.float64)
my_sol = np.zeros((n_mesh, n_steps + 1), dtype=np.float64)
my_sol[:, 0] = x0
non_linear_sol = np.zeros((n_mesh, n_steps + 1), dtype = np.float64)

k_vec_0 = f_k(x0)
a_vec_0 = f_a(k_vec_0)
non_linear_sol[:, 0] = -1 / (a_vec_0 * dt)

# ==================================================================
# Cycle
# ==================================================================
b_last = 2 * Tg * (1 / dr**2 + 1 / (r_i[-1] * 2 * dr))

tic1 = time.time()

    
for t in range(n_steps):
    max_iter = 200
    toll = 1e-3
    iter = 0
    err = np.inf
    k_vec = f_k(my_sol[:, t]) #get the temperature for the series of nodes at that step
    my_sol_prec = my_sol[:, t]

    while iter < max_iter and err > toll:
        a_vec = f_a(k_vec)
        d_linear = - 2 / dr**2
        d_non_linear = -1 / (a_vec * dt) # to store in the snapshot matrix of non linearity
        d = d_linear + d_non_linear
        d[0] += d_0
        d[-1] += d_last
        diagonals = [d, d_l, d_u]
        A = diags(diagonals, [0, -1, 1]).toarray()

        b0 = q * dr / (2 * np.pi * r0 * k_vec[0]) * (1 / dr**2 - 1 / (r_i[0] * 2 * dr))
        b = -my_sol[:, t] / (a_vec * dt)
        b[0] += -b0
        b[-1] += -b_last
        my_sol_new = np.linalg.solve(A, b)
        k_vec = f_k(my_sol_new)
        
        iter += 1
        err = np.linalg.norm(my_sol_new - my_sol_prec)

        my_sol_prec = my_sol_new

    a_vec = k_vec / (rho_dry * cp_dry)
    d_non_linear = -1 / (a_vec * dt)
    non_linear_sol[:, t + 1] = d_non_linear
    my_sol[:, t + 1] = my_sol_new

toc1 = time.time()

print(f"The calculation time for the whole problem was: {toc1-tic1: .2f} s")

arrays = {"base_system": my_sol, "non_linear": non_linear_sol, "n_mesh": n_mesh, "n_steps": n_steps}
output_dir = Path(__file__).resolve().parent / "solutions" / "non_linear_sys"
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = output_dir / f"fdm_results_{timestamp}"

np.savez_compressed(output_path, **arrays)
print(f"Results saved to {output_path}.npz")

# computational time: 107.20 s
# ==================================================================
# Plot
# ==================================================================

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import dill
import plotly.graph_objects as go

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

graphics_dir = Path(__file__).resolve().parent / "graphics" / "non_linear_sys"
graphics_dir.mkdir(parents=True, exist_ok=True)
fig_path = graphics_dir / f"fdm_plot_{timestamp}.pickle"
with open(fig_path, "wb") as f:
    dill.dump(fig, f)
print(f"Interactive figure saved to {fig_path}")

html_stride = 24
frame_steps = list(range(0, n_steps + 1, html_stride))
if frame_steps[-1] != n_steps:
    frame_steps.append(n_steps)

frames = [
    go.Frame(
        data=[go.Scatter(x=r_i, y=my_sol[:, k])],
        name=str(k),
        layout=go.Layout(title=f"t = {time_vec[k]/3600:.0f} h"),
    )
    for k in frame_steps
]

fig_plotly = go.Figure(
    data=[go.Scatter(x=r_i, y=my_sol[:, 0])],
    frames=frames,
    layout=go.Layout(
        xaxis_title="r [m]",
        yaxis_title="Temperature [°C]",
        yaxis_range=[my_sol.min(), my_sol.max()],
        title="t = 0 h",
        sliders=[{
            "currentvalue": {"prefix": "step: "},
            "steps": [
                {
                    "args": [[str(k)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                    "label": f"{time_vec[k]/3600:.0f} h",
                    "method": "animate",
                }
                for k in frame_steps
            ],
        }],
    ),
)

html_path = graphics_dir / f"fdm_plot_{timestamp}.html"
fig_plotly.write_html(html_path)
print(f"HTML interactive plot saved to {html_path}")

plt.show()
