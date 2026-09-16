import numpy as np
from scipy.sparse import diags
from pathlib import Path
from datetime import datetime
import time

k_dry = 1.83  # ground thermal conductivity [W / m K]
cp_dry = 1600  # ground specific thermal capacity [J / kg K]
rho_dry = 1400  # ground density [kg / m3]

a_dry = k_dry / (rho_dry * cp_dry)

k_w = 0.6  # water thermal conductivity [W / m K]
cp_w = 4186  # water specific thermal capacity [J / kg K]
rho_w = 1000  # water density [kg / m3]

a_w = k_w / (rho_w * cp_w)  # ground thermal diffusivity [m2 / s]

sw_values = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
]  # water content list to use in the alpha formula
f_a = lambda sw: a_dry + sw * (a_w - a_dry)  # alpha formula
f_k = lambda sw: k_dry + sw * (k_w - k_dry)

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
solution_list = []
A_dict = {}  # dictionary to store all the matrixes created

# ==================================================================
# Cycle
# ==================================================================
b_last = 2 * Tg * (1 / dr**2 + 1 / (r_i[-1] * 2 * dr))

tic1 = time.time()

for sw in sw_values:
    a_loc = f_a(sw)
    k_loc = f_k(sw)

    my_sol = np.zeros((n_mesh, n_steps + 1), dtype=np.float64)
    my_sol[:, 0] = x0

    d = np.full(n_mesh, (-1 / (a_loc * dt) - 2 / dr**2), dtype=np.float64)
    d[0] += d_0
    d[-1] += d_last
    diagonals = [d, d_l, d_u]
    A = diags(diagonals, [0, -1, 1]).toarray()
    A_dict[sw] = A

    b0 = q * dr / (2 * np.pi * r0 * k_loc) * (1 / dr**2 - 1 / (r_i[0] * 2 * dr))

    for t in range(n_steps):
        b = np.full(n_mesh, (-my_sol[:, t] / (a_loc * dt)), dtype=np.float64)
        b[0] += -b0
        b[-1] += -b_last
        my_sol[:, t + 1] = np.linalg.solve(A, b)

    solution_list.append(my_sol.copy())

toc1 = time.time()

print(f"The calculation time for the whole problem was: {toc1-tic1: .2f} s")

my_sol = np.hstack(solution_list)

arrays = {"base_system": my_sol, "n_mesh": n_mesh, "n_steps": n_steps, "A_dict": A_dict}
output_dir = Path(__file__).resolve().parent / "solutions" / "variable_props"
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = output_dir / f"fdm_results_{timestamp}"

np.savez_compressed(output_path, **arrays)
print(f"Results saved to {output_path}.npz")
# ==================================================================
# Plot
# ==================================================================

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import dill
import plotly.graph_objects as go

my_sol_3d = my_sol.reshape(n_mesh, len(sw_values), n_steps + 1)
fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(bottom=0.25)

lines = []
for i in range(len(sw_values)):
    (line,) = ax.plot(r_i, my_sol_3d[:, i, 0], label=f"Sw={sw_values[i]}")
    lines.append(line)

ax.legend()

ax.set_xlabel("r [m]")
ax.set_ylabel("Temperature [°C]")
ax.set_ylim(my_sol.min(), my_sol.max())
title = ax.set_title("t = 0 h")

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, "step", 0, n_steps, valinit=0, valstep=1)


def update(val):
    step = int(slider.val)
    for i, ln in enumerate(lines):
        ln.set_ydata(my_sol_3d[:, i, step])
    title.set_text(f"t = {time_vec[step]/3600:.0f} h")
    fig.canvas.draw_idle()


slider.on_changed(update)

graphics_dir = Path(__file__).resolve().parent / "graphics" / "variable_props"
graphics_dir.mkdir(parents=True, exist_ok=True)
fig_path = graphics_dir / f"fdm_plot_{timestamp}.pickle"
with open(fig_path, "wb") as f:
    dill.dump(fig, f)
print(f"Interactive figure saved to {fig_path}")


html_stride = 24  # one frame per day instead of per hour
frame_steps = list(range(0, n_steps + 1, html_stride))
if frame_steps[-1] != n_steps:
    frame_steps.append(n_steps)

frames = [
    go.Frame(
        data=[
            go.Scatter(x=r_i, y=my_sol_3d[:, i, k], name=f"Sw={sw_values[i]}")
            for i in range(len(sw_values))
        ],
        name=str(k),
        layout=go.Layout(title=f"t = {time_vec[k]/3600:.0f} h"),
    )
    for k in frame_steps
]

fig_plotly = go.Figure(
    data=[
        go.Scatter(x=r_i, y=my_sol_3d[:, i, 0], name=f"Sw={sw_values[i]}")
        for i in range(len(sw_values))
    ],
    frames=frames,
    layout=go.Layout(
        xaxis_title="r [m]",
        yaxis_title="Temperature [°C]",
        yaxis_range=[my_sol.min(), my_sol.max()],
        title="t = 0 h",
        sliders=[
            {
                "currentvalue": {"prefix": "step: "},
                "steps": [
                    {
                        "args": [
                            [str(k)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                            },
                        ],
                        "label": f"{time_vec[k]/3600:.0f} h",
                        "method": "animate",
                    }
                    for k in frame_steps
                ],
            }
        ],
    ),
)

html_path = graphics_dir / f"fdm_plot_{timestamp}.html"
fig_plotly.write_html(html_path)
print(f"HTML interactive plot saved to {html_path}")

plt.show()

# Computational time: 1039.94 s
