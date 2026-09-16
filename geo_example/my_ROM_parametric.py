import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from datetime import datetime

# ==================================================================
# Data loading
# ==================================================================
solutions_dir = Path(__file__).resolve().parent / "solutions" / "variable_props"
data_path = sorted(solutions_dir.glob("fdm_results_*.npz"))[-1]
data = np.load(data_path, allow_pickle=True)
my_sol = data["base_system"]  # shape (n_mesh, len(sw_values), n_steps + 1)
A_dict = data["A_dict"].item()
n_steps = data["n_steps"]
n_mesh = data["n_mesh"]

# ==================================================================
# main data
# ==================================================================

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
Tstart = Tg  # starting point [°C]
q = 50  # heat injected [W / m]
L_he = 100  # heat exchanger length [m]

time_vec = np.linspace(0, n_steps * dt, n_steps + 1, dtype=np.float64)
dr = (r_max - r0) / n_mesh

r_i = np.arange(r0 + dr / 2, r_max, dr, dtype=np.float64)

# ==================================================================
# SVD
# ==================================================================

U, sigma, Vt = np.linalg.svd(my_sol)

fig, ax = plt.subplots(figsize=(4, 3))
plt.semilogy(sigma, marker="o")
plt.tight_layout()
plt.show()
r = int(
    input(
        "Now that you were shown the singular values you can decide how many orders to keep: "
    )
)
V = U[:, :r]
sol_reduced = np.zeros((r, (n_steps + 1) * len(sw_values)), dtype=np.float64)
sol_0 = V.T @ my_sol[:, 0]

b_last = 2 * Tg * (1 / dr**2 + 1 / (r_i[-1] * 2 * dr))

tic1 = time.time()

for i, sw in enumerate(sw_values):
    a_loc = f_a(sw)
    k_loc = f_k(sw)

    A_loc = A_dict[sw]
    A_r = V.T @ A_loc @ V

    sol_loc = np.zeros((r, n_steps + 1), dtype=np.float64)
    sol_loc[:, 0] = sol_0
    b0 = q * dr / (2 * np.pi * r0 * k_loc) * (1 / dr**2 - 1 / (r_i[0] * 2 * dr))

    for t in range(n_steps):
        sol_full_t_loc = V @ sol_loc[:, t]
        b = -sol_full_t_loc / (a_loc * dt)
        b[0] += -b0
        b[-1] += -b_last
        b_r = V.T @ b
        sol_loc[:, t + 1] = np.linalg.solve(A_r, b_r)

    sol_reduced[:, i * (n_steps + 1) : (i + 1) * (n_steps + 1)] = sol_loc

toc1 = time.time()

print(f"The calculation time for the reduced-order model was: {toc1-tic1: .2f} s")

sol_new = V @ sol_reduced

# ==================================================================
# Plot
# ==================================================================

from matplotlib.widgets import Slider
import dill
import plotly.graph_objects as go

# my_sol / sol_new are stored as (n_mesh, len(sw_values) * (n_steps + 1)),
# one block of columns per Sw (see how sol_reduced is filled above) -
# reshape to (n_mesh, n_sw, n_steps + 1) to index each Sw separately.
n_sw = len(sw_values)
my_sol_3d = my_sol.reshape(n_mesh, n_sw, n_steps + 1)
sol_new_3d = sol_new.reshape(n_mesh, n_sw, n_steps + 1)

graphics_dir = Path(__file__).resolve().parent / "graphics" / "variable_props"
graphics_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

html_stride = 24  # one frame per day instead of per hour
frame_steps = list(range(0, n_steps + 1, html_stride))
if frame_steps[-1] != n_steps:
    frame_steps.append(n_steps)

# ------------------------------------------------------------------
# Figure 1
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(bottom=0.25)

lines_full, lines_rom = [], []
for i, sw in enumerate(sw_values):
    (lf,) = ax.plot(r_i, my_sol_3d[:, i, 0], label=f"Full Sw={sw}")
    (lr,) = ax.plot(
        r_i, sol_new_3d[:, i, 0], "--", color=lf.get_color(), label=f"ROM Sw={sw}"
    )
    lines_full.append(lf)
    lines_rom.append(lr)

ax.set_xlabel("r [m]")
ax.set_ylabel("Temperature [°C]")
ax.set_ylim(min(my_sol.min(), sol_new.min()), max(my_sol.max(), sol_new.max()))
ax.legend(fontsize=7, ncol=2)
title = ax.set_title("t = 0 h")

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, "step", 0, n_steps, valinit=0, valstep=1)


def update(val):
    step = int(slider.val)
    for i in range(n_sw):
        lines_full[i].set_ydata(my_sol_3d[:, i, step])
        lines_rom[i].set_ydata(sol_new_3d[:, i, step])
    title.set_text(f"t = {time_vec[step]/3600:.0f} h")
    fig.canvas.draw_idle()


slider.on_changed(update)

fig_path = graphics_dir / f"rom_plot_{timestamp}.pickle"
with open(fig_path, "wb") as f:
    dill.dump(fig, f)
print(f"Interactive figure saved to {fig_path}")

frames = [
    go.Frame(
        data=[
            go.Scatter(x=r_i, y=my_sol_3d[:, i, k], name=f"Full Sw={sw_values[i]}")
            for i in range(n_sw)
        ]
        + [
            go.Scatter(
                x=r_i,
                y=sol_new_3d[:, i, k],
                name=f"ROM Sw={sw_values[i]}",
                line=dict(dash="dash"),
            )
            for i in range(n_sw)
        ],
        name=str(k),
        layout=go.Layout(title=f"t = {time_vec[k]/3600:.0f} h"),
    )
    for k in frame_steps
]

fig_plotly = go.Figure(
    data=[
        go.Scatter(x=r_i, y=my_sol_3d[:, i, 0], name=f"Full Sw={sw_values[i]}")
        for i in range(n_sw)
    ]
    + [
        go.Scatter(
            x=r_i,
            y=sol_new_3d[:, i, 0],
            name=f"ROM Sw={sw_values[i]}",
            line=dict(dash="dash"),
        )
        for i in range(n_sw)
    ],
    frames=frames,
    layout=go.Layout(
        xaxis_title="r [m]",
        yaxis_title="Temperature [°C]",
        yaxis_range=[
            min(my_sol.min(), sol_new.min()),
            max(my_sol.max(), sol_new.max()),
        ],
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

html_path = graphics_dir / f"rom_plot_{timestamp}.html"
fig_plotly.write_html(html_path)
print(f"HTML interactive plot saved to {html_path}")

plt.show()

# Computational time: 1.53 s 50 nodes

# ------------------------------------------------------------------
# Figure 2: pointwise error vs r (one curve per Sw), moving through time
# ------------------------------------------------------------------

error_field_3d = np.abs(my_sol_3d - sol_new_3d)  # shape (n_mesh, n_sw, n_steps + 1)

fig_err, ax_err = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(bottom=0.25)

error_lines = []
for i, sw in enumerate(sw_values):
    (le,) = ax_err.plot(r_i, error_field_3d[:, i, 0], label=f"Sw={sw}")
    error_lines.append(le)

ax_err.set_xlabel("r [m]")
ax_err.set_ylabel("Error |Full - ROM| [°C]")
ax_err.set_ylim(0, error_field_3d.max() * 1.1)
ax_err.legend()
title_err = ax_err.set_title("t = 0 h")

ax_slider_err = plt.axes([0.2, 0.1, 0.6, 0.03])
slider_err = Slider(ax_slider_err, "step", 0, n_steps, valinit=0, valstep=1)


def update_error(val):
    step = int(slider_err.val)
    for i in range(n_sw):
        error_lines[i].set_ydata(error_field_3d[:, i, step])
    title_err.set_text(f"t = {time_vec[step]/3600:.0f} h")
    fig_err.canvas.draw_idle()


slider_err.on_changed(update_error)

err_fig_path = graphics_dir / f"rom_error_plot_{timestamp}.pickle"
with open(err_fig_path, "wb") as f:
    dill.dump(fig_err, f)
print(f"Interactive error figure saved to {err_fig_path}")

error_frames = [
    go.Frame(
        data=[
            go.Scatter(x=r_i, y=error_field_3d[:, i, k], name=f"Sw={sw_values[i]}")
            for i in range(n_sw)
        ],
        name=str(k),
        layout=go.Layout(title=f"t = {time_vec[k]/3600:.0f} h"),
    )
    for k in frame_steps
]

fig_err_plotly = go.Figure(
    data=[
        go.Scatter(x=r_i, y=error_field_3d[:, i, 0], name=f"Sw={sw_values[i]}")
        for i in range(n_sw)
    ],
    frames=error_frames,
    layout=go.Layout(
        xaxis_title="r [m]",
        yaxis_title="Error |Full - ROM| [°C]",
        yaxis_range=[0, error_field_3d.max() * 1.1],
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

err_html_path = graphics_dir / f"rom_error_plot_{timestamp}.html"
fig_err_plotly.write_html(err_html_path)
print(f"HTML interactive error plot saved to {err_html_path}")

plt.show()
