import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from datetime import datetime
from scipy.sparse import diags

# ==================================================================
# Data loading
# ==================================================================
solutions_dir = Path(__file__).resolve().parent / "solutions" / "non_linear_sys"
data_path = sorted(solutions_dir.glob("fdm_results_*.npz"))[-1]
data = np.load(data_path, allow_pickle=True)
my_sol = data["base_system"]  # shape (n_mesh, n_steps + 1)
non_linear_sol = data["non_linear"]
n_steps = data["n_steps"]
n_mesh = data["n_mesh"]

# ==================================================================
# main data
# ==================================================================
k_dry = 1.83  # ground thermal conductivity [W / m K]
cp_dry = 1600  # ground specific thermal capacity [J / kg K]
rho_dry = 1400  # ground density [kg / m3]

beta = 1e-3  # coefficient to account for non linearity

dt = 3600  # timestep [s]

r0 = 0.015  # borehole radius [m]
r_max = 10  # maximum radius [m]

Tg = 13  # undisturbed ground temeprature [°C]
Tstart = Tg  # staring point [°C]
q = 50  # heat injected [W / m]
L_he = 100  # heat exchanger length [m]

time_vec = np.linspace(0, n_steps * dt, n_steps + 1, dtype=np.float64)
dr = (r_max - r0) / n_mesh
r_i = np.arange(r0 + dr / 2, r_max, dr, dtype=np.float64)

T_rif = Tg
f_k = lambda T: k_dry * (1 + beta * (T - T_rif))  # non linearity function
f_a = lambda k: k / (
    rho_dry * cp_dry
)  # to evaluate thermal diffusivity at each iteration

# ==================================================================
# Matrix construction
# ==================================================================
d_0 = 1 / dr**2 - 1 / (2 * dr * r_i[0])
d_last = -(1 / dr**2 + 1 / (r_i[-1] * 2 * dr))

d_linear = np.full(n_mesh, -2 / dr**2, dtype=np.float64)
d_linear[0] += d_0
d_linear[-1] += d_last
d_u = 1 / dr**2 + 1 / (2 * dr * r_i[:-1])
d_l = 1 / dr**2 - 1 / (2 * dr * r_i[1:])

diagonals_linear = [d_linear, d_l, d_u]
A_static = diags(diagonals_linear, [0, -1, 1]).toarray()


b_last = 2 * Tg * (1 / dr**2 + 1 / (r_i[-1] * 2 * dr))

tic1 = time.time()

# ==================================================================
# DEIM
# ==================================================================

# -------------------------------------------------------------------
# ON NON LINEAR TERM
# -------------------------------------------------------------------
U_deim, sigma_deim, Vt_deim = np.linalg.svd(non_linear_sol)

fig, ax = plt.subplots(figsize=(4, 3))
plt.semilogy(sigma_deim, marker="o")
plt.tight_layout()
plt.show()
m = int(
    input(
        "Now that you were shown the singular values you can decide how many orders 'm' to keep: "
    )
)

points = []
points.append(np.argmax(abs(U_deim[:, 0])))
for i in range(1, m):
    U_sub = U_deim[points, :i]
    b_sub = U_deim[points, i]
    c = np.linalg.solve(U_sub, b_sub)
    approx = U_deim[:, :i] @ c
    residual = U_deim[:, i] - approx
    points.append((np.argmax(abs(residual))))

# -------------------------------------------------------------------
# ON LINEAR TERM
# -------------------------------------------------------------------
U, sigma, Vt = np.linalg.svd(my_sol)

fig, ax = plt.subplots(figsize=(4, 3))
plt.semilogy(sigma, marker="o")
plt.tight_layout()
plt.show()
r = int(
    input(
        "Now that you were shown the singular values you can decide how many orders 'r' to keep: "
    )
)
V = U[:, :r]
U_P = U_deim[points, :m]
W_appr = U_deim[:, :m] @ np.linalg.inv(U_P)
Ar_static = V.T @ A_static @ V
V_points = V[points, :]  # m x r matrix
v0 = V[0, :]

sol_reduced = np.zeros((r, (n_steps + 1)), dtype=np.float64)
sol_0 = V.T @ my_sol[:, 0]
sol_reduced[:, 0] = sol_0

b_last = 2 * Tg * (1 / dr**2 + 1 / (r_i[-1] * 2 * dr))

for t in range(n_steps):
    max_iter = 200
    toll = 1e-3
    iter = 0
    err = np.inf
    my_sol_red_prev = sol_reduced[:, t]
    T_points_prev = V_points @ my_sol_red_prev
    T0_prev = v0 @ my_sol_red_prev
    k_vec_points = f_k(
        T_points_prev
    )  # get the temperature for the series of nodes at that step
    k0 = f_k(T0_prev)

    my_sol_red_prec = my_sol_red_prev.copy()

    while iter < max_iter and err > toll:
        a_vec_points = f_a(k_vec_points)
        d_nl_points = -1 / (a_vec_points * dt)
        d_nl_full = W_appr @ d_nl_points
        A_r = Ar_static + V.T @ (d_nl_full[:, None] * V)
        sol_full_prev = V @ my_sol_red_prev
        b_full = d_nl_full * sol_full_prev
        b0 = q * dr / (2 * np.pi * r0 * k0) * (1 / dr**2 - 1 / (r_i[0] * 2 * dr))
        b_full[0] += -b0
        b_full[-1] += -b_last
        b_r = V.T @ b_full

        my_sol_new = np.linalg.solve(A_r, b_r)
        T_points_new = V_points @ my_sol_new
        T0_new = v0 @ my_sol_new
        k_vec_points = f_k(T_points_new)
        k0 = f_k(T0_new)

        iter += 1
        err = np.linalg.norm(my_sol_new - my_sol_red_prec)

        my_sol_red_prec = my_sol_new

    sol_reduced[:, t + 1] = my_sol_red_prec

toc1 = time.time()

print(f"The calculation time for the reduced-order model was: {toc1-tic1: .2f} s")

sol_new = V @ sol_reduced

error_t = np.linalg.norm(my_sol - sol_new, axis=0)

# ==================================================================
# Plot
# ==================================================================

from matplotlib.widgets import Slider
import dill
import plotly.graph_objects as go

graphics_dir = Path(__file__).resolve().parent / "graphics" / "non_linear_sys"
graphics_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(bottom=0.25)

(line1,) = ax.plot(r_i, my_sol[:, 0], label="Full")
(line2,) = ax.plot(r_i, sol_new[:, 0], label="ROM")
ax.set_xlabel("r [m]")
ax.set_ylabel("Temperature [°C]")
ax.set_ylim(min(my_sol.min(), sol_new.min()), max(my_sol.max(), sol_new.max()))
title = ax.set_title("t = 0 h")

# Second y-axis: L2 error at the current step, drawn as a flat reference line
# that moves up/down as the slider scrolls through time.
ax2 = ax.twinx()
ax2.set_ylabel("Error (L2 norm) [°C]")
ax2.set_ylim(0, error_t.max() * 1.1)
(error_line,) = ax2.plot(
    r_i, np.full_like(r_i, error_t[0]), "--", color="tab:red", label="Error"
)

lines = [line1, line2, error_line]
ax.legend(lines, [ln.get_label() for ln in lines], loc="upper right")

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, "step", 0, n_steps, valinit=0, valstep=1)


def update(val):
    step = int(slider.val)
    line1.set_ydata(my_sol[:, step])
    line2.set_ydata(sol_new[:, step])
    error_line.set_ydata(np.full_like(r_i, error_t[step]))
    title.set_text(f"t = {time_vec[step]/3600:.0f} h")
    fig.canvas.draw_idle()


slider.on_changed(update)

fig_path = graphics_dir / f"rom_plot_{timestamp}.pickle"
with open(fig_path, "wb") as f:
    dill.dump(fig, f)
print(f"Interactive figure saved to {fig_path}")

html_stride = 24  # one frame per day instead of per hour
frame_steps = list(range(0, n_steps + 1, html_stride))
if frame_steps[-1] != n_steps:
    frame_steps.append(n_steps)

error_line_x = [r_i[0], r_i[-1]]  # flat reference line spanning the r domain

frames = [
    go.Frame(
        data=[
            go.Scatter(x=r_i, y=my_sol[:, k], name="Full"),
            go.Scatter(x=r_i, y=sol_new[:, k], name="ROM"),
            go.Scatter(
                x=error_line_x,
                y=[error_t[k], error_t[k]],
                name="Error",
                yaxis="y2",
                line=dict(dash="dash", color="red"),
            ),
        ],
        name=str(k),
        layout=go.Layout(title=f"t = {time_vec[k]/3600:.0f} h"),
    )
    for k in frame_steps
]

fig_plotly = go.Figure(
    data=[
        go.Scatter(x=r_i, y=my_sol[:, 0], name="Full"),
        go.Scatter(x=r_i, y=sol_new[:, 0], name="ROM"),
        go.Scatter(
            x=error_line_x,
            y=[error_t[0], error_t[0]],
            name="Error",
            yaxis="y2",
            line=dict(dash="dash", color="red"),
        ),
    ],
    frames=frames,
    layout=go.Layout(
        xaxis_title="r [m]",
        yaxis_title="Temperature [°C]",
        yaxis_range=[
            min(my_sol.min(), sol_new.min()),
            max(my_sol.max(), sol_new.max()),
        ],
        yaxis2=dict(
            title="Error (L2 norm) [°C]",
            overlaying="y",
            side="right",
            range=[0, error_t.max() * 1.1],
        ),
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

plt.plot(time_vec, error_t)
plt.xlabel("time [s]")
plt.ylabel("Error (L2 norm) [°C]")
plt.tight_layout()
plt.show()
