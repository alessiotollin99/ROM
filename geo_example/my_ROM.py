import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from datetime import datetime

# ==================================================================
# Data loading
# ==================================================================
solutions_dir = Path(__file__).resolve().parent / "solutions" / "base_case"
data_path = sorted(solutions_dir.glob("fdm_results_*.npz"))[-1]
data = np.load(data_path)
my_sol = data["base_system"]
A = data["A"]
n_steps = data["n_steps"]
n_mesh = data["n_mesh"]
print(f"Loaded data from {data_path}")

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

U, sigma, Vt = np.linalg.svd(my_sol)

fig, ax = plt.subplots(figsize=(4, 3))
plt.semilogy(sigma, marker="o")
plt.tight_layout()
plt.show()
r = int(input("Now that you were shown the singular values you can decide how many orders to keep: "))
V = U[:, :r]

sol_reduced = np.zeros((r, n_steps + 1), dtype=np.float64)
A_r = V.T @ A @ V
sol_0 = V.T @ my_sol[:, 0]        
sol_reduced[:, 0] = sol_0          

b0 = q * dr / (2 * np.pi * r0 * k) * (1 / dr**2 - 1 / (r_i[0] * 2 * dr))
b_last = 26 * (1 / dr**2 + 1 / (r_i[-1] * 2 * dr))

tic1 = time.time()

for t in range(n_steps):
    sol_full_t = V @ sol_reduced[:, t]
    b = -sol_full_t / (a * dt)
    b[0] += -b0
    b[-1] += -b_last
    b_r = V.T @ b
    sol_reduced[:, t + 1] = np.linalg.solve(A_r, b_r) 

toc1 = time.time()

print(f"The calculation time for the reduced-order model was: {toc1-tic1: .2f} s")

sol_new = V @ sol_reduced

error_t = np.linalg.norm(my_sol - sol_new, axis=0)

# ==================================================================
# Save ROM results
# ==================================================================

rom_arrays = {
    "rom_system": sol_new,
    "sol_reduced": sol_reduced,
    "V": V,
    "A_r": A_r,
    "r": r,
    "n_steps": n_steps,
    "n_mesh": n_mesh,
    "error_t": error_t,
}
output_dir = Path(__file__).resolve().parent / "solutions" / "base_case"
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = output_dir / f"rom_results_{timestamp}"

np.savez_compressed(output_path, **rom_arrays)
print(f"ROM results saved to {output_path}.npz")

# ==================================================================
# Plot
# ==================================================================

from matplotlib.widgets import Slider
import dill
import plotly.graph_objects as go

fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(bottom=0.25)

line1, = ax.plot(r_i, my_sol[:, 0], label='Full')
line2, = ax.plot(r_i, sol_new[:, 0], label='ROM')
ax.set_xlabel('r [m]')
ax.set_ylabel('Temperature [°C]')
ax.set_ylim(my_sol.min(), my_sol.max())
title = ax.set_title('t = 0 h')

# Second y-axis: L2 error at the current step, drawn as a flat reference line
# that moves up/down as the slider scrolls through time.
ax2 = ax.twinx()
ax2.set_ylabel('Error (L2 norm) [°C]')
ax2.set_ylim(0, error_t.max() * 1.1)
error_line, = ax2.plot(r_i, np.full_like(r_i, error_t[0]), '--', color='tab:red', label='Error')

lines = [line1, line2, error_line]
ax.legend(lines, [ln.get_label() for ln in lines], loc='upper right')

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'step', 0, n_steps, valinit=0, valstep=1)

def update(val):
    step = int(slider.val)
    line1.set_ydata(my_sol[:, step])
    line2.set_ydata(sol_new[:, step])
    error_line.set_ydata(np.full_like(r_i, error_t[step]))
    title.set_text(f't = {time_vec[step]/3600:.0f} h')
    fig.canvas.draw_idle()

slider.on_changed(update)

graphics_dir = Path(__file__).resolve().parent / "graphics" / "base_case"
graphics_dir.mkdir(parents=True, exist_ok=True)
fig_path = graphics_dir / f"rom_plot_{timestamp}.pickle"
with open(fig_path, "wb") as f:
    dill.dump(fig, f)
print(f"Interactive figure saved to {fig_path}")

html_stride = 24
frame_steps = list(range(0, n_steps + 1, html_stride))
if frame_steps[-1] != n_steps:
    frame_steps.append(n_steps)

error_line_x = [r_i[0], r_i[-1]]  # flat reference line spanning the r domain

frames = [
    go.Frame(
        data=[go.Scatter(x=r_i, y=my_sol[:, k], name="Full"),
              go.Scatter(x=r_i, y=sol_new[:, k], name="ROM"),
              go.Scatter(x=error_line_x, y=[error_t[k], error_t[k]], name="Error",
                         yaxis="y2", line=dict(dash="dash", color="red"))],
        name=str(k),
        layout=go.Layout(title=f"t = {time_vec[k]/3600:.0f} h"),
    )
    for k in frame_steps
]

fig_plotly = go.Figure(
    data=[go.Scatter(x=r_i, y=my_sol[:, 0], name="Full"),
          go.Scatter(x=r_i, y=sol_new[:, 0], name="ROM"),
          go.Scatter(x=error_line_x, y=[error_t[0], error_t[0]], name="Error",
                     yaxis="y2", line=dict(dash="dash", color="red"))],
    frames=frames,
    layout=go.Layout(
        xaxis_title="r [m]",
        yaxis_title="Temperature [°C]",
        yaxis_range=[my_sol.min(), my_sol.max()],
        yaxis2=dict(title="Error (L2 norm) [°C]", overlaying="y", side="right",
                    range=[0, error_t.max() * 1.1]),
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

html_path = graphics_dir / f"rom_plot_{timestamp}.html"
fig_plotly.write_html(html_path)
print(f"HTML interactive plot saved to {html_path}")

plt.show()

# Computational time: 0.31 s 40 nodes

plt.plot(time_vec, error_t)
plt.tight_layout()
plt.show()
