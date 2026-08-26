import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
import time

print("\nPrima simulazione di ROM con problema fdm 1D e generazione di calore interna.\n")
L = int(input("Inserire la lunghezza della barra da simulare: "))
n = int(input("Inserire il numero di nodi che si vuole simulare: ")) #2000
dt = int(input("Inserire il timestep da simulare: "))
n_steps = int(input("Inserire il numero di step da simulare: "))

dx = L / n # spessore di una discretizzazione
time_series = np.linspace(0, n_steps * dt, n_steps + 1)

# thermal properties
k_list = [10, 100, 1000] # W / m K
rho = 2700 # kg / m3
cp = 900 # J / kg K

# my input
omega_list = [0.001, 0.01, 0.1]
fg = lambda t, omega: 1e5 * np.sin(omega * t)

x0 = 0 # condizione di partenza
x0_vec = np.full(n, x0, dtype = np.float64)
snap = np.zeros((n, (n_steps + 1) * len(k_list) * len(omega_list)), dtype = np.float64)

#sweep combo
k_matrix, omega_matrix = np.meshgrid(k_list, omega_list)

# building matrix
A_dict = {}
for k in k_list:
    d = np.full(n, ((rho * cp) / (dt) + (2 * k) / dx**2), dtype = np.float64)
    d_l = np.full(n-1, -k / dx**2, dtype = np.float64)
    d_u = d_l
    diagonals = [d, d_l, d_u]

    A_loc = diags(diagonals, [0, -1, 1]).toarray()
    A_loc[0, 1] *= 2
    A_loc[n-1,n-2] *= 2

    A_dict[k] = A_loc

tic1 = time.time()

for i, (k, omega) in enumerate(zip(k_matrix.flatten(), omega_matrix.flatten())):
    k_loc, omega_loc = k, omega
    snap_loc = np.zeros((n, n_steps+1))
    snap_loc[:, 0] = x0_vec

    for t in range(n_steps):
        q_gen = fg(time_series[t+1], omega_loc)
        b_vec = (rho*cp/dt) * snap_loc[:, t]
        b_vec[0] += q_gen
        snap_loc[:, t + 1] = np.linalg.solve(A_dict[k_loc], b_vec)

    snap[:, i * (n_steps + 1): (i+1) * (n_steps +1)] = snap_loc

toc1 = time.time()

print(f"Computational time for FOM with parametric sweep was: {toc1 - tic1} s")

U, sigma, Vt = np.linalg.svd(snap)

fig, ax = plt.subplots(figsize = (4,3))
plt.semilogy(sigma, marker = "o")
plt.tight_layout()
plt.show()
r = int(input("Scegli l'ordine r del tuo nuovo modello ridotto: "))
V = U[:,:r]
x_vec_r = np.zeros((r, (n_steps+1) * len(k_list) * len(omega_list)))

tic2 = time.time()

xr_0 = V.T @ snap[:, 0]

Ar_dict = {}

for k in k_list:
    Ar_dict[k] = V.T @ A_dict[k] @ V

for i, (k, omega) in enumerate(zip(k_matrix.flatten(), omega_matrix.flatten())):
    k_loc, omega_loc = k, omega
    xr_loc = np.zeros((r, n_steps + 1), dtype = np.float64)
    xr_loc[:, 0] = xr_0

    for t in range(n_steps):
        x_full = V @ xr_loc[:, t]
        q_gen = fg(time_series[t+1], omega_loc)
        b_vec = (rho*cp/dt) * x_full
        b_vec[0] += q_gen
        b_r = V.T @ b_vec
        xr_loc[:, t+1] = np.linalg.solve(Ar_dict[k_loc], b_r)

    x_vec_r[:, i * (n_steps + 1) : (i + 1) * (n_steps + 1)] = xr_loc

x_full_sweep = V @ x_vec_r

toc2 = time.time()

print(f"Computational time for ROM with parametric sweep was: {toc2-tic2} s")

snap_rep = np.reshape(snap, (n, len(omega_list), len(k_list), n_steps + 1))
x_full_rep = np.reshape(x_full_sweep, (n, len(omega_list), len(k_list), n_steps + 1))
diff = snap_rep - x_full_rep
error_grid = np.linalg.norm(diff, axis=(0,3))

fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(error_grid.T, aspect='auto', origin='lower',
               extent=[min(omega_list), max(omega_list), min(k_list), max(k_list)])
ax.set_xlabel("omega")
ax.set_ylabel("k")
plt.colorbar(im, label="Errore relativo ROM vs FOM")
plt.tight_layout()
plt.show()

