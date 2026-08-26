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
k = 200 # W / m K
rho = 2700 # kg / m3
cp = 900 # J / kg K

# building matrix
d = np.full(n, ((rho * cp) / (dt) + (2 * k) / dx**2), dtype = np.float64)
d_l = np.full(n-1, -k / dx**2, dtype = np.float64)
d_u = d_l
diagonals = [d, d_l, d_u]

A = diags(diagonals, [0, -1, 1]).toarray()
A[0, 1] *= 2
A[n-1,n-2] *= 2

# my input
omega = 0.001
fg = lambda t: 1e5 * np.sin(omega * t)

x0 = 0 # condizione di partenza
x0_vec = np.full(n, x0, dtype = np.float64)
snap = np.zeros((n, n_steps + 1), dtype = np.float64)
snap[:, 0] = x0_vec

tic1 = time.time()

for t in range(n_steps):
    qgen = fg(time_series[t+1])
    b = (rho*cp/dt) * snap[:, t]
    b[0] += + qgen
    snap[:, t + 1] = np.linalg.solve(A, b)

toc1 = time.time()

print(f"Il tempo di simulazione del FOM è: {toc1-tic1} s")

fig,ax = plt.subplots(figsize = (8, 4))
ax.plot(np.linspace(0, L, n), snap[:, 10], color = "green", label = "10", linewidth = 1.0, linestyle = "--")
ax.plot(np.linspace(0, L, n), snap[:, 500], color = "red", label = "500", linewidth = 1.0, linestyle = "--")
ax.plot(np.linspace(0, L, n), snap[:, 100], color = "orange", label = "100", linewidth = 1.0, linestyle = "--")

ax.legend()
plt.tight_layout()
plt.show()


#==================================================================================
# my reduced order model
#==================================================================================

U, sigma, Vt = np.linalg.svd(snap)

fig, ax = plt.subplots(figsize = (4,3))
plt.semilogy(sigma, marker = "o")
plt.tight_layout()
plt.show()
r = int(input("Now that you were showed the singular values you can decide how many orders to keep: "))
V = U[:,:r]

x_vec_reduced = np.zeros((r, n_steps + 1), dtype = np.float64)

tic2 = time.time()

A_r = V.T @ A @ V
x_r_0 = V.T @ snap[:, 0] 
x_vec_reduced[:, 0] = x_r_0

for t in range(n_steps):
    x_full_appr = V @ x_vec_reduced[:, t] 
    qgen = fg(time_series[t+1])
    b_vec = (rho*cp/dt) * x_full_appr
    b_vec[0] += + qgen
    b_r = V.T @ b_vec 
    x_vec_reduced[:, t+1] = np.linalg.solve(A_r, b_r)

x_full = V @ x_vec_reduced

toc2 = time.time()

print(f"Il tempo di simulazione del ROM è: {toc2-tic2} s")
error_t = np.linalg.norm(snap - x_full, axis=0)
plt.plot(time_series, error_t)
plt.tight_layout()
plt.show()

node = n // 2  # nodo centrale

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(time_series, snap[node, :], label='FOM', linewidth=2)
ax.plot(time_series, x_full[node, :], label='ROM', linestyle='--', linewidth=2)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Temperature [K]")
ax.legend()
plt.tight_layout()
plt.show()


#==================================================================================
# Sweep frequenze
#==================================================================================

omega_list = [0.0001, 0.001, 0.01, 0.1, 1]
fg = lambda t, omega: 1e5 * np.sin(omega * t)

snap_sweep = np.zeros((n, (n_steps + 1) * len(omega_list)), dtype = np.float64)

tic3 = time.time()

for i, omega in enumerate(omega_list):
        snap_loc = np.zeros((n,n_steps +1), dtype = np.float64)
        snap_loc[:, 0] = x0_vec
        for t in range(n_steps):  
                qgen = fg(time_series[t+1], omega)
                b = (rho*cp/dt) * snap_loc[:, t]
                b[0] += qgen
                snap_loc[:, t + 1] = np.linalg.solve(A, b)
        snap_sweep[:, i * (n_steps + 1): (i+1) * (n_steps +1)] = snap_loc
toc3 = time.time()

print(f"Il tempo di simulazione del FOM per lo sweep delle frequenze è: {toc3-tic3} s")

U_sweep, sigma_sweep, Vt_sweep = np.linalg.svd(snap_sweep)

fig, ax = plt.subplots(figsize = (4,3))
plt.semilogy(sigma_sweep, marker = "o")
plt.tight_layout()
plt.show()
r_sweep = int(input("Now that you were showed the singular values you can decide how many orders to keep: "))
V_sweep = U_sweep[:,:r_sweep]

omega_sweep_reduced = np.linspace(0.0001, 1000, 100)

x_vec_reduced_sweep = np.zeros((r_sweep, (n_steps + 1) * len(omega_sweep_reduced)), dtype = np.float64)

tic4 = time.time()

A_r_sweep = V_sweep.T @ A @ V_sweep
x_r_0_sweep = V_sweep.T @ snap_sweep[:, 0]

for i, omega in enumerate(omega_sweep_reduced):
    x_vec_reduced_sweep_loc = np.zeros((r_sweep, n_steps + 1), dtype = np.float64)
    x_vec_reduced_sweep_loc[:, 0] = x_r_0_sweep
    for t in range(n_steps):
        x_full_appr_sweep = V_sweep @ x_vec_reduced_sweep_loc[:, t] 
        qgen = fg(time_series[t+1], omega)
        b_vec = (rho*cp/dt) * x_full_appr_sweep
        b_vec[0] += + qgen
        b_r = V_sweep.T @ b_vec 
        x_vec_reduced_sweep_loc[:, t+1] = np.linalg.solve(A_r_sweep, b_r)
    x_vec_reduced_sweep[:, i * (n_steps + 1) : (i + 1) * (n_steps + 1)] = x_vec_reduced_sweep_loc

x_full_sweep = V_sweep @ x_vec_reduced_sweep

toc4 = time.time()

print(f"Il tempo di simulazione dello sweep tramite il ROM è: {toc4-tic4} s")


node = n // 2
T_amplitude = np.zeros(len(omega_sweep_reduced))

for i, omega in enumerate(omega_sweep_reduced):
    fom_block = snap_sweep[:, i*(n_steps+1):(i+1)*(n_steps+1)]
    T_amplitude[i] = fom_block[node, :].max() - fom_block[node, :].min()

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(omega_sweep_reduced, T_amplitude, marker='o')
ax.set_xlabel("omega")
ax.set_ylabel(f"Ampiezza T al nodo {node} [K]")
ax.set_xscale('log')
plt.tight_layout()
plt.show()