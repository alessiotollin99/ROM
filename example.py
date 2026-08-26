#=================================================================================================
#Module test to try to apply POD
#=================================================================================================

#=================================================================================================
#Base case with scalar
#=================================================================================================

import numpy as np
import matplotlib.pyplot as plt

a = -0.8
b = 1
u_t = lambda t: np.sin(5 * t)

x_0 = int(input("Insert input starting point: "))
timestep = float(input("Please, enter the timestep required: "))
n_steps = int(input("Please, provide the number of steps tou want to solve your problem: "))

x_vec = np.empty(n_steps + 1, dtype = np.float64)
x_vec[0] = x_0
time = np.linspace(0, n_steps*timestep, n_steps+1)


for t in range(n_steps):
    x_vec[t+1] = (x_vec[t] + timestep * b * u_t(time[t+1])) / (1 - a * timestep)


# plot for first graph

fig, ax = plt.subplots(figsize = (4, 3))

ax.plot(time, x_vec, color = "green", linestyle = "--", linewidth = 0.8)
ax.set_xlabel("Time [s]")
ax.set_ylabel("X [m]")

plt.grid(True, linestyle = "--", alpha = 0.3)
plt.tight_layout()
plt.show()

#=================================================================================================
#Advanced case with matrix
#=================================================================================================

N = int(input("Please, enter the number of Nodes you want to simulate now: "))
alpha_N = float(input("Please, eneter the max number of coeffcient alpha: "))
alpha_vec = np.linspace(1, alpha_N, N)
x_vec_N = np.zeros((N, n_steps + 1), dtype = np.float64)

A = np.identity(N, dtype = np.float64) * (1 - a * timestep)

x_vec_N[:, 0] = x_0

for t in range(n_steps):
    b_vec = (x_vec_N[:, t] + timestep * b * u_t(time[t+1])) / alpha_vec
    x_vec_N[:, t+1] = np.linalg.solve(A, b_vec)


# plot for second graph

fig, ax = plt.subplots(figsize = (4, 3))

ax.plot(time, x_vec_N[0], color = "green", marker = "o", linewidth = 0.8, label = "N = 0", markersize = 1)
ax.plot(time, x_vec_N[1], color = "orange", marker = "s", linewidth = 0.8, label = "N = 1", markersize = 1)
ax.plot(time, x_vec_N[2], color = "red", marker = "^", linewidth = 0.8, label = "N = 2", markersize = 1)

ax.set_xlabel("Time [s]")
ax.set_ylabel("X [m]")
ax.legend(loc = "upper left", fontsize = 6)

plt.grid(True, linestyle = "--", alpha = 0.3)
plt.tight_layout()
plt.show()

#=================================================================================================
#ROM usage
#=================================================================================================

# X_vec_N è la matrice degli snapshot qui, ci calcolo sopra SVD


U, sigma, Vt = np.linalg.svd(x_vec_N)

fig, ax = plt.subplots(figsize = (4,3))
energy = np.cumsum(sigma**2) / np.sum(sigma**2)
# plt.plot(energy, marker='o')
plt.semilogy(sigma, marker = "o")
plt.tight_layout()
plt.show()
r = input("Now that yoy were showed the singular valeus you can decide how many orders keep: ")
V = U[:,:r] # 3 x 2

x_vec_reduced = np.zeros((r, n_steps + 1), dtype = np.float64) # 2 x 121

A_r = V.T @ A @ V # 2 x 2
x_r_0 = V.T @ x_vec_N[:, 0] # 2 x 1
x_vec_reduced[:, 0] = x_r_0

for t in range(n_steps):
    x_full_appr = V @ x_vec_reduced[:, t] # 3 x 1
    b_vec = (x_full_appr + timestep * b * u_t(time[t+1])) / alpha_vec # 3 x 1
    b_r = V.T @ b_vec # 2 x 1
    x_vec_reduced[:, t+1] = np.linalg.solve(A_r, b_r)

x_full = V @ x_vec_reduced

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(time, x_vec_N.T, linestyle='-', alpha=0.5, label='full')
ax.plot(time, x_full.T, linestyle='--', label='ROM')
ax.set_xlabel("Time [s]")
ax.set_ylabel("X [m]")
ax.legend()
plt.tight_layout()
plt.show()