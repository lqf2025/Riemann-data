import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
import warnings
from matplotlib.ticker import FormatStrFormatter

warnings.filterwarnings("ignore")

# ========================= High-precision settings =========================
mp.mp.dps = 80                     # 80 digits are fully sufficient here and faster than 100
N = 2**12
logN_mp = mp.log(N) / mp.log(2)

# ========================= Ultra-fine grid (core part!) =========================
nbeta, nt = 250, 250              # 450×450 = 202500 points → extremely smooth
beta_min, beta_max = 0.01, 1.5
t_min, t_max = 5, 35

beta_grid = np.linspace(beta_min, beta_max, nbeta)
t_grid    = np.linspace(t_min,    t_max,    nt)
B, T = np.meshgrid(beta_grid, t_grid, indexing='xy')

# ========================= Precompute all high-precision constants (10x speedup) =========================
n_list   = [mp.mpf(k) for k in range(1, N+1)]
logn_mp  = [mp.log(n) for n in n_list]
sign_mp  = [mp.mpf(1) if k % 2 == 1 else mp.mpf(-1) for k in range(1, N+1)]

# Precomputing all n^{-β} is not possible globally because β varies,
# but the loop below uses a partially vectorized strategy.

d_dim = float(logN_mp)

# Convert ln n into a float array in advance for convenient use with NumPy
logn_arr = np.array([float(x) for x in logn_mp])

F = np.zeros((nt, nbeta), dtype=float)  # Free-energy density
S = np.zeros((nt, nbeta), dtype=float)  # Entropy density

for ib, b_float in enumerate(beta_grid):
    if (ib + 1) % 100 == 0:
        print(f"Progress: {ib+1}/{nbeta}")

    b = mp.mpf(b_float)

    # ---------- Z(β) and its derivative ----------
    # n^{-β}
    n_pow_minus_b = [mp.power(n, -b) for n in n_list]

    # Z(β) = Σ n^{-β}
    Zb = mp.fsum(n_pow_minus_b)

    # Σ (ln n) n^{-β}
    sum_ln_n_pow = mp.fsum([logn_mp[k] * n_pow_minus_b[k] for k in range(N)])

    # <ln n>_β = Σ ln n n^{-β} / Σ n^{-β}
    avg_lnn = float(sum_ln_n_pow / Zb)

    # ---------- Q(β,t) and its derivative ----------
    # c_n = (-1)^{n+1} n^{-β}
    coeff_mpf = [sign_mp[k] * n_pow_minus_b[k] for k in range(N)]
    coeff = np.array([complex(c) for c in coeff_mpf])  # (N,)

    # phases(t,n) = exp(-i t log n), shape (nt, N)
    phases = np.exp(-1j * t_grid[:, np.newaxis] * logn_arr[np.newaxis, :])

    # Q(t,β) = Σ c_n e^{-i t log n}
    Q = phases @ coeff                    # (nt,) complex

    # L = -Q/Z
    L = -Q / complex(Zb)
    absL = np.abs(L)
    absL[absL < 1e-30] = 1e-30  # Avoid log(0)

    # Free-energy density F = -1/d ln |L|
    F[:, ib] = -np.log(absL) / d_dim

    # Q'(β,t) = - Σ (ln n) c_n e^{-i t log n}
    weighted_coeff = coeff * logn_arr     # c_n * ln n
    Q_prime = -(phases @ weighted_coeff)  # (nt,) complex

    # Q'/Q; note that numerical instability may appear near points where |Q| is very small,
    # which physically corresponds to divergences in the free energy.
    ratio_Q = Q_prime / Q                 # (nt,) complex

    # ∂β ln|L| = Re(Q'/Q) + <ln n>_β
    d_ln_absL_d_beta = np.real(ratio_Q) + avg_lnn   # (nt,)

    # ∂β F = -1/d * ∂β ln|L|
    dF_d_beta = -(1.0 / d_dim) * d_ln_absL_d_beta

    # Entropy density s(t,β) = β ∂β F - F
    S[:, ib] = b_float * dF_d_beta - F[:, ib]

S_clipped = np.clip(S, -3.5, 2.5)

fig = plt.figure(figsize=(22, 15), dpi=300)
ax = fig.add_subplot(111, projection='3d')

import matplotlib.colors as mcolors
import matplotlib.cm as cm

cmap = cm.get_cmap('turbo')

# Color normalization: explicitly use the range [-3.5, 2.5]
norm = mcolors.Normalize(vmin=-3.5, vmax=2.5)

surf = ax.plot_surface(
    B, T, S_clipped,
    rstride=1, cstride=1,
    cmap=cmap,
    norm=norm,          # Key point: color according to the range of S_clipped
    linewidth=0,
    antialiased=True,
    alpha=0.98,
    shade=True
)

# ==================== 1. Set β-axis ticks with interval 0.25 ====================
ax.set_xlim(beta_min, beta_max)
ax.set_xticks(np.arange(0.0, beta_max + 0.01, 0.25))   # 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5
ax.set_xticklabels([f'{x:.2f}' for x in np.arange(0.0, beta_max + 0.01, 0.25)], fontsize=16)

# ==================== 2. Draw two vertical planes at β=0.5 and β=1.0 ====================
z_plane_height = S_clipped.max() * 1.1

# Create a t-z grid for generating rectangular planes
t_plane = np.linspace(t_min-2, t_max+2, 500)
z_plane = np.linspace(S_clipped.min() * 1.1, z_plane_height, 100)
T_plane, Z_plane = np.meshgrid(t_plane, z_plane)

# Full plane at β=0.5 (semi-transparent red)
beta05 = 0.5
X05 = np.full_like(T_plane, beta05)
ax.plot_surface(
    X05, T_plane, Z_plane,
    color='crimson', alpha=0.1, linewidth=0, antialiased=True,
    zorder=10
)   # Higher zorder helps bring it to the front

# Full plane at β=1.0 (semi-transparent blue)
beta10 = 1.0
X10 = np.full_like(T_plane, beta10)
ax.plot_surface(
    X10, T_plane, Z_plane,
    color='dodgerblue', alpha=0.1, linewidth=0, antialiased=True,
    zorder=10
)

# ==================== Colorbar + labels ====================
cbar = fig.colorbar(surf, shrink=0.6, aspect=20, pad=0.05)
cbar.ax.tick_params(labelsize=18)

ax.set_xlabel(r'$\beta$', fontsize=26, labelpad=20)
ax.set_ylabel(r'$t$',    fontsize=26, labelpad=20)
ax.set_zlabel(r'$S(\beta,t)$', fontsize=26, labelpad=10)
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.zaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.zaxis.set_rotate_label(False)
ax.zaxis.label.set_rotation(90)
ax.tick_params(axis='both', which='major', labelsize=18, pad=2)

ax.view_init(elev=15, azim=275)   # This angle shows both planes clearly
ax.set_box_aspect([1.8, 2.0, 1.4])
ax.view_init(elev=10, azim=245)

plt.tight_layout()

# Save the figure (filename includes N and the marker)
plt.savefig(f"Ex_fig2.pdf", dpi=600, bbox_inches='tight')

# S_mean = S.mean(axis=0)   # Shape: (nbeta,)

# fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=300)

# ax2.plot(beta_grid, S_mean, '-', linewidth=2)

# # Optional: mark vertical lines at β = 0.5 and β = 1.0
# # to correspond to the 3D plot
# ax2.axvline(0.5, color='crimson', linestyle='--', linewidth=1.5, label=r'$\beta=0.5$')
# ax2.axvline(1.0, color='dodgerblue', linestyle='--', linewidth=1.5, label=r'$\beta=1.0$')

# ax2.set_xlabel(r'$\beta$', fontsize=18)
# ax2.set_ylabel(r'$\langle S(\beta,t) \rangle_t$', fontsize=18)
# ax2.tick_params(axis='both', which='major', labelsize=14)

# ax2.legend(frameon=False, fontsize=12)

# plt.tight_layout()
# plt.savefig(f"Extended2.pdf", bbox_inches='tight')