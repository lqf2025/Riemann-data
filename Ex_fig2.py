import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
import warnings
from matplotlib.ticker import FormatStrFormatter

warnings.filterwarnings("ignore")

mp.mp.dps = 80
N = 2**12
logN_mp = mp.log(N) / mp.log(2)

nbeta, nt = 250, 250
beta_min, beta_max = 0.01, 1.5
t_min, t_max = 5, 35

beta_grid = np.linspace(beta_min, beta_max, nbeta)
t_grid    = np.linspace(t_min,    t_max,    nt)
B, T = np.meshgrid(beta_grid, t_grid, indexing='xy')

n_list   = [mp.mpf(k) for k in range(1, N+1)]
logn_mp  = [mp.log(n) for n in n_list]
sign_mp  = [mp.mpf(1) if k % 2 == 1 else mp.mpf(-1) for k in range(1, N+1)]

d_dim = float(logN_mp)

logn_arr = np.array([float(x) for x in logn_mp])

F = np.zeros((nt, nbeta), dtype=float)
S = np.zeros((nt, nbeta), dtype=float)

for ib, b_float in enumerate(beta_grid):
    if (ib + 1) % 100 == 0:
        print(f"Progress: {ib+1}/{nbeta}")

    b = mp.mpf(b_float)

    n_pow_minus_b = [mp.power(n, -b) for n in n_list]

    Zb = mp.fsum(n_pow_minus_b)

    sum_ln_n_pow = mp.fsum([logn_mp[k] * n_pow_minus_b[k] for k in range(N)])

    avg_lnn = float(sum_ln_n_pow / Zb)

    coeff_mpf = [sign_mp[k] * n_pow_minus_b[k] for k in range(N)]
    coeff = np.array([complex(c) for c in coeff_mpf])

    phases = np.exp(-1j * t_grid[:, np.newaxis] * logn_arr[np.newaxis, :])

    Q = phases @ coeff

    L = -Q / complex(Zb)
    absL = np.abs(L)
    absL[absL < 1e-30] = 1e-30

    F[:, ib] = -np.log(absL) / d_dim

    weighted_coeff = coeff * logn_arr
    Q_prime = -(phases @ weighted_coeff)

    ratio_Q = Q_prime / Q

    d_ln_absL_d_beta = np.real(ratio_Q) + avg_lnn

    dF_d_beta = -(1.0 / d_dim) * d_ln_absL_d_beta

    S[:, ib] = b_float * dF_d_beta - F[:, ib]

S_clipped = np.clip(S, -3.5, 2.5)

fig = plt.figure(figsize=(22, 15), dpi=300)
ax = fig.add_subplot(111, projection='3d')

import matplotlib.colors as mcolors
import matplotlib.cm as cm

cmap = cm.get_cmap('turbo')

norm = mcolors.Normalize(vmin=-3.5, vmax=2.5)

surf = ax.plot_surface(
    B, T, S_clipped,
    rstride=1, cstride=1,
    cmap=cmap,
    norm=norm,
    linewidth=0,
    antialiased=True,
    alpha=0.98,
    shade=True
)

ax.set_xlim(beta_min, beta_max)
ax.set_xticks(np.arange(0.0, beta_max + 0.01, 0.25))
ax.set_xticklabels([f'{x:.2f}' for x in np.arange(0.0, beta_max + 0.01, 0.25)], fontsize=16)

z_plane_height = S_clipped.max() * 1.1

t_plane = np.linspace(t_min-2, t_max+2, 500)
z_plane = np.linspace(S_clipped.min() * 1.1, z_plane_height, 100)
T_plane, Z_plane = np.meshgrid(t_plane, z_plane)

beta05 = 0.5
X05 = np.full_like(T_plane, beta05)
ax.plot_surface(
    X05, T_plane, Z_plane,
    color='crimson', alpha=0.1, linewidth=0, antialiased=True,
    zorder=10
)

beta10 = 1.0
X10 = np.full_like(T_plane, beta10)
ax.plot_surface(
    X10, T_plane, Z_plane,
    color='dodgerblue', alpha=0.1, linewidth=0, antialiased=True,
    zorder=10
)

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

ax.view_init(elev=15, azim=275)
ax.set_box_aspect([1.8, 2.0, 1.4])
ax.view_init(elev=10, azim=245)

plt.tight_layout()

plt.savefig(f"Ex_fig2.pdf", dpi=600, bbox_inches='tight')
