import warnings
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

warnings.filterwarnings("ignore")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)

N = 2**12
d_dim = np.log2(N)

nbeta, nt = 250, 250
beta_min, beta_max = 0.01, 1.5
t_min, t_max = 5, 35

beta_grid = np.linspace(beta_min, beta_max, nbeta)
t_grid = np.linspace(t_min, t_max, nt)
B, T = np.meshgrid(beta_grid, t_grid, indexing="xy")

n_arr = np.arange(1, N + 1, dtype=float)
logn_arr = np.log(n_arr)
sign_arr = np.where(np.arange(1, N + 1) % 2 == 1, 1.0, -1.0)

# This does not depend on beta, so compute it once.
phases = np.exp(-1j * t_grid[:, np.newaxis] * logn_arr[np.newaxis, :])

weights = np.exp(-beta_grid[:, np.newaxis] * logn_arr[np.newaxis, :])
Zb = weights.sum(axis=1)
avg_lnn = (weights * logn_arr[np.newaxis, :]).sum(axis=1) / Zb

coeff = sign_arr[np.newaxis, :] * weights
Q = phases @ coeff.T
Q_safe = np.where(np.abs(Q) < 1e-300, 1e-300 + 0j, Q)

L = -Q / Zb[np.newaxis, :]
absL = np.abs(L)
absL[absL < 1e-30] = 1e-30

F = -np.log(absL) / d_dim

Q_prime = -(phases @ (coeff * logn_arr[np.newaxis, :]).T)
ratio_Q = Q_prime / Q_safe
d_ln_absL_d_beta = np.real(ratio_Q) + avg_lnn[np.newaxis, :]
dF_d_beta = -(1.0 / d_dim) * d_ln_absL_d_beta
S = beta_grid[np.newaxis, :] * dF_d_beta - F

S_clipped = np.clip(S, -3.5, 2.5)

# Manual layout is more reliable than tight_layout for 3D axes.
# The proportions below keep the plot close to the colorbar without crowding labels.
fig = plt.figure(figsize=(10.4, 6.25), dpi=300)
ax = fig.add_axes([0.055, 0.045, 0.735, 0.91], projection="3d")

cmap = cm.get_cmap("turbo")
norm = mcolors.Normalize(vmin=-3.5, vmax=2.5)

surf = ax.plot_surface(
    B,
    T,
    S_clipped,
    rstride=1,
    cstride=1,
    cmap=cmap,
    norm=norm,
    linewidth=0,
    antialiased=True,
    alpha=0.98,
    shade=True,
)

ax.set_xlim(beta_min, beta_max)
xticks = np.arange(0.0, beta_max + 0.01, 0.5)
ax.set_xticks(xticks)
ax.set_xticklabels([f"{x:.1f}" if x else "0" for x in xticks], fontsize=11)
ax.set_yticks([5, 15, 25, 35])
ax.set_zticks([-3, -2, -1, 0, 1, 2])

z_plane_height = S_clipped.max() * 1.1

# Use two large rectangles for the reference planes. This avoids distracting
# translucent mesh artifacts from densely sampled plane surfaces.
T_plane = np.array(
    [[t_min - 1.0, t_max + 1.0], [t_min - 1.0, t_max + 1.0]]
)
Z_plane = np.array(
    [
        [S_clipped.min() * 1.05, S_clipped.min() * 1.05],
        [z_plane_height, z_plane_height],
    ]
)

beta05 = 0.5
X05 = np.full_like(T_plane, beta05)
ax.plot_surface(
    X05,
    T_plane,
    Z_plane,
    color="crimson",
    alpha=0.065,
    linewidth=0,
    antialiased=False,
    shade=False,
    zorder=10,
)

beta10 = 1.0
X10 = np.full_like(T_plane, beta10)
ax.plot_surface(
    X10,
    T_plane,
    Z_plane,
    color="dodgerblue",
    alpha=0.065,
    linewidth=0,
    antialiased=False,
    shade=False,
    zorder=10,
)

# Put the colorbar in a fixed, narrow axis close to the plot instead of letting
# fig.colorbar allocate extra canvas space.
cax = fig.add_axes([0.715, 0.235, 0.018, 0.53])
cbar = fig.colorbar(surf, cax=cax)
cbar.set_ticks([-3, -2, -1, 0, 1, 2])
cbar.ax.tick_params(labelsize=11, width=0.8, length=3, pad=2)
cbar.outline.set_linewidth(0.8)

ax.set_xlabel(r"$\beta$", fontsize=15, labelpad=6)
ax.set_ylabel(r"$t$", fontsize=15, labelpad=6)
ax.set_zlabel(r"$\mathscr{S}(\beta,t)$", fontsize=15, labelpad=3)
ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
ax.zaxis.set_major_formatter(FormatStrFormatter("%g"))
ax.zaxis.set_rotate_label(False)
ax.zaxis.label.set_rotation(90)
ax.tick_params(axis="both", which="major", labelsize=11, pad=0, width=0.8)

for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    axis.pane.set_edgecolor((0.86, 0.86, 0.86, 1.0))
    axis._axinfo["grid"]["color"] = (0.72, 0.72, 0.72, 0.55)
    axis._axinfo["grid"]["linewidth"] = 0.55
    axis._axinfo["axisline"]["linewidth"] = 0.8

ax.set_box_aspect([1.8, 2.0, 1.4])
ax.view_init(elev=10, azim=245)

# Do not call tight_layout for 3D axes; it reintroduces large margins.
output_pdf = Path("Supplementary_Figure_1.pdf")

fig.savefig(
    output_pdf,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0,
)

# Matplotlib's 3D axes can still leave an invisible canvas around the artwork.
# Run pdfcrop from inside the plotting script so the generated PDF is tight.
pdfcrop = shutil.which("pdfcrop")
if pdfcrop is not None:
    cropped_pdf = output_pdf.with_name(f"{output_pdf.stem}_cropped_tmp.pdf")
    subprocess.run(
        [pdfcrop, "--margins", "3 3 3 3", str(output_pdf), str(cropped_pdf)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cropped_pdf.replace(output_pdf)
