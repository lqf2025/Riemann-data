import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

DATA_DIR = Path("data")
EXP_DIR = DATA_DIR / "exp"
THEORY_DIR = DATA_DIR / "theory"

exp_03 = np.load(EXP_DIR / "tscan_beta03_exp.npz")
exp_05 = np.load(EXP_DIR / "tscan_beta05_exp.npz")
exp_beta = np.load(EXP_DIR / "beta_scan_exp.npz")

theo_03 = np.load(THEORY_DIR / "tscan_beta03_theory.npz")
theo_05 = np.load(THEORY_DIR / "tscan_beta05_theory.npz")
theo_beta = np.load(THEORY_DIR / "beta_scan_theory.npz")

M, N = exp_05["x_mean"].shape

BETA_05 = 0.5
BETA_03 = 0.3

LINE_WIDTH_THEO = 1.2
LINE_WIDTH_tick = 1.0
MARKER_SIZE_A = 3
MARKER_SIZE_ZERO = 2.5
CAP_SIZE = 2.0

plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, bm} \boldmath"
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.it'] = 'Computer Modern:italic'
mpl.rcParams['mathtext.bf'] = 'Computer Modern:bold'

plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

FONT_NAME = "Arial"
FONT_SIZE_AXES_LABEL = 8
FONT_SIZE_TICK_LABEL = 8
FONT_SIZE_TITLE = 8
FONT_SIZE_LEGEND = 8
LETTER_FONT_SIZE = 9
LINE_WIDTH_ERR = 0.8
LINE_WIDTH_EXP_POINT = 0.8

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Arial",
    "mathtext.fontset": "stix",
    "axes.linewidth": LINE_WIDTH_tick,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)]) / 255.0

COLOR_THEO_BLUE = hex2rgb("#0072B2")
COLOR_EXP_ORANGE = hex2rgb("#D55E00")
COLOR_SHADE_CYAN = hex2rgb("#DFF3FF")
COLOR_LOOP_BLUE = hex2rgb("#005EA8")
COLOR_GRAY_ZERO = np.array([0.4, 0.4, 0.4])

col_t = {}
col_t["A_th"] = COLOR_THEO_BLUE
col_t["A_exp"] = COLOR_EXP_ORANGE
col_t["A_shade"] = COLOR_SHADE_CYAN
col_t["ErrBar_A"] = COLOR_EXP_ORANGE
col_t["X_exp"] = COLOR_EXP_ORANGE
col_t["XY_th"] = COLOR_LOOP_BLUE
col_t["F_exp"] = COLOR_EXP_ORANGE
col_t["F_ErrBar"] = COLOR_EXP_ORANGE
col_t["F_theo_line"] = COLOR_THEO_BLUE

T_POINTS_F = np.array([
    14.134725142,
    21.022039639,
    25.010857580,
    30.424876126,
    32.935061588
])
ln2 = np.log(2.0)
sigma = 0.025
zeros_list = T_POINTS_F.copy()

Y_TICKS_A = np.array([0, 0.25, 0.5])
X_TICKS_A = np.arange(10, 36, 5)
F_TICKS = np.array([0.0, 0.5, 1.0, 1.5])

plt.rcParams.update({
    "axes.linewidth": LINE_WIDTH_tick,
})

rows_zeros = exp_05["rows_zeros"]

abs_mean_exp_05 = exp_05["abs_mean"]
var_abs_05 = exp_05["abs_var"]

abs_m_05 = abs_mean_exp_05.T.reshape(-1)
lAbs_05 = np.sqrt(var_abs_05.T.reshape(-1))
uAbs_05 = lAbs_05.copy()

x_exp_05 = exp_05["t_flat"]
x_th_05 = theo_05["t_flat"]

X_t_05 = theo_05["curve_x_full"].T.reshape(-1)
Y_t_05 = theo_05["curve_y_full"].T.reshape(-1)
Abs_th_05 = np.sqrt(np.real(X_t_05)**2 + np.real(Y_t_05)**2)

X_mean_s_05 = exp_05["x_mean"]
Y_mean_s_05 = exp_05["y_mean"]
std_X_05 = np.sqrt(exp_05["x_var"])
std_Y_05 = np.sqrt(exp_05["y_var"])

X_theo_full_xy_05 = theo_05["xy_x_full"]
Y_theo_full_xy_05 = theo_05["xy_y_full"]

abs_mean_exp_03 = exp_03["abs_mean"]
var_abs_03 = exp_03["abs_var"]

abs_m_03 = abs_mean_exp_03.T.reshape(-1)
lAbs_03 = np.sqrt(var_abs_03.T.reshape(-1))
uAbs_03 = lAbs_03.copy()

x_exp_03 = exp_03["t_flat"]
x_th_03 = theo_03["t_flat"]

X_t_03 = theo_03["curve_x_full"].T.reshape(-1)
Y_t_03 = theo_03["curve_y_full"].T.reshape(-1)
Abs_th_03 = np.sqrt(np.real(X_t_03)**2 + np.real(Y_t_03)**2)

X_mean_s_03 = exp_03["x_mean"]
Y_mean_s_03 = exp_03["y_mean"]
std_X_03 = np.sqrt(exp_03["x_var"])
std_Y_03 = np.sqrt(exp_03["y_var"])

X_theo_full_xy_03 = theo_03["xy_x_full"]
Y_theo_full_xy_03 = theo_03["xy_y_full"]

X_data_03_exp = X_mean_s_03[rows_zeros, :]
X_data_03_th = np.real(X_theo_full_xy_03[rows_zeros, 200:600+1])
Y_data_03_exp = Y_mean_s_03[rows_zeros, :]
Y_data_03_th = np.real(Y_theo_full_xy_03[rows_zeros, 200:600+1])

X_data_05_exp = X_mean_s_05[rows_zeros, :]
X_data_05_th = np.real(X_theo_full_xy_05[rows_zeros, 200:600+1])
Y_data_05_exp = Y_mean_s_05[rows_zeros, :]
Y_data_05_th = np.real(Y_theo_full_xy_05[rows_zeros, 200:600+1])

std_X_03_sub = std_X_03[rows_zeros, :]
std_Y_03_sub = std_Y_03[rows_zeros, :]
std_X_05_sub = std_X_05[rows_zeros, :]
std_Y_05_sub = std_Y_05[rows_zeros, :]

X_theo_beta = theo_beta["curve_x"]
Y_theo_beta = theo_beta["curve_y"]
abs_theo_beta = np.sqrt(X_theo_beta**2 + Y_theo_beta**2)

BETA_VALUES = exp_beta["beta"]
N_POINTS = BETA_VALUES.size

X_mean_exp_beta = exp_beta["x_mean"]
Y_mean_exp_beta = exp_beta["y_mean"]
abs_mean_exp_beta = exp_beta["abs_mean"]

err_X_beta = np.sqrt(exp_beta["x_var"])
err_Y_beta = np.sqrt(exp_beta["y_var"])
err_abs_beta = np.sqrt(exp_beta["abs_var"])

fig = plt.figure(figsize=(7.8, 2.3))
fig.patch.set_facecolor("white")

LEFT = 0.00
RIGHT = 0.00
GAP12 = 0.04
GAP23 = 0.04

col_ratio = np.array([1.5, 1.5, 0.75], dtype=float)
col_ratio /= col_ratio.sum()

total_w = 1.0 - LEFT - RIGHT - GAP12 - GAP23
col_w = total_w * col_ratio

x1 = LEFT
x2 = x1 + col_w[0] + GAP12
x3 = x2 + col_w[1] + GAP23

TOP = -0.02
BOTTOM = -0.02
V_GAP = 0.16

avail_h = 1.0 - TOP - BOTTOM - V_GAP
row1_h = 1.9/3.0 * avail_h
row2_h = 1.1/3.0 * avail_h

y_bottom = BOTTOM
y_top = BOTTOM + row2_h + V_GAP

pos_A = [x1, y_top, col_w[0], row1_h]
pos_B = [x2, y_top, col_w[1], row1_h]
pos_C = [x3, y_top, col_w[2], row1_h]

pos_D = [x1, y_bottom, col_w[0], row2_h]
pos_E = [x2, y_bottom, col_w[1], row2_h]
pos_F = [x3, y_bottom, col_w[2], row2_h]

ax_A = fig.add_axes(pos_A)
ax_B = fig.add_axes(pos_B)

for ax in (ax_A, ax_B):
    ax.set_box_aspect(None)
    ax.set_facecolor("white")
    ax.tick_params(labelsize=FONT_SIZE_TICK_LABEL)

ax_A.fill_between(
    x_th_03,
    Abs_th_03,
    0.0,
    facecolor=col_t["A_shade"],
    alpha=0.9,
    linewidth=0
)
h_th_A, = ax_A.plot(
    x_th_03, Abs_th_03,
    color=col_t["A_th"],
    linewidth=LINE_WIDTH_THEO
)

h_exp_A = ax_A.errorbar(
    x_exp_03, abs_m_03, yerr=[lAbs_03, uAbs_03],
    fmt='o',
    ecolor=col_t["ErrBar_A"],
    elinewidth=LINE_WIDTH_ERR,
    capsize=CAP_SIZE,
    capthick=LINE_WIDTH_ERR,
    markerfacecolor='none',
    markeredgecolor=col_t["A_exp"],
    markeredgewidth=LINE_WIDTH_EXP_POINT,
    markersize=MARKER_SIZE_A,
)

ax_A.set_ylabel(r"$|\mathcal{L}|$", fontsize=FONT_SIZE_AXES_LABEL, labelpad=1)
ax_A.text(-0.10, 1.1, "a", transform=ax_A.transAxes,
          fontsize=LETTER_FONT_SIZE, fontweight='bold', va='top')
ax_A.text(0.5, 0.9, r"$\beta=0.3$", transform=ax_A.transAxes,
          fontsize=LETTER_FONT_SIZE, fontweight='bold',
          ha="center", va="bottom")
ax_A.text(0.5, -0.2, r"$t$", transform=ax_A.transAxes,
          fontsize=FONT_SIZE_AXES_LABEL, ha='center')

ax_B.fill_between(
    x_th_05,
    Abs_th_05,
    0.0,
    facecolor=col_t["A_shade"],
    alpha=0.9,
    linewidth=0
)
h_th_B, = ax_B.plot(
    x_th_05, Abs_th_05,
    color=col_t["A_th"],
    linewidth=LINE_WIDTH_THEO
)
h_exp_B = ax_B.errorbar(
    x_exp_05, abs_m_05, yerr=[lAbs_05, uAbs_05],
    fmt='o',
    ecolor=col_t["ErrBar_A"],
    elinewidth=LINE_WIDTH_ERR,
    capsize=CAP_SIZE,
    capthick=LINE_WIDTH_ERR,
    markerfacecolor='none',
    markeredgecolor=col_t["A_exp"],
    markeredgewidth=LINE_WIDTH_EXP_POINT,
    markersize=MARKER_SIZE_A,
)
ax_B.set_yticklabels([])
ax_B.text(-0.10, 1.1, "b", transform=ax_B.transAxes,
          fontsize=LETTER_FONT_SIZE, fontweight='bold', va='top')
ax_B.text(0.5, 0.9, r"$\beta=0.5$", transform=ax_B.transAxes,
          fontsize=LETTER_FONT_SIZE, fontweight='bold',
          ha="center", va="bottom")
ax_B.text(0.5, -0.2, r"$t$", transform=ax_B.transAxes,
          fontsize=FONT_SIZE_AXES_LABEL, ha='center')

xmin_A = min(x_th_03.min(), x_th_05.min()) - 0.4
xmax_A = max(x_th_03.max(), x_th_05.max()) + 0.4
for ax in (ax_A, ax_B):
    ax.set_xlim(xmin_A, xmax_A)
    ax.set_xticks([10, 15, 20, 25, 30, 35])
ax_A.set_ylim(0, 0.63)
ax_B.set_ylim(0, 0.63)
ax_A.set_yticks(Y_TICKS_A)
ax_B.set_yticks([0.25, 0.5])
ax_B.set_yticklabels(['', ''])

handles = [h_th_A, h_exp_A]
labels = [r"${\mathcal{L}}_{\mathrm{theo}}$",
          r"$\mathcal{L}_{\mathrm{exp}}$"]

lg_A = ax_A.legend(
    handles, labels,
    loc='upper center',
    bbox_to_anchor=(0.25, 0.85),
    frameon=False,
    ncol=2,
    fontsize=FONT_SIZE_LEGEND,
    columnspacing=0.5
)
lg_A._loc = 9
lg_A._legend_box.align = "center"

for ax in fig.get_axes():
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

N_SUBS = 3
INNER_GAP_X = 0.014
INNER_HEIGHT = 1.0

ax_D_parent = fig.add_axes(pos_D)
ax_D_parent.set_axis_off()
ax_D_parent.text(-0.10, 1.2, "d", transform=ax_D_parent.transAxes,
                 fontsize=LETTER_FONT_SIZE, fontweight='bold',
                 va='top', ha='left')

TOTAL_WIDTH_D = pos_D[2]
INNER_WIDTH_D = (TOTAL_WIDTH_D - (N_SUBS - 1)*INNER_GAP_X) / N_SUBS

ax_E_parent = fig.add_axes(pos_E)
ax_E_parent.set_axis_off()
ax_E_parent.text(-0.10, 1.2, "e", transform=ax_E_parent.transAxes,
                 fontsize=LETTER_FONT_SIZE, fontweight='bold',
                 va='top', ha='left')

TOTAL_WIDTH_E = pos_E[2]
INNER_WIDTH_E = (TOTAL_WIDTH_E - (N_SUBS - 1)*INNER_GAP_X) / N_SUBS
CAP_SIZE = 5/3
ax_D_zeros = []
ax_E_zeros = []

t_labels_pos = np.array([0.4, 0.65, 0.365])
t_labels_text = [
    r"$t \in [12, 16]$",
    r"$t \in [23, 27]$",
    r"$t \in [31, 35]$"
]
F_label_y = np.array([0.04, 0.8, 0.05])

for k in range(N_SUBS):
    sub_start_x_D = pos_D[0] + k * (INNER_WIDTH_D + INNER_GAP_X)
    sub_pos_D = [
        sub_start_x_D,
        pos_D[1] + pos_D[3]*(1 - INNER_HEIGHT)/2,
        INNER_WIDTH_D,
        pos_D[3]*INNER_HEIGHT
    ]
    axd = fig.add_axes(sub_pos_D)
    axd.tick_params(labelsize=FONT_SIZE_TICK_LABEL)
    axd.grid(False)

    axd.plot(
        X_data_03_th[k, :], Y_data_03_th[k, :],
        '-', linewidth=LINE_WIDTH_THEO, color=col_t["XY_th"]
    )
    axd.errorbar(
        X_data_03_exp[k, :], Y_data_03_exp[k, :],
        yerr=std_Y_03_sub[k, :], xerr=std_X_03_sub[k, :],
        fmt='o',
        ecolor=col_t["ErrBar_A"],
        elinewidth=LINE_WIDTH_ERR,
        capsize=CAP_SIZE,
        capthick=LINE_WIDTH_ERR,
        markerfacecolor='none',
        markeredgecolor=col_t["X_exp"],
        markeredgewidth=LINE_WIDTH_ERR,
        markersize=MARKER_SIZE_ZERO,
    )

    axd.axvline(0, linestyle=':', color='k', linewidth=0.5)
    axd.axhline(0, linestyle=':', color='k', linewidth=0.5)

    axd.set_ylim(-0.42, 0.3)
    if k == 1:
        axd.set_xlim(-0.19, 0.45)
    axd.set_xticks([0.0, 0.2, 0.4])

    if k == 1:
        axd.set_xlabel(r"$\langle\sigma_x\rangle$",
                       fontsize=FONT_SIZE_AXES_LABEL, labelpad=0.8)
    else:
        axd.set_xlabel("")
    if k == 0:
        axd.set_ylabel(r"$\langle\sigma_y\rangle$",
                       fontsize=FONT_SIZE_AXES_LABEL, labelpad=0.5)
    else:
        axd.set_yticklabels([])

    axd.text(t_labels_pos[k], F_label_y[k], t_labels_text[k],
             transform=axd.transAxes, fontsize=FONT_SIZE_TICK_LABEL,
             ha='center', va='bottom', color='black')

    ax_D_zeros.append(axd)

    sub_start_x_E = pos_E[0] + k * (INNER_WIDTH_E + INNER_GAP_X)
    sub_pos_E = [
        sub_start_x_E,
        pos_E[1] + pos_E[3]*(1 - INNER_HEIGHT)/2,
        INNER_WIDTH_E,
        pos_E[3]*INNER_HEIGHT
    ]
    axe = fig.add_axes(sub_pos_E)
    axe.tick_params(labelsize=FONT_SIZE_TICK_LABEL)
    axe.grid(False)

    axe.plot(
        X_data_05_th[k, :], Y_data_05_th[k, :],
        '-', linewidth=LINE_WIDTH_THEO, color=col_t["XY_th"]
    )
    axe.axvline(0, linestyle=':', color='k', linewidth=0.5)
    axe.axhline(0, linestyle=':', color='k', linewidth=0.5)

    axe.errorbar(
        X_data_05_exp[k, :], Y_data_05_exp[k, :],
        yerr=std_Y_05_sub[k, :], xerr=std_X_05_sub[k, :],
        fmt='o',
        ecolor=col_t["ErrBar_A"],
        elinewidth=LINE_WIDTH_ERR,
        capsize=CAP_SIZE,
        capthick=LINE_WIDTH_ERR,
        markerfacecolor='none',
        markeredgecolor=col_t["X_exp"],
        markeredgewidth=LINE_WIDTH_ERR,
        markersize=MARKER_SIZE_ZERO,
    )

    axe.set_ylim(-0.42, 0.3)
    if k == 1:
        axe.set_xlim(-0.19, 0.49)
    axe.set_xticks([0.0, 0.2, 0.4])
    axe.set_yticklabels([])
    if k == 1:
        axe.set_xlabel(r"$\langle\sigma_x\rangle$",
                       fontsize=FONT_SIZE_AXES_LABEL, labelpad=0.8)
    else:
        axe.set_xlabel("")

    axe.text(t_labels_pos[k], F_label_y[k], t_labels_text[k],
             transform=axe.transAxes, fontsize=FONT_SIZE_TICK_LABEL,
             ha='center', va='bottom', color='black')

    ax_E_zeros.append(axe)
    for ax in [axd, axe]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(top=False, right=False)
CAP_SIZE = 2.0

ax_C = fig.add_axes(pos_C)
ax_C.set_facecolor("white")
ax_C.grid(False)
for spine in ["top", "right"]:
    ax_C.spines[spine].set_visible(False)
ax_C.tick_params(labelsize=FONT_SIZE_TICK_LABEL, width=0.8)

ax_C.text(
    -0.10, 1.1, "c",
    transform=ax_C.transAxes,
    va="top", ha="left",
    fontweight="bold",
    fontsize=LETTER_FONT_SIZE,
)

idx_beta = slice(1, 18)

ax_C.fill_between(
    BETA_VALUES[idx_beta],
    abs_theo_beta[idx_beta],
    0.0,
    facecolor=col_t["A_shade"],
    alpha=0.9,
    linewidth=0,
)
ax_C.text(
    0.5, 0.9, r"$t=14.134725$", transform=ax_C.transAxes,
    fontsize=LETTER_FONT_SIZE, fontweight='bold',
    ha="center", va="bottom"
)

h_L_th_C, = ax_C.plot(
    BETA_VALUES[idx_beta], abs_theo_beta[idx_beta],
    "-", color=col_t["A_th"], linewidth=LINE_WIDTH_THEO,
)

h_L_exp_C = ax_C.errorbar(
    BETA_VALUES[idx_beta], abs_mean_exp_beta[idx_beta],
    yerr=err_abs_beta[idx_beta],
    fmt='o',
    ecolor=col_t["ErrBar_A"],
    elinewidth=LINE_WIDTH_ERR,
    capsize=CAP_SIZE,
    capthick=LINE_WIDTH_ERR,
    markerfacecolor='none',
    markeredgecolor=col_t["X_exp"],
    markeredgewidth=LINE_WIDTH_EXP_POINT,
    markersize=MARKER_SIZE_A,
)

ax_C.axvline(
    0.5,
    ymin=0, ymax=0.8,
    color=(0.5, 0.5, 0.5),
    linestyle=":",
    linewidth=0.8,
)

ax_C.text(
    0.5, -0.2, r"$\beta$",
    transform=ax_C.transAxes,
    fontsize=FONT_SIZE_AXES_LABEL, fontfamily=FONT_NAME,
    ha="center"
)

ax_C.set_xlim(0, 1)
ax_C.set_xticks([0.1, 0.5, 0.9])
ax_C.set_ylim(0, 0.18)
ax_C.set_yticks([0, 0.08, 0.16])
ax_C.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax_C.yaxis.set_major_formatter(FormatStrFormatter('%g'))

ax_F = fig.add_axes(pos_F)
ax_F.set_facecolor("white")
ax_F.grid(False)
for spine in ["top", "right"]:
    ax_F.spines[spine].set_visible(False)
ax_F.tick_params(labelsize=FONT_SIZE_TICK_LABEL, width=0.8)

ax_F.text(
    -0.10, 1.1, "f",
    transform=ax_F.transAxes,
    va="top", ha="left",
    fontweight="bold",
    fontsize=LETTER_FONT_SIZE,
)

idx_beta = slice(6, 13)

h_xy_th, = ax_F.plot(
    X_theo_beta[idx_beta], Y_theo_beta[idx_beta],
    "-", linewidth=LINE_WIDTH_THEO, color=col_t["XY_th"],
)

h_xy_exp = ax_F.errorbar(
    X_mean_exp_beta[idx_beta], Y_mean_exp_beta[idx_beta],
    xerr=err_X_beta[idx_beta], yerr=err_Y_beta[idx_beta],
    fmt='o',
    ecolor=col_t["ErrBar_A"],
    elinewidth=LINE_WIDTH_ERR,
    capsize=CAP_SIZE,
    capthick=LINE_WIDTH_ERR,
    markerfacecolor='none',
    markeredgecolor=col_t["X_exp"],
    markeredgewidth=LINE_WIDTH_ERR,
    markersize=MARKER_SIZE_ZERO,
)

ax_F.axvline(0, linestyle=":", color="k", linewidth=0.5)
ax_F.axhline(0, linestyle=":", color="k", linewidth=0.5)

ax_F.set_xlabel(r"$\langle\sigma_x\rangle$", fontsize=FONT_SIZE_AXES_LABEL, labelpad=0.8)

x_min = min(X_theo_beta[idx_beta].min(), X_mean_exp_beta[idx_beta].min())
x_max = max(X_theo_beta[idx_beta].max(), X_mean_exp_beta[idx_beta].max())
y_min = min(Y_theo_beta[idx_beta].min(), Y_mean_exp_beta[idx_beta].min())
y_max = max(Y_theo_beta[idx_beta].max(), Y_mean_exp_beta[idx_beta].max())

if y_max > y_min:
    pad_y = 0.05 * (y_max - y_min)
else:
    pad_y = 0.1

ax_F.set_xlim(x_min - 0.01, x_max + 0.01)
ax_F.set_ylim(-0.01, 0.045)
ax_F.set_yticks([0, 0.03])

ax_F.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax_F.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax_F.text(
    0.68, 0.8, r"$\beta \in [0.35,0.65]$",
    transform=ax_F.transAxes,
    fontsize=FONT_SIZE_TICK_LABEL,
    ha='center', va='bottom', color='black'
)

for ax in [ax_A, ax_B]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)
ax_B.tick_params(axis='y', which='both', labelleft=False)

plt.savefig("Fig3.pdf", bbox_inches="tight", pad_inches=0.01)