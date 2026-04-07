import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.io import loadmat
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

# ============================================================
# I. Data loading (t-scan: Type1/Type2)
# ============================================================
DATA_DIR = Path("data")  # Directory containing your .mat files

def load_mat(name):
    mat = loadmat(DATA_DIR / f"{name}.mat")
    return {k: v for k, v in mat.items() if not k.startswith("__")}

# -------------------- Type 1 (beta = 0.5) --------------------
Y_expdata_type1      = load_mat("Y_expdata_type1")["Y_expdata_type1"]
X_expdata_type1      = load_mat("X_expdata_type1")["X_expdata_type1"]
Y_expdata_type1_full = load_mat("Y_expdata_type1_full")["Y_expdata_type1_full"]
X_expdata_type1_full = load_mat("X_expdata_type1_full")["X_expdata_type1_full"]
Y_expdata_type11_full = load_mat("Y_expdata_type11_full")["Y_expdata_type11_full"]
X_expdata_type11_full = load_mat("X_expdata_type11_full")["X_expdata_type11_full"]
tt_type1             = load_mat("tt_type1")["tt_type1"]
tt_type11_full       = load_mat("tt_type11_full")["tt_type11_full"]

mat_exp_xy_type11 = load_mat("exp_xy_type11")
real_values_type1     = mat_exp_xy_type11["real_values_type1"]
imag_values_type1     = mat_exp_xy_type11["imag_values_type1"]
real_values_all_type1 = mat_exp_xy_type11["real_values_all_type1"]
imag_values_all_type1 = mat_exp_xy_type11["imag_values_all_type1"]

# -------------------- Type 2 (beta = 0.3) --------------------
Y_expdata_type2       = load_mat("Y_expdata_type2")["Y_expdata_type2"]
X_expdata_type2       = load_mat("X_expdata_type2")["X_expdata_type2"]
Y_expdata_type2_full  = load_mat("Y_expdata_type2_full")["Y_expdata_type2_full"]
X_expdata_type2_full  = load_mat("X_expdata_type2_full")["X_expdata_type2_full"]
Y_expdata_type21_full = load_mat("Y_expdata_type21_full")["Y_expdata_type21_full"]
X_expdata_type21_full = load_mat("X_expdata_type21_full")["X_expdata_type21_full"]

mat_exp_xy_type22 = load_mat("exp_xy_type22")
real_values_type2     = mat_exp_xy_type22["real_values_type2"]
imag_values_type2     = mat_exp_xy_type22["imag_values_type2"]
real_values_all_type2 = mat_exp_xy_type22["real_values_all_type2"]
imag_values_all_type2 = mat_exp_xy_type22["imag_values_all_type2"]

# --- These free-energy results are not actually used in the current version,
# --- but the loading is kept here.
mat_results_free3 = load_mat("results_free3")
data_type2       = mat_results_free3["data"]

mat_results_free5 = load_mat("results_free5")
data_type1       = mat_results_free5["data"]


# ============================================================
# II. Global parameters & style for t-scan
# ============================================================
scale2 = 2/2.54
M, N = real_values_all_type1.shape

BETA_05 = 0.5
BETA_03 = 0.3

# Slightly thicker lines / markers to better match the visual style of Fig. 4
LINE_WIDTH_THEO       = 1.2
LINE_WIDTH_tick       = 1.0
MARKER_SIZE_A         = 3
MARKER_SIZE_ZERO      = 2.5
CAP_SIZE              = 2.0

# Font settings: consistent with your Fig. 4, using Arial + bold labels
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, bm} \boldmath"
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.it'] = 'Computer Modern:italic'
mpl.rcParams['mathtext.bf'] = 'Computer Modern:bold'

plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

FONT_NAME = "Arial"
FONT_SIZE_AXES_LABEL  = 8
FONT_SIZE_TICK_LABEL  = 8
FONT_SIZE_TITLE       = 8
FONT_SIZE_LEGEND      = 8
LETTER_FONT_SIZE      = 9
LINE_WIDTH_ERR= 0.8
LINE_WIDTH_EXP_POINT=0.8

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

# ============================================================
# Unified color palette: align as closely as possible with Fig. 4
# - Theoretical curve: bright blue (#0072B2)
# - Experimental points/errors: orange-red (#D55E00)
# - Background fill: light cyan (#DFF3FF)
# ============================================================
def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)]) / 255.0

COLOR_THEO_BLUE   = hex2rgb("#0072B2")  # Blue curve similar to Fig. 4
COLOR_EXP_ORANGE  = hex2rgb("#D55E00")  # Experimental points in the lower row of Fig. 4
COLOR_SHADE_CYAN  = hex2rgb("#DFF3FF")  # Light cyan fill
COLOR_LOOP_BLUE   = hex2rgb("#005EA8")  # Slightly darker blue for XY trajectories
COLOR_GRAY_ZERO   = np.array([0.4, 0.4, 0.4])

# t-scan section
col_t = {}
col_t["A_th"]     = COLOR_THEO_BLUE       # Theoretical |L| curve
col_t["A_exp"]    = COLOR_EXP_ORANGE      # Marker edge for experimental |L|
col_t["A_shade"]  = COLOR_SHADE_CYAN
col_t["ErrBar_A"] = COLOR_EXP_ORANGE      # Error-bar lines/caps
col_t["X_exp"]    = COLOR_EXP_ORANGE
col_t["XY_th"]    = COLOR_LOOP_BLUE       # Theoretical XY trajectory
col_t["F_exp"]    = COLOR_EXP_ORANGE
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
F_TICKS   = np.array([0.0, 0.5, 1.0, 1.5])

plt.rcParams.update({
    "axes.linewidth": LINE_WIDTH_tick,
})

# ============================================================
# III. Data preprocessing for t-scan
# ============================================================
def cell_to_list_2d(cell_arr):
    M, N = cell_arr.shape
    out = [[None]*N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            out[i][j] = np.array(cell_arr[i, j]).ravel()
    return out

real_all_1 = cell_to_list_2d(real_values_all_type1)
imag_all_1 = cell_to_list_2d(imag_values_all_type1)
real_all_2 = cell_to_list_2d(real_values_all_type2)
imag_all_2 = cell_to_list_2d(imag_values_all_type2)

# -------------------- |L|(t), beta=0.5 --------------------
X_all_s_05 = [[None]*N for _ in range(M)]
Y_all_s_05 = [[None]*N for _ in range(M)]
for i in range(M):
    for j in range(N):
        X_all_s_05[i][j] = np.real(real_all_1[i][j]) * scale2
        Y_all_s_05[i][j] = np.real(imag_all_1[i][j]) * scale2

abs_mean_exp_05 = np.zeros((M, N))
std_abs_05 = np.zeros((M, N))
for i in range(M):
    for j in range(N):
        curX = X_all_s_05[i][j]
        curY = Y_all_s_05[i][j]
        cur_abs = np.sqrt(curX**2 + curY**2)
        abs_mean_exp_05[i, j] = np.mean(cur_abs)
        std_abs_05[i, j] = np.std(cur_abs, ddof=0)

abs_m_05 = abs_mean_exp_05.T.reshape(-1)
lAbs_05 = std_abs_05.T.reshape(-1)
uAbs_05 = std_abs_05.T.reshape(-1)

x_exp_05 = tt_type1.T.reshape(-1)
x_th_05  = tt_type11_full.T.reshape(-1)

X_t_05 = X_expdata_type11_full.T.reshape(-1)
Y_t_05 = Y_expdata_type11_full.T.reshape(-1)
Abs_th_05 = np.sqrt(np.real(X_t_05)**2 + np.real(Y_t_05)**2)

# Riemann-zero X-Y data (beta=0.5)
X_mean_s_05 = np.real(real_values_type1) * scale2
Y_mean_s_05 = np.real(imag_values_type1) * scale2

std_X_05 = np.zeros((M, N))
std_Y_05 = np.zeros((M, N))
for i in range(M):
    for j in range(N):
        x_arr = np.real(real_all_1[i][j]) * scale2
        y_arr = np.real(imag_all_1[i][j]) * scale2
        std_X_05[i, j] = np.std(x_arr, ddof=0)
        std_Y_05[i, j] = np.std(y_arr, ddof=0)

X_theo_full_xy_05 = X_expdata_type1_full
Y_theo_full_xy_05 = Y_expdata_type1_full

# -------------------- |L|(t), beta=0.3 --------------------
X_all_s_03 = [[None]*N for _ in range(M)]
Y_all_s_03 = [[None]*N for _ in range(M)]
for i in range(M):
    for j in range(N):
        X_all_s_03[i][j] = np.real(real_all_2[i][j]) * scale2
        Y_all_s_03[i][j] = np.real(imag_all_2[i][j]) * scale2

abs_mean_exp_03 = np.zeros((M, N))
std_abs_03 = np.zeros((M, N))
for i in range(M):
    for j in range(N):
        curX = X_all_s_03[i][j]
        curY = Y_all_s_03[i][j]
        cur_abs = np.sqrt(curX**2 + curY**2)
        abs_mean_exp_03[i, j] = np.mean(cur_abs)
        std_abs_03[i, j] = np.std(cur_abs, ddof=0)

abs_m_03 = abs_mean_exp_03.T.reshape(-1)
lAbs_03 = std_abs_03.T.reshape(-1)
uAbs_03 = std_abs_03.T.reshape(-1)

x_exp_03 = tt_type1.T.reshape(-1)
x_th_03  = tt_type11_full.T.reshape(-1)

X_t_03 = X_expdata_type21_full.T.reshape(-1)
Y_t_03 = Y_expdata_type21_full.T.reshape(-1)
Abs_th_03 = np.sqrt(np.real(X_t_03)**2 + np.real(Y_t_03)**2)

X_mean_s_03 = np.real(real_values_type2) * scale2
Y_mean_s_03 = np.real(imag_values_type2) * scale2

std_X_03 = np.zeros((M, N))
std_Y_03 = np.zeros((M, N))
for i in range(M):
    for j in range(N):
        x_arr = np.real(real_all_2[i][j]) * scale2
        y_arr = np.real(imag_all_2[i][j]) * scale2
        std_X_03[i, j] = np.std(x_arr, ddof=0)
        std_Y_03[i, j] = np.std(y_arr, ddof=0)

X_theo_full_xy_03 = X_expdata_type2_full
Y_theo_full_xy_03 = Y_expdata_type2_full

# -------------------- Three t-interval subsets at Riemann zeros --------------------
rows_zeros = np.array([0, 2, 4])  # MATLAB [1,3,5] → Python 0,2,4

X_data_03_exp = X_mean_s_03[rows_zeros, :]
X_data_03_th  = np.real(X_theo_full_xy_03[rows_zeros, 200:600+1])
Y_data_03_exp = Y_mean_s_03[rows_zeros, :]
Y_data_03_th  = np.real(Y_theo_full_xy_03[rows_zeros, 200:600+1])

X_data_05_exp = X_mean_s_05[rows_zeros, :]
X_data_05_th  = np.real(X_theo_full_xy_05[rows_zeros, 200:600+1])
Y_data_05_exp = Y_mean_s_05[rows_zeros, :]
Y_data_05_th  = np.real(Y_theo_full_xy_05[rows_zeros, 200:600+1])

std_X_03_sub = std_X_03[rows_zeros, :]
std_Y_03_sub = std_Y_03[rows_zeros, :]
std_X_05_sub = std_X_05[rows_zeros, :]
std_Y_05_sub = std_Y_05[rows_zeros, :]

# ============================================================
# IV. beta-scan data loading and processing (type3)
# ============================================================
mat_type3 = loadmat(DATA_DIR / "type3_0925_8.mat")
real_values_beta = np.array(mat_type3["real_values"]).squeeze()
imag_values_beta = np.array(mat_type3["imag_values"]).squeeze()
real_values_all_raw_beta = mat_type3["real_values_all"]
imag_values_all_raw_beta = mat_type3["imag_values_all"]

mat_Y3 = loadmat(DATA_DIR / "Y_expdata_type3.mat")
mat_X3 = loadmat(DATA_DIR / "X_expdata_type3.mat")
Y_expdata_type3 = np.array(mat_Y3["Y_expdata_type3"]).squeeze()
X_expdata_type3 = np.array(mat_X3["X_expdata_type3"]).squeeze()

mat_free = loadmat(DATA_DIR / "results_freereal_new.mat")
data_beta = np.array(mat_free["real_part"]).squeeze()
data_F_npz = np.load("data/data.npz")
result_d4_beta  = data_F_npz["result_d4"].squeeze()
result_d8_beta  = data_F_npz["result_d8"].squeeze()

# The beta-scan colors also follow the main palette of Fig. 4
col_beta = {
    "X_exp": COLOR_EXP_ORANGE,
    "Y_exp": COLOR_EXP_ORANGE,
    "A_exp": COLOR_EXP_ORANGE,
    "X_th":  COLOR_THEO_BLUE,
    "Y_th":  COLOR_THEO_BLUE,
    "A_th":  COLOR_THEO_BLUE,
    "F_exp_abs": COLOR_EXP_ORANGE,
    "F_d4": hex2rgb("#33A02C"),   # Brighter green (for possible future d=4 curves)
    "F_d6": hex2rgb("#1F78B4"),
    "F_d8": hex2rgb("#00A6D6"),
    "Limit": COLOR_GRAY_ZERO,
    "New_Curve": COLOR_GRAY_ZERO,
}

scale2_beta = 2 / 2.54

X_theo_beta = X_expdata_type3
Y_theo_beta = Y_expdata_type3
abs_theo_beta = np.sqrt(X_theo_beta**2 + Y_theo_beta**2)

x_points = np.arange(1, 19) * 0.05
BETA_VALUES = x_points
N_POINTS = x_points.size

def cell_to_list_of_arrays_beta(var):
    arr = np.array(var)
    if arr.dtype == object:
        flat = arr.flatten(order="F")
        return [np.array(x).astype(float) for x in flat]
    else:
        arr = np.array(arr, dtype=float)
        if arr.ndim == 1:
            return [arr]
        else:
            return [row for row in arr]

real_values_all_list_beta = cell_to_list_of_arrays_beta(real_values_all_raw_beta)
imag_values_all_list_beta = cell_to_list_of_arrays_beta(imag_values_all_raw_beta)

X_mean_exp_beta = real_values_beta * scale2_beta
Y_mean_exp_beta = imag_values_beta * scale2_beta

abs_mean_exp_beta = np.zeros(N_POINTS)
err_X_beta = np.zeros(N_POINTS)
err_Y_beta = np.zeros(N_POINTS)
err_abs_beta = np.zeros(N_POINTS)
F_exp_beta = np.full(N_POINTS, np.nan)
std_F_exp_beta = np.full(N_POINTS, np.nan)

for i in range(N_POINTS):
    X_i = np.array(real_values_all_list_beta[i], dtype=float) * scale2_beta
    Y_i = np.array(imag_values_all_list_beta[i], dtype=float) * scale2_beta
    abs_i = np.sqrt(X_i**2 + Y_i**2)
    abs_mean_exp_beta[i] = abs_i.mean()

    if abs_i.size > 1:
        err_X_beta[i] = X_i.std(ddof=1)
        err_Y_beta[i] = Y_i.std(ddof=1)
        err_abs_beta[i] = abs_i.std(ddof=1)

        valid = abs_i > 0
        if np.any(valid):
            F_samples = -np.log(abs_i[valid]) / 4.0
            F_exp_beta[i] = F_samples.mean()
            std_F_exp_beta[i] = F_samples.std(ddof=1)

lF_exp_beta = std_F_exp_beta
uF_exp_beta = std_F_exp_beta

data_beta = np.array(data_beta, dtype=float)
F_d4_beta = result_d4_beta.astype(float) * data_beta / 4.0
F_d8_beta = result_d8_beta.astype(float) * data_beta / 8.0

# “Spike-like” curve in the thermodynamic limit (for possible later use)
real_part_new = 0.1 + 0.0005 * np.arange(1, 1601)
LN2 = np.log(2.0)
y_main_new = (1 - real_part_new) * LN2
x0 = 0.5
sigma_beta = 0.001
amp = 0.5 * LN2
y_main_new = y_main_new + amp * np.exp(-(real_part_new - x0) ** 2 / (2 * sigma_beta**2))

# ============================================================
# V. 3-column × 2-row layout: A,B,C / D,E,F
# ============================================================
fig = plt.figure(figsize=(7.8, 2.3))
fig.patch.set_facecolor("white")

LEFT   = 0.00
RIGHT  = 0.00
GAP12  = 0.04
GAP23  = 0.04

col_ratio = np.array([1.5, 1.5, 0.75], dtype=float)
col_ratio /= col_ratio.sum()

total_w = 1.0 - LEFT - RIGHT - GAP12 - GAP23
col_w   = total_w * col_ratio

x1 = LEFT
x2 = x1 + col_w[0] + GAP12
x3 = x2 + col_w[1] + GAP23

TOP    = -0.02
BOTTOM = -0.02
V_GAP  = 0.16

avail_h = 1.0 - TOP - BOTTOM - V_GAP
row1_h  = 1.9/3.0 * avail_h
row2_h  = 1.1/3.0 * avail_h

y_bottom = BOTTOM
y_top    = BOTTOM + row2_h + V_GAP

pos_A = [x1, y_top,    col_w[0], row1_h]
pos_B = [x2, y_top,    col_w[1], row1_h]
pos_C = [x3, y_top,    col_w[2], row1_h]

pos_D = [x1, y_bottom, col_w[0], row2_h]
pos_E = [x2, y_bottom, col_w[1], row2_h]
pos_F = [x3, y_bottom, col_w[2], row2_h]

# -------------------- Panel A, B: |L|(t) --------------------
ax_A = fig.add_axes(pos_A)
ax_B = fig.add_axes(pos_B)

for ax in (ax_A, ax_B):
    ax.set_box_aspect(None)
    ax.set_facecolor("white")
    ax.tick_params(labelsize=FONT_SIZE_TICK_LABEL)

# A: beta = 0.3
ax_A.fill_between(
    x_th_03,
    Abs_th_03,
    0.0,
    facecolor=col_t["A_shade"],
    alpha=0.9,
    linewidth=0
)
# Theoretical curve
h_th_A, = ax_A.plot(
    x_th_03, Abs_th_03,
    color=col_t["A_th"],
    linewidth=LINE_WIDTH_THEO
)

# --- Experiment: error bars + markers centered on the points ---

# 1) Draw only error bars (vertical lines + caps), without built-in markers
h_exp_A = ax_A.errorbar(
    x_exp_03, abs_m_03, yerr=[lAbs_03, uAbs_03],
    fmt='o',                          # Draw both points and error bars
    ecolor=col_t["ErrBar_A"],         # Error-bar color
    elinewidth=LINE_WIDTH_ERR,        # Vertical line width
    capsize=CAP_SIZE,                 # Cap length
    capthick=LINE_WIDTH_ERR,          # Cap line width
    markerfacecolor='none',           # Hollow circles
    markeredgecolor=col_t["A_exp"],   # Marker edge color
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

# B: beta = 0.5
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
    fmt='o',                          # Draw points + error bars together
    ecolor=col_t["ErrBar_A"],         # Error-bar color
    elinewidth=LINE_WIDTH_ERR,        # Vertical line width of error bars
    capsize=CAP_SIZE,                 # Cap length
    capthick=LINE_WIDTH_ERR,          # Cap line width
    markerfacecolor='none',           # Hollow circles
    markeredgecolor=col_t["A_exp"],   # Marker edge color
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
ax_B.set_yticklabels(['', ''])  # No numbers, only short tick marks remain

handles = [h_th_A, h_exp_A]
labels  = [r"${\mathcal{L}}_{\mathrm{theo}}$",
           r"$\mathcal{L}_{\mathrm{exp}}$"]

lg_A=ax_A.legend(
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

# Use integer-style formatting on all axes: do not show values like 0.0
for ax in fig.get_axes():
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

# -------------------- Panel D, E: XY zero-point subplots --------------------
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
CAP_SIZE              = 5/3
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
    # ---- D: beta=0.3 ----
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
        fmt='o',                             # Draw circles + error bars
        ecolor=col_t["ErrBar_A"],            # Error-bar color
        elinewidth=LINE_WIDTH_ERR,           # Error-bar line width
        capsize=CAP_SIZE,                    # Cap length
        capthick=LINE_WIDTH_ERR,             # Cap line width
        markerfacecolor='none',              # Hollow circles
        markeredgecolor=col_t["X_exp"],      # Marker edge color
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

    # ---- E: beta=0.5 ----
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
CAP_SIZE  = 2.0

# -------------------- Panel C: |L|(β) --------------------
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

# -------------------- Panel F: (σx, σy) vs beta --------------------
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
# plt.show()