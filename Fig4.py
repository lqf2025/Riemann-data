import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import mpmath as mp
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar
def solid_arrow(ax, start, end, head_length=0.2, head_width=0.01,
                color='#555555', lw=1.0):
    start = np.array(start, dtype=float)
    end   = np.array(end,   dtype=float)

    v = end - start
    L = np.linalg.norm(v)
    if L == 0:
        return
    v_hat = v / L

    z_axis = np.array([0.0, 0.0, 1.0])

    if np.linalg.norm(np.cross(v_hat, z_axis)) < 1e-8:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = z_axis
    n = ref - np.dot(ref, v_hat) * v_hat
    n /= np.linalg.norm(n)

    ax.plot(
        [start[0], end[0]*0.98+start[0]*0.02],
        [start[1], end[1]*0.98+start[1]*0.02],
        [start[2], end[2]*0.98+start[2]*0.02],
        color=color,
        lw=lw,
    )

    tip = end
    base1 = tip - head_length * v_hat + head_width * n
    base2 = tip - head_length * v_hat - head_width * n

    ax.plot(
        [tip[0], base1[0]],
        [tip[1], base1[1]],
        [tip[2], base1[2]],
        color=color,
        lw=lw,
    )
    ax.plot(
        [tip[0], base2[0]],
        [tip[1], base2[1]],
        [tip[2], base2[2]],
        color=color,
        lw=lw,
    )
def add_missing_axes_edges(ax, color='#555555', lw=0.8):
    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()

    ax.plot([x1, x1], [5, 5], [z0, z1],
            color=color, lw=lw)
mp.mp.dps = 80

NBETA, NT = 500, 500
BETA_MIN, BETA_MAX = 0.01, 1.5
T_MIN, T_MAX       = 5,    35

beta_grid = np.linspace(BETA_MIN, BETA_MAX, NBETA)
t_grid    = np.linspace(T_MIN,    T_MAX,    NT)
B_mesh, T_mesh = np.meshgrid(beta_grid, t_grid, indexing='xy')

MAX_F      = 1.5
VMIN_F     = 0.0
VMAX_F     = MAX_F
NORM_F     = mcolors.Normalize(vmin=VMIN_F, vmax=VMAX_F)
CMAP_F     = cm.get_cmap('turbo')

DATA_DIR = Path("data")
EXP_DIR = DATA_DIR / "exp"
THEORY_DIR = DATA_DIR / "theory"

exp_03 = np.load(EXP_DIR / "tscan_beta03_exp.npz")
exp_05 = np.load(EXP_DIR / "tscan_beta05_exp.npz")
exp_beta = np.load(EXP_DIR / "beta_scan_exp.npz")

theo_03 = np.load(THEORY_DIR / "tscan_beta03_theory.npz")
theo_05 = np.load(THEORY_DIR / "tscan_beta05_theory.npz")
theo_beta = np.load(THEORY_DIR / "beta_scan_theory.npz")
free_axes = np.load(THEORY_DIR / "free_energy_axes_theory.npz")

ln2 = np.log(2.0)
sigma = 0.025
T_POINTS_F = np.array([
    14.134725142,
    21.022039639,
    25.010857580,
    30.424876126,
    32.935061588
])
zeros_list = T_POINTS_F.copy()

col = {
    "A_exp":    np.array([212, 86, 46]) / 255.0,
    "A_th":     np.array([68, 133, 199]) / 255.0,
    "A_shade":  np.array([0.85, 0.9, 0.98]),
    "ErrBar_A": np.array([229, 127, 102]) / 255.0,

    "X_exp":    np.array([212, 86, 46]) / 255.0,
    "XY_th":    np.array([68, 133, 199]) / 255.0,
}

curve_colors = {
    "limit": CMAP_F(NORM_F(MAX_F)),
    "d8":    CMAP_F(NORM_F(MAX_F)),
    "d4":    CMAP_F(NORM_F(MAX_F)),
    "exp":   CMAP_F(NORM_F(MAX_F)),
}

LINE_WIDTH_THEO      = 0.7
LINE_WIDTH_EXP_POINT = 0.7
LINE_WIDTH_ERR       = 0.7
LINE_WIDTH_tick      = 0.7
LINE_WIDTH_D         = 0.7

MARKER_SIZE_A   = 3
CAP_SIZE        = 2

FONT_NAME            = "Times New Roman"
FONT_SIZE_AXES_LABEL = 8
FONT_SIZE_TICK_LABEL = 8
FONT_SIZE_LEGEND     = 8

FONT_3D_LABEL        = 8
FONT_3D_TICK         = 8
PANEL_LABEL_SIZE     = 10

plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, bm} \boldmath"
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.it'] = 'Computer Modern:italic'
mpl.rcParams['mathtext.bf'] = 'Computer Modern:bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Arial",
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

def tscan_F_from_npz(exp_npz):
    abs_mean = exp_npz["abs_mean"]
    abs_var = exp_npz["abs_var"]
    F_ref = (-np.log(abs_mean) / 4.0).T.reshape(-1)
    F_var_ref = (abs_var / (16.0 * abs_mean**2)).T.reshape(-1)

    if "F_mean" in exp_npz and np.allclose(exp_npz["F_mean"], F_ref):
        F_mean = exp_npz["F_mean"]
    else:
        print("warning: t-scan F_mean in NPZ is inconsistent with -log(abs_mean)/4; using recomputed values")
        F_mean = F_ref
    if "F_var" in exp_npz and np.allclose(exp_npz["F_var"], F_var_ref):
        F_var = exp_npz["F_var"]
    else:
        print("warning: t-scan F_var in NPZ is inconsistent with delta-method value; using recomputed values")
        F_var = F_var_ref
    return F_mean, F_var
def cell_to_list_2d(cell_arr):
    M, N = cell_arr.shape
    out = [[None]*N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            out[i][j] = np.array(cell_arr[i, j]).ravel()
    return out
def cell_to_list_of_arrays(var):
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
def add_colored_line(ax, x, y,
                     linestyle='-',
                     linewidth=1.0,
                     cmap=CMAP_F,
                     n_segments=1000):
    VMIN_F, VMAX_F = 0.0, 1.5
    norm = mcolors.Normalize(vmin=VMIN_F, vmax=VMAX_F)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return None
    idx_sort = np.argsort(x)
    x = x[idx_sort]
    y = y[idx_sort]

    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        return None
    edges = np.linspace(x_min, x_max, n_segments + 1)

    if linestyle == '-':
        points   = np.column_stack([x, y])
        segments = np.stack([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidth=linewidth,
        )

        lc.set_array(y[:-1])

        ax.add_collection(lc)
        ax.autoscale_view()
        return lc
    elif linestyle == '--':
        period, on_len = 20, 10
    elif linestyle == '-.':
        period, on_len = 12, 4
    last_line = None
    for i in range(n_segments):
        if (i % period) >= on_len:
            continue
        seg_mask = (x >= edges[i]) & (x <= edges[i+1])
        x_seg = x[seg_mask]
        y_seg = y[seg_mask]

        if x_seg.size < 2:
            continue
        y_mid = float(np.mean(y_seg))
        color = cmap(norm(y_mid))

        line, = ax.plot(
            x_seg, y_seg,
            color=color,
            linewidth=linewidth,
        )
        last_line = line
    ax.autoscale_view()

    return last_line
class HandlerGradDashLine(HandlerLine2D):
    def __init__(self, cmap, norm, nseg=1000, **kwargs):
        self.cmap = cmap
        self.norm = norm
        self.nseg = nseg
        super().__init__(**kwargs)
    def create_artists(
        self, legend, orig_handle,
        x0, y0, width, height, fontsize, trans
    ):
        scale = 1.8
        x_start = x0
        x_end   = x0 + width * scale
        y_mid   = y0 + 0.5 * height

        xs = np.linspace(x_start, x_end, self.nseg + 1)

        ls = orig_handle.get_linestyle()
        if ls in ("solid", "-"):
            period, on_len = 1, 1
        elif ls in ("--", "dashed"):
            period, on_len = 20, 10
        elif ls in ("-.", "dashdot"):
            period, on_len = 12, 3
        artists = []
        for i in range(self.nseg):
            if (i % period) >= on_len:
                continue
            x_seg = [xs[i], xs[i+1]]
            y_seg = [y_mid, y_mid]

            v = MAX_F * (i + 0.5) / self.nseg
            color = self.cmap(self.norm(v))

            seg = plt.Line2D(
                x_seg, y_seg,
                linestyle='-',
                linewidth=orig_handle.get_linewidth(),
                color=color,
                transform=trans,
            )
            artists.append(seg)
        return artists
def make_grad_legend_handle(linestyle, label):
    line = plt.Line2D(
        [], [],
        linestyle=linestyle,
        linewidth=LINE_WIDTH_D,
        color='black',
        label=label,
    )

    handler = HandlerGradDashLine(
        cmap=CMAP_F,
        norm=NORM_F,
        nseg=120,
    )
    return line, handler
def make_legend_colored_line(y_value, linestyle='-', linewidth=3, cmap=CMAP_F, norm=NORM_F):
    x_fake = np.linspace(0, 1, 20)
    y_fake = np.full_like(x_fake, y_value)

    points = np.array([x_fake, y_fake]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm,
                        linewidth=linewidth, linestyle=linestyle)
    lc.set_array(y_fake[:-1])
    return lc
def plot_L2_surface_on_axis(ax, N):
    print(f"Starting calculation of F(beta, t) for N = {N}...")
    logN_mp = mp.log(N) / mp.log(2)

    n_list  = [mp.mpf(k) for k in range(1, N+1)]
    logn_mp = [mp.log(n) for n in n_list]
    sign_mp = [mp.mpf(1) if k % 2 == 1 else mp.mpf(-1) for k in range(1, N+1)]

    logn_float = np.array([float(x) for x in logn_mp])
    F = np.zeros((NT, NBETA))

    for ib, b_float in enumerate(beta_grid):
        if (ib+1) % 100 == 0:
            print(f"N={N} progress: {ib+1}/{NBETA}")
        b = mp.mpf(b_float)
        n_pow_minus_b = [mp.power(n, -b) for n in n_list]
        Zb = mp.fsum(n_pow_minus_b)
        coeff = [sign_mp[k] * n_pow_minus_b[k] for k in range(N)]

        phases   = np.exp(-1j * t_grid[:, np.newaxis] * logn_float[np.newaxis, :])
        coeff_c  = np.array([complex(c) for c in coeff])
        s_complex = phases @ coeff_c

        L   = -s_complex / complex(Zb)
        val = np.abs(L)
        val[val < 1e-30] = 1e-30

        with mp.workdps(30):
            F[:, ib] = -np.log(val) / float(logN_mp)
    print(f"N={N} completed! F range: {F.min():.4f} ~ {F.max():.4f}")

    F_clipped = np.clip(F, 0.0, MAX_F)

    surf = ax.plot_surface(
        B_mesh, T_mesh, F_clipped,
        rstride=1, cstride=1,
        cmap=CMAP_F,
        norm=NORM_F,
        linewidth=0,
        antialiased=True,
        alpha=1.0,
        shade=False,
    )

    black_lw = 3.3
    core_lw  = 2.0

    def plot_beta_ridge(beta_val,color):
        idx = np.argmin(np.abs(beta_grid - beta_val))
        z = F_clipped[:, idx]
        x = np.full_like(t_grid, beta_val)

        ax.plot(x, t_grid, z, color=color, linewidth=black_lw, zorder=20)
        for i in range(len(t_grid) - 1):
            c = CMAP_F(NORM_F(z[i]))
            ax.plot(
                [beta_val, beta_val],
                t_grid[i:i+2],
                z[i:i+2],
                color=c,
                linewidth=core_lw,
                zorder=21
            )
    def plot_t_ridge(t_val,color):
        idx = np.argmin(np.abs(t_grid - t_val))
        z = F_clipped[idx, :]
        y = np.full_like(beta_grid, t_val)

        ax.plot(beta_grid, y, z, color=color, linewidth=black_lw, zorder=20)
        for i in range(len(beta_grid) - 1):
            c = CMAP_F(NORM_F(z[i]))
            ax.plot(
                beta_grid[i:i+2],
                [t_val, t_val],
                z[i:i+2],
                color=c,
                linewidth=core_lw,
                zorder=21
            )
    plot_beta_ridge(0.3,'white')
    plot_beta_ridge(0.5,'#FF4B00')
    plot_t_ridge(14.134725,'black')

    ax.set_xlim(BETA_MIN, BETA_MAX)
    ax.set_xticks(np.arange(0.0, BETA_MAX + 0.01, 0.25))
    ax.set_xticklabels([f"{x:.2f}" for x in np.arange(0.0, BETA_MAX + 0.01, 0.25)],
                       fontsize=FONT_3D_TICK)
    ax.set_xlabel(r'$\beta$', fontsize=FONT_3D_LABEL, labelpad=-10)
    ax.set_ylabel(r'$t$',    fontsize=FONT_3D_LABEL, labelpad=-3)

    ax.tick_params(axis='both', which='major', labelsize=FONT_3D_TICK, pad=-3)
    ax.xaxis._axinfo['tick']['pad'] = -10
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.label.set_rotation(90)
    shift = -0.2
    for tick in ax.yaxis.get_major_ticks():
        lab = tick.label1
        x, y = lab.get_position()
        lab.set_position((x + shift, y))
    for tick in ax.xaxis.get_major_ticks():
        lab = tick.label1
        x, y = lab.get_position()
        lab.set_position((x, y-0.2))
    ax.view_init(elev=35, azim=245)
    ax.set_box_aspect([2, 2.35, 1.48])

    pane_color = (1, 1, 1, 1.0)

    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)
    grid_color = (0.8, 0.8, 0.8, 0.5)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["color"] = grid_color
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis._axinfo['tick']['inward_factor']  = 0.2
        axis._axinfo['tick']['outward_factor'] = 0.0
    return F_clipped, surf
def compute_F1_gauss(beta_grid, t_grid):
    beta_grid = np.asarray(beta_grid, dtype=float)
    t_grid    = np.asarray(t_grid, dtype=float)

    B, T = np.meshgrid(beta_grid, t_grid, indexing="xy")
    F = np.zeros_like(B, dtype=float)

    mask_mid = (B > 0.0) & (B < 1.0)
    F[mask_mid] = (1.0 - B[mask_mid]) * ln2

    A = ln2 / 2.0

    list1 = [2*np.pi/np.log(2)*1,
             2*np.pi/np.log(2)*2,
             2*np.pi/np.log(2)*3]
    list2 = [14.134725142,
             21.022039639,
             25.010857580,
             30.424876126,
             32.935061588]
    for t0 in list2:
        F += A * np.exp(-((B-0.5)**2*5 + (T-t0)**2/5) / 2e-3)
    for t0 in list1:
        F += 2*A * np.exp(-((B-1.0)**2*5 + (T-t0)**2/5) / 2e-3)
    return F
def plot_inf_L2_panel_c(ax, beta_grid, t_grid, F):
    beta_grid = np.asarray(beta_grid, dtype=float)
    t_grid    = np.asarray(t_grid, dtype=float)
    B, T = np.meshgrid(beta_grid, t_grid, indexing='xy')

    F_clipped = np.clip(F, 0.0, MAX_F)

    surf = ax.plot_surface(
        B, T, F_clipped,
        rstride=1, cstride=1,
        cmap=CMAP_F,
        norm=NORM_F,
        linewidth=0,
        antialiased=True,
        alpha=1.0,
        shade=False,
    )
    beta0 = 0.5
    t_plane = np.linspace(T_MIN, T_MAX, 80)
    z_plane = np.linspace(0.0, np.log(2), 2)
    T_plane, Z_plane = np.meshgrid(t_plane, z_plane, indexing='xy')
    B_plane = np.full_like(T_plane, beta0)

    ax.plot_surface(
    B_plane, T_plane, Z_plane,
    color='#FFB703',
    alpha=0.9,
    linewidth=0,
    antialiased=False,
    shade=False,
    )

    ax.text(
        1, 34, 0.95*np.log(2),
        "RH",
        fontsize=8,
        fontweight='bold',
        ha='center',
        va='center'
    )
    start = np.array([0.95, 33.0, 0.88 * np.log(2.0)])
    end   = np.array([0.6, 30, 0.9 * np.log(2.0)])

    solid_arrow(ax, start, end)
    black_lw = 3.0
    core_lw  = 2.0

    def plot_beta_ridge(beta_val,color):
        idx = np.argmin(np.abs(beta_grid - beta_val))
        z = F_clipped[:, idx]
        x = np.full_like(t_grid, beta_val)

        ax.plot(x, t_grid, z, color=color, linewidth=black_lw, zorder=20)
        for i in range(len(t_grid) - 1):
            c = CMAP_F(NORM_F(z[i]))
            ax.plot(
                [beta_val, beta_val],
                t_grid[i:i+2],
                z[i:i+2],
                color=c,
                linewidth=core_lw,
                zorder=21
            )
    def plot_t_ridge(t_val,color):
        idx = np.argmin(np.abs(t_grid - t_val))
        z = F_clipped[idx, :]
        y = np.full_like(beta_grid, t_val)

        ax.plot(beta_grid, y, z, color=color, linewidth=black_lw, zorder=20)
        for i in range(len(beta_grid) - 1):
            c = CMAP_F(NORM_F(z[i]))
            ax.plot(
                beta_grid[i:i+2],
                [t_val, t_val],
                z[i:i+2],
                color=c,
                linewidth=core_lw,
                zorder=21
            )
    plot_beta_ridge(0.3,'white')
    plot_beta_ridge(0.5,'#FF4B00')
    plot_t_ridge(14.134725,'black')

    ax.set_xlim(BETA_MIN, BETA_MAX)
    ax.set_xticks(np.arange(0.0, BETA_MAX + 0.01, 0.25))
    ax.set_xticklabels([f"{x:.2f}" for x in np.arange(0.0, BETA_MAX + 0.01, 0.25)],
                       fontsize=FONT_3D_TICK)
    ax.set_xlabel(r'$\beta$', fontsize=FONT_3D_LABEL, labelpad=-10)
    ax.set_ylabel(r'$t$',    fontsize=FONT_3D_LABEL, labelpad=-3)
    ax.tick_params(axis='both', which='major', labelsize=FONT_3D_TICK, pad=-3)
    ax.xaxis._axinfo['tick']['pad'] = -10
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.label.set_rotation(90)

    ax.view_init(elev=35, azim=245)
    ax.set_box_aspect([2, 2.35, 1.48])

    pane_color = (1, 1, 1, 1.0)

    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)
    grid_color = (0.8, 0.8, 0.8, 0.5)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["color"] = grid_color
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis._axinfo['tick']['inward_factor']  = 0.2
        axis._axinfo['tick']['outward_factor'] = 0.0
    return F_clipped, surf
Y_expdata_type11_full = theo_05["curve_y_full"]
X_expdata_type11_full = theo_05["curve_x_full"]
tt_type11_full = theo_05["t_grid"]

data_type2 = free_axes["t_beta03"]
data_type1 = free_axes["t_beta05"]

data_npz = np.load(DATA_DIR / "data2.npz")
results_d4_type2 = data_npz['results_d4_type2']
results_d8_type2 = data_npz['results_d8_type2']
results_d4_type1 = data_npz['results_d4_type1']
results_d8_type1 = data_npz['results_d8_type1']

BETA_05 = 0.5
BETA_03 = 0.3

x_exp_05 = exp_05["t_flat"]
x_exp_03 = exp_03["t_flat"]

x_th_05  = tt_type11_full.T.reshape(-1)
x_th_03  = tt_type11_full.T.reshape(-1)

abs_m_05 = exp_05["abs_mean"].T.reshape(-1)
abs_m_03 = exp_03["abs_mean"].T.reshape(-1)

F_m_05, F_var_05 = tscan_F_from_npz(exp_05)
std_F_05 = np.sqrt(F_var_05)
lF_05 = std_F_05.copy()
uF_05 = std_F_05.copy()

F_m_03, F_var_03 = tscan_F_from_npz(exp_03)
std_F_03 = np.sqrt(F_var_03)
lF_03 = std_F_03.copy()
uF_03 = std_F_03.copy()

x_F_05 = np.real(data_type1).astype(float).ravel()
x_F_03 = np.real(data_type2).astype(float).ravel()

F_d4_05 = (np.real(results_d4_type1).astype(float) * BETA_05 / 4.0).ravel()
F_d8_05 = (np.real(results_d8_type1).astype(float) * BETA_05 / 8.0).ravel()
F_d4_03 = (np.real(results_d4_type2).astype(float) * BETA_03 / 4.0).ravel()
F_d8_03 = (np.real(results_d8_type2).astype(float) * BETA_03 / 8.0).ravel()

data_limit = x_th_05.copy()
F_limit_th_05 = 0.5 * ln2 * np.ones_like(data_limit)
for x0 in zeros_list:
    F_limit_th_05 += 0.5 * ln2 * np.exp(-(data_limit - x0)**2 / (2 * sigma**2))
data_beta = free_axes["beta"]

data_npz_type3 = np.load(DATA_DIR / "data.npz")
result_d4_beta = data_npz_type3["result_d4"].squeeze()
result_d8_beta = data_npz_type3["result_d8"].squeeze()

BETA_VALUES = exp_beta["beta"]
N_POINTS = BETA_VALUES.size

F_exp_type3 = exp_beta["F_mean"]
std_F_exp_type3 = np.sqrt(exp_beta["F_var"])
lF_exp_type3 = std_F_exp_type3.copy()
uF_exp_type3 = std_F_exp_type3.copy()

data_beta = np.array(data_beta, dtype=float)
F_d4_type3 = result_d4_beta.astype(float) * data_beta / 4.0
F_d8_type3 = result_d8_beta.astype(float) * data_beta / 8.0

real_part_new = 0.1 + 0.0005 * np.arange(1, 1601)
LN2 = np.log(2.0)
y_main_new = (1 - real_part_new) * LN2
x0_peak = 0.5
sigma_peak = 0.001
amp_peak = 0.5 * LN2
y_main_new = y_main_new + amp_peak * np.exp(-(real_part_new - x0_peak) ** 2 / (2 * sigma_peak**2))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Arial",
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})
fig = plt.figure(figsize=(6.7, 4.2))

gs = fig.add_gridspec(
    2, 3,
    left=0.05, right=0.95,
    bottom=0.08, top=0.98,
    height_ratios=[3.8, 2],
    wspace=0.10,
    hspace=0.1
)

ax_a3 = fig.add_subplot(gs[0, 0], projection='3d')
ax_b3 = fig.add_subplot(gs[0, 1], projection='3d')
ax_c3 = fig.add_subplot(gs[0, 2], projection='3d')

ax_d2 = fig.add_subplot(gs[1, 0])
ax_e2 = fig.add_subplot(gs[1, 1])
ax_f2 = fig.add_subplot(gs[1, 2])

F_N16,  surf_a = plot_L2_surface_on_axis(ax_a3, N=16)
ax_a3.set_zlabel(r'$\mathcal{F}_1$', fontsize=FONT_3D_LABEL,labelpad=-9)

ax_b3.zaxis.label.set_position((0.02, 0.5))
F_N256, surf_b = plot_L2_surface_on_axis(ax_b3, N=256)
F_inf = compute_F1_gauss(beta_grid, t_grid)
F_inf_clip, surf_c = plot_inf_L2_panel_c(ax_c3, beta_grid, t_grid, F_inf)

ax_b3.set_zlabel("")
ax_c3.set_zlabel("")

ax_a3.text2D(-0.05, 0.95, "a", transform=ax_a3.transAxes,
           fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
ax_b3.text2D(-0.05, 0.95, "b", transform=ax_b3.transAxes,
           fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
ax_c3.text2D(-0.05, 0.95, "c", transform=ax_c3.transAxes,
           fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
for ax in (ax_a3, ax_b3, ax_c3):
    ax.set_xlim(0, 1.5)
    ax.set_xticks([0,0.5, 1.0, 1.5])
    ax.set_xticklabels(["0","0.5", "1", "1.5"], fontsize=FONT_3D_TICK)
ax_b3.set_zticks([0.0, 0.5, 1])
ax_b3.set_zticklabels(['0','0.5','1'],
                      fontsize=FONT_3D_TICK)
ax_a3.set_zticks([0.0, 0.5, 1.0, 1.5])
ax_a3.set_zticklabels(["0", "0.5", "1", "1.5"],
                      fontsize=FONT_3D_TICK)
ax_c3.set_zticks([0, 0.2, 0.4, 0.6])
ax_c3.set_zticklabels(["0", "0.2", "0.4", "0.6"],
                      fontsize=FONT_3D_TICK)
ax_a3.text2D(
    0.1, 0.93, r"$d=4$",
    transform=ax_a3.transAxes,
    ha="left", va="top",
    fontsize=10
)

ax_b3.text2D(
    0.1, 0.93, r"$d=8$",
    transform=ax_b3.transAxes,
    ha="left", va="top",
    fontsize=10
)

ax_c3.text2D(
    0.1, 0.93, r"$d\rightarrow \infty$",
    transform=ax_c3.transAxes,
    ha="left", va="top",
    fontsize=10
)

for ax in (ax_d2, ax_e2, ax_f2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZE_TICK_LABEL)
F_07_ln2 = 0.7 * ln2

y_line_D = np.full_like(x_F_03, F_07_ln2, dtype=float)

line_limit_D = add_colored_line(
    ax_d2,
    x_F_03, y_line_D,
    linestyle='-.',
    linewidth=LINE_WIDTH_D,
)

line_d8_D = add_colored_line(
    ax_d2,
    x_F_03, F_d8_03,
    linestyle='--',
    linewidth=LINE_WIDTH_D,
)

line_d4_D = add_colored_line(
    ax_d2,
    x_F_03, F_d4_03,
    linestyle='-',
    linewidth=LINE_WIDTH_D,
)

ax_d2.errorbar(
    x_exp_03, F_m_03, yerr=[lF_03, uF_03],
    fmt='o',
    color=col["A_exp"],
    mfc='none', mec=col["A_exp"],
    linewidth=LINE_WIDTH_EXP_POINT,
    capsize=CAP_SIZE,
    markersize=MARKER_SIZE_A,
    label=r"$\mathcal{F}_{\mathrm{exp}}(d=4)$"
)

ax_d2.set_ylabel(r"$\mathcal{F}_{1}$", fontsize=FONT_SIZE_AXES_LABEL,labelpad=0)
ax_d2.set_xlabel(r"$t$", fontsize=FONT_SIZE_AXES_LABEL)
ax_d2.text(-0.06, 1.08, "d", transform=ax_d2.transAxes,
           fontsize=PANEL_LABEL_SIZE, fontweight='bold',
           va='top', ha='left')
line_limit_E = add_colored_line(
    ax_e2, data_limit, F_limit_th_05,
    linestyle='-.',
    linewidth=LINE_WIDTH_D,
)

zeros_t = np.array([
    14.134725142,
    21.022039639,
    25.010857580,
    30.424876126,
    32.935061588,
])

delta_t = 0.1
for t0 in zeros_t:
    mask_local = (data_limit >= t0 - delta_t) & (data_limit <= t0 + delta_t)
    if np.any(mask_local):
        add_colored_line(
            ax_e2,
            data_limit[mask_local],
            F_limit_th_05[mask_local],
            linestyle='-',
            linewidth=LINE_WIDTH_D * 1.2,
        )
line_d8_E = add_colored_line(
    ax_e2, x_F_05, F_d8_05,
    linestyle='--',
    linewidth=LINE_WIDTH_D,
)

line_d4_E = add_colored_line(
    ax_e2, x_F_05, F_d4_05,
    linestyle='-',
    linewidth=LINE_WIDTH_D,
)

exp_plot =ax_e2.errorbar(
    x_exp_05, F_m_05, yerr=[lF_05, uF_05],
    fmt='o',
    color=col["A_exp"],
    mfc='none', mec=col["A_exp"],
    linewidth=LINE_WIDTH_EXP_POINT,
    capsize=CAP_SIZE,
    markersize=MARKER_SIZE_A,
    label=r"$\mathcal{F}_{\mathrm{exp}}(d=4)$"
)

ax_e2.set_xlabel(r"$t$", fontsize=FONT_SIZE_AXES_LABEL)
ax_e2.set_yticklabels([])
ax_e2.text(-0.06, 1.08, "e", transform=ax_e2.transAxes,
           fontsize=PANEL_LABEL_SIZE, fontweight='bold',
           va='top', ha='left')
for ax in (ax_d2, ax_e2,ax_f2):
    ax.set_ylim(0.0, 1.85)
xmin_F = min(x_th_03.min(), x_th_05.min()) * 0.95
xmax_F = max(x_th_03.max(), x_th_05.max()) * 1.02
for ax in (ax_d2, ax_e2):
    ax.set_xlim(xmin_F, xmax_F)
t_labels_pos_F = np.array([
    T_POINTS_F[0],
    T_POINTS_F[1] -0.35,
    T_POINTS_F[2]+0.05,
    T_POINTS_F[3] - 0.25,
    T_POINTS_F[4]
])
t_labels_text_F = [
    r"$t_1{=}14.12$",
    r"$t_2{=}20.96$",
    r"$t_3{=}25.09$",
    r"$t_4{=}30.44$",
    r"$t_5{=}32.93$"
]
F_label_y_F = np.array([1.43, 0.95, 1.1, 0.95, 1.57])

for xpos, ypos, txt in zip(t_labels_pos_F, F_label_y_F, t_labels_text_F):
    ax_e2.text(
        xpos, ypos, txt,
        fontsize=7,
        ha='center', va='bottom',fontweight='bold'
    )
legend_F_vals = np.linspace(VMIN_F, VMAX_F, 6)

legend_d4,  handler_d4  = make_grad_legend_handle(
    linestyle='-',
    label=r"$\mathcal{F}_{\mathrm{th}}(d=4)$"
)
legend_d8,  handler_d8  = make_grad_legend_handle(
    linestyle='--',
    label=r"$\mathcal{F}_{\mathrm{th}}(d=8)$"
)
legend_lim, handler_lim = make_grad_legend_handle(
    linestyle='-.',
    label=r"$\mathcal{F}_{\mathrm{th}}(d\rightarrow\infty)$"
)

handles = [legend_d4, exp_plot, legend_d8, legend_lim]
labels  = [h.get_label() for h in handles]
ax_d2.legend(
    handles=handles,
    labels=labels,
    handler_map={
        legend_d4:  handler_d4,
        legend_lim:  handler_lim,
        legend_d8: handler_d8,
        ErrorbarContainer: HandlerErrorbar(
            numpoints=1,
            xerr_size=0,
            yerr_size=0.7
        ),
    },
    loc='upper center',
    bbox_to_anchor=(0.5, 0.85),
    frameon=False,
    ncol=2,
    handlelength=1.8,
    handletextpad=2,
    fontsize=7.5,
    labelspacing=0.5,
    columnspacing=0.8
)

x_ticks = np.arange(10, 36, 5)

for ax in [ax_d2, ax_e2]:
    ax.set_xticks(x_ticks)
    ax.set_xlim(10, 35)
ax_d2.set_yticks([0.0, 0.5, 1.0, 1.5])
ax_d2.set_yticklabels(
    ["0", "0.5", "1", "1.5"]
)

ax_f2.grid(False)
ax_f2.set_facecolor("white")
ax_f2.tick_params(labelsize=FONT_SIZE_TICK_LABEL, width=0.8)

ax_f2.text(
    -0.06, 1.08, "f",
    transform=ax_f2.transAxes,
    va="top",
    ha="left",
    fontweight="bold",
    fontsize=PANEL_LABEL_SIZE,
)
idxF = slice(1, N_POINTS)

line_inf_F = add_colored_line(
    ax_f2,
    real_part_new, y_main_new,
    linestyle='-.',
    linewidth=LINE_WIDTH_D,
)

line_d8_F = add_colored_line(
    ax_f2,
    data_beta, F_d8_type3,
    linestyle='--',
    linewidth=LINE_WIDTH_D,
)

line_d4_F = add_colored_line(
    ax_f2,
    data_beta, F_d4_type3,
    linestyle='-',
    linewidth=LINE_WIDTH_D,
)

beta0 = 0.5
delta_beta = 0.01
mask_beta0 = (real_part_new >= beta0 - delta_beta) & (real_part_new <= beta0 + delta_beta)

if np.any(mask_beta0):
    add_colored_line(
        ax_f2,
        real_part_new[mask_beta0],
        y_main_new[mask_beta0],
        linestyle='-',
        linewidth=LINE_WIDTH_D * 1.2,
    )
ax_f2.errorbar(
    BETA_VALUES[idxF], F_exp_type3[idxF],
    yerr=[lF_exp_type3[idxF], uF_exp_type3[idxF]],
    fmt='o',
    color=col["A_exp"],
    mfc='none',
    linewidth=LINE_WIDTH_EXP_POINT,
    capsize=CAP_SIZE,
    markersize=MARKER_SIZE_A,
)

ax_f2.axvline(
    0.5, ymin=0, ymax=0.8,
    color=(0.5, 0.5, 0.5),
    linestyle=":",
    linewidth=0.5,
)

ax_f2.set_xlabel(r"$\beta$", fontsize=FONT_SIZE_AXES_LABEL, fontfamily=FONT_NAME)
ax_f2.set_xticks([0.0, 0.2,0.4,0.6,0.8, 1])
ax_f2.set_xticklabels(['0','0.2','0.4','0.6','0.8','1'],
                      fontsize=FONT_3D_TICK)
ax_f2.set_ylabel("")
ax_f2.set_yticklabels([])

ax_f2.set_xlim(0.0, 1)

ax_d2.text(
    0.5, 0.9, r"$\beta=0.3$",
    transform=ax_d2.transAxes,
    fontsize=10,
    ha="center",
    va="bottom",
)
ax_e2.text(
    0.5, 0.9, r"$\beta=0.5$",
    transform=ax_e2.transAxes,
    fontsize=10,
    ha="center",
    va="bottom",
)
ax_f2.text(
    0.5, 0.9, r"$t=14.134725$",
    transform=ax_f2.transAxes,
    fontsize=10,
    ha="center",
    va="bottom",
)

fig.subplots_adjust(
    left=0.01,
    right=0.99,
    bottom=0.01,
    top=0.99,
    wspace=0.01,
    hspace=0.02
)

pos_f2   = ax_f2.get_position()
right_f2 = pos_f2.x1

cbar_width = 0.015
gap        = 0.01

new_right_3d = right_f2 - cbar_width - gap
i=0
for ax in (ax_a3, ax_b3, ax_c3):
    i=i+1.4
    pos = ax.get_position()
    new_width = new_right_3d - pos.x0
    ax.set_position([pos.x0-cbar_width*i, pos.y0, pos.width, pos.height])
all_axes = [ax_a3, ax_b3, ax_c3, ax_d2, ax_e2, ax_f2]
pos_all  = [ax.get_position() for ax in all_axes]
y0 = min(p.y0 for p in pos_all)
y1 = max(p.y1 for p in pos_all)

h = (y1 - y0 -0.08) / 2.0
cbar_bottom = y0 + h

cbar_left = right_f2 - 3*cbar_width

cax = fig.add_axes([cbar_left, cbar_bottom+0.02, cbar_width, h-0.04])

import matplotlib.colors as colors
cbar = fig.colorbar(
    surf_c,
    cax=cax,
    norm=colors.Normalize(vmin=0, vmax=1.5)
)

ticks = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]
cbar.set_ticks(ticks)
cbar.set_ticklabels(["0", "0.3", "0.6", "0.9", "1.2", "1.5"])
cbar.ax.tick_params(
    labelsize=8,
    direction='in',
    length=3,
    width=0.8,
)
import matplotlib.patches as patches
bottom_color = CMAP_F(NORM_F(0.0))
cbar.ax.add_patch(
    patches.Rectangle(
        (0, 0), 1, 0.02,
        transform=cbar.ax.transAxes,
        color=bottom_color,
        linewidth=0,
        alpha=0.7,
        clip_on=False,
    )
)

cbar.outline.set_zorder(3)

plt.savefig("Fig4.pdf", bbox_inches="tight")
