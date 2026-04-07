import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from scipy.io import loadmat
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
    """
    3D double-stroke arrow:
      - The arrow shaft points from start to end
      - The arrowhead is a 'V' shape made of two line segments
      - The plane containing the arrowhead is determined by the direction vector v and the z-axis,
        so it is perpendicular to the x-y plane
    """
    start = np.array(start, dtype=float)
    end   = np.array(end,   dtype=float)

    v = end - start
    L = np.linalg.norm(v)
    if L == 0:
        return
    v_hat = v / L

    # Choose a reference vector; prefer the z-axis so the plane contains the z-axis -> perpendicular to the x-y plane
    z_axis = np.array([0.0, 0.0, 1.0])

    # If v is almost parallel to the z-axis, use the x-axis as the reference to avoid degeneracy
    if np.linalg.norm(np.cross(v_hat, z_axis)) < 1e-8:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = z_axis

    # In the plane spanned by {v_hat, ref}, find a unit vector n orthogonal to v_hat
    # Then the plane spanned by {v_hat, n} contains ref (usually z), and is therefore perpendicular to the x-y plane
    n = ref - np.dot(ref, v_hat) * v_hat
    n /= np.linalg.norm(n)

    # Draw the arrow shaft from start to end
    ax.plot(
        [start[0], end[0]*0.98+start[0]*0.02],
        [start[1], end[1]*0.98+start[1]*0.02],
        [start[2], end[2]*0.98+start[2]*0.02],
        color=color,
        lw=lw,
    )

    # Draw the arrowhead: use end as the tip, step back along -v_hat, then offset by head_width*n on both sides
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
    """
    Manually draw three "right-side/back-side" frame edges on a 3D axis.
    You can comment out any of them as needed.
    """
    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()


    # 3) Vertical edge at the back-right corner (x = x1, y = y1)
    ax.plot([x1, x1], [5, 5], [z0, z1],
            color=color, lw=lw)

# ============================================================
# 0. Global settings
# ============================================================
mp.mp.dps = 80  # High precision

# Beta-t grid (shared by all 3D panels)
NBETA, NT = 500, 500
BETA_MIN, BETA_MAX = 0.01, 1.5
T_MIN, T_MAX       = 5,    35

beta_grid = np.linspace(BETA_MIN, BETA_MAX, NBETA)
t_grid    = np.linspace(T_MIN,    T_MAX,    NT)
B_mesh, T_mesh = np.meshgrid(beta_grid, t_grid, indexing='xy')

# 3D colormap settings
MAX_F      = 1.5
VMIN_F     = 0.0
VMAX_F     = MAX_F
NORM_F     = mcolors.Normalize(vmin=VMIN_F, vmax=VMAX_F)
CMAP_F     = cm.get_cmap('turbo')

# Common constants and styles
DATA_DIR = Path("data")

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

# Original color dictionary (can still be used elsewhere if needed)
col = {
    "A_exp":    np.array([212, 86, 46]) / 255.0,
    "A_th":     np.array([68, 133, 199]) / 255.0,
    "A_shade":  np.array([0.85, 0.9, 0.98]),
    "ErrBar_A": np.array([229, 127, 102]) / 255.0,

    "X_exp":    np.array([212, 86, 46]) / 255.0,
    "XY_th":    np.array([68, 133, 199]) / 255.0,
}

# Use the same colormap above for all line colors in the three F panels below
curve_colors = {
    "limit": CMAP_F(NORM_F(MAX_F)),        # Close to the top of the colorbar
    "d8":    CMAP_F(NORM_F(MAX_F)),
    "d4":    CMAP_F(NORM_F(MAX_F)),
    "exp":   CMAP_F(NORM_F(MAX_F)),   # Experiment
}

# Line widths and fonts
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
mpl.rcParams['mathtext.rm'] = 'Arial'       # Set regular text in math expressions to Cambria
mpl.rcParams['mathtext.it'] = 'Computer Modern:italic'  # Set italic math text to Cambria
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

# ============================================================
# 1. General helper functions
# ============================================================

def load_mat(name):
    """Read name.mat from the data directory and remove MATLAB metadata."""
    mat = loadmat(DATA_DIR / f"{name}.mat")
    return {k: v for k, v in mat.items() if not k.startswith("__")}

def cell_to_list_2d(cell_arr):
    """Convert a MATLAB cell array (M, N) to a Python list[M][N], where each element is a 1D np.array."""
    M, N = cell_arr.shape
    out = [[None]*N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            out[i][j] = np.array(cell_arr[i, j]).ravel()
    return out

def cell_to_list_of_arrays(var):
    """Convert a MATLAB cell array (1xN / Nx1) to a Python list[ndarray]."""
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

    # Keep only finite values to avoid NaNs creating spurious line connections
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return None

    # Sort by x to avoid backtracking line segments
    idx_sort = np.argsort(x)
    x = x[idx_sort]
    y = y[idx_sort]

    # Split uniformly into n_segments along x
    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        return None  # Cannot draw a line if all x values are identical

    edges = np.linspace(x_min, x_max, n_segments + 1)

    # Different linestyles use different draw/skip patterns
    # mode = (period, on_len): in each period, draw the first on_len segments and skip the rest
    # Choose draw/skip patterns so dashed and dotted lines do not look too sparse
    if linestyle == '-':
        # Build segments with shape (n-1, 2, 2)
        points   = np.column_stack([x, y])
        segments = np.stack([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,
            cmap=cmap,      # Use the CMAP_F passed into the function
            norm=norm,      # Use the NORM_F passed into the function (0~1.5)
            linewidth=linewidth,
        )
        # Use y as the scalar so colors are mapped automatically to [vmin, vmax] = [0, 1.5]
        lc.set_array(y[:-1])

        ax.add_collection(lc)
        ax.autoscale_view()
        return lc

    elif linestyle == '--':
        # Short dashed line: draw 2 segments / skip 1 -> much denser than before
        period, on_len = 20, 10

    elif linestyle == '-.':
        # Dot-dash line: draw 1 segment / skip 1 -> much denser than before, appearing as a tight dotted line
        period, on_len = 12, 4



    last_line = None
    for i in range(n_segments):
        # Decide whether to draw this segment according to the pattern
        if (i % period) >= on_len:
            continue  # Skip this segment to create gaps for dashed/dash-dot styles

        # Current segment [edges[i], edges[i+1]]
        seg_mask = (x >= edges[i]) & (x <= edges[i+1])
        x_seg = x[seg_mask]
        y_seg = y[seg_mask]

        # Need at least two points to draw a segment
        if x_seg.size < 2:
            continue

        # Average height of this segment -> used to determine the color
        y_mid = float(np.mean(y_seg))
        color = cmap(norm(y_mid))

        line, = ax.plot(
            x_seg, y_seg,
            color=color,
            linewidth=linewidth,
        )
        last_line = line

    # You will usually call set_xlim / set_ylim later, so autoscale here is optional
    ax.autoscale_view()

    return last_line

class HandlerGradDashLine(HandlerLine2D):
    """Draw a line with a colormap gradient inside the legend (supports -, --, -., etc.)."""
    def __init__(self, cmap, norm, nseg=1000, **kwargs):
        self.cmap = cmap
        self.norm = norm
        self.nseg = nseg
        super().__init__(**kwargs)

    def create_artists(
        self, legend, orig_handle,
        x0, y0, width, height, fontsize, trans
    ):
        # Make the line inside the legend slightly longer
        scale = 1.8
        x_start = x0
        x_end   = x0 + width * scale
        y_mid   = y0 + 0.5 * height

        xs = np.linspace(x_start, x_end, self.nseg + 1)

        # Choose the draw/skip pattern based on the original handle's linestyle
        ls = orig_handle.get_linestyle()
        if ls in ("solid", "-"):
            period, on_len = 1, 1          # Solid line
        elif ls in ("--", "dashed"):
            period, on_len = 20, 10          # Relatively dense dashed line
        elif ls in ("-.", "dashdot"):
            period, on_len = 12, 3          # Slightly sparser dash-dot line

        artists = []
        for i in range(self.nseg):
            # Decide whether to draw this small segment based on the pattern
            if (i % period) >= on_len:
                continue

            x_seg = [xs[i], xs[i+1]]
            y_seg = [y_mid, y_mid]

            # Sweep the color from 0 to MAX_F
            v = MAX_F * (i + 0.5) / self.nseg
            color = self.cmap(self.norm(v))

            seg = plt.Line2D(
                x_seg, y_seg,
                linestyle='-',   # Draw solid mini-segments so the overall appearance is dashed
                linewidth=orig_handle.get_linewidth(),
                color=color,
                transform=trans,
            )
            artists.append(seg)

        return artists


def make_grad_legend_handle(linestyle, label):
    """
    Create a dummy line for legend use only:
      - The linestyle is determined by linestyle (-, --, -., :)
      - The color follows the colormap gradient over 0~MAX_F
    Returns (handle, handler) for convenient binding with handler_map in the legend.
    """
    # First create a regular Line2D to serve as a placeholder handle
    line = plt.Line2D(
        [], [],
        linestyle=linestyle,
        linewidth=LINE_WIDTH_D,
        color='black',   # The actual color will be overridden by HandlerGradDashLine
        label=label,
    )
    # Matching handler: share the global colormap / norm
    handler = HandlerGradDashLine(
        cmap=CMAP_F,
        norm=NORM_F,
        nseg=120,        # Use more segments for a smoother gradient
    )
    return line, handler



def make_legend_colored_line(y_value, linestyle='-', linewidth=3, cmap=CMAP_F, norm=NORM_F):
    """
    Generate a short colored line segment for the legend, with color corresponding to a given y_value.
    """
    # Create horizontal dummy data with a length suitable for legend display
    x_fake = np.linspace(0, 1, 20)
    y_fake = np.full_like(x_fake, y_value)

    points = np.array([x_fake, y_fake]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm,
                        linewidth=linewidth, linestyle=linestyle)
    lc.set_array(y_fake[:-1])  # Color values
    return lc


# ============================================================
# 2. 3D panels: L2 for N=16 / 256, and inf L2
# ============================================================

def plot_L2_surface_on_axis(ax, N):
    """
    Plot the truncated-N L2 surface on the given 3D axis ax:
    - Colored surface F_N(beta, t)
    - Two ridges at beta = 0.3 and 0.5
    - One ridge at t = t_1
    """
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

    # Main surface
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


    # Ridge-line styles
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

    # Three ridges
    plot_beta_ridge(0.3,'white')
    plot_beta_ridge(0.5,'#FF4B00')
    plot_t_ridge(14.134725,'black')

    # Axes and labels
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
        lab = tick.label1   # The labels in the left column
        x, y = lab.get_position()
        lab.set_position((x + shift, y))
    for tick in ax.xaxis.get_major_ticks():
        lab = tick.label1   # The labels in the left column
        x, y = lab.get_position()
        lab.set_position((x, y-0.2))

    ax.view_init(elev=35, azim=245)
    ax.set_box_aspect([2, 2.35, 1.48])
    # Overall axes background: very light gray
    

    # Color of the three coordinate panes: slightly darker light gray than the background
    pane_color = (1, 1, 1, 1.0)  # R, G, B, A

    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)
    grid_color = (0.8, 0.8, 0.8, 0.5)   # RGBA

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["color"] = grid_color
        axis._axinfo["grid"]["linewidth"] = 0.5 
        axis._axinfo['tick']['inward_factor']  = 0.2  # Inward tick length (relative units)
        axis._axinfo['tick']['outward_factor'] = 0.0


    return F_clipped, surf

def compute_F1_gauss(beta_grid, t_grid):
    """
    inf L2: generate F(beta, t) using a Gaussian model, consistent with the logic of your original script.
    """
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
    """
    inf L2 3D panel: colored surface + three ridges.
    """
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

    # Very light color + semi-transparency
    ax.plot_surface(
    B_plane, T_plane, Z_plane,
    color='#FFB703',   # A more saturated and brighter orange-yellow than #FFD166
    alpha=0.9,         # Increase opacity to make the plane appear more solid
    linewidth=0,
    antialiased=False,
    shade=False,
    )


    # Write RH on the plane (in 3D coordinates)
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
    # Overall axes background: very light gray
    

    # Color of the three coordinate panes: slightly darker light gray than the background
    pane_color = (1, 1, 1, 1.0)  # R, G, B, A

    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)
    grid_color = (0.8, 0.8, 0.8, 0.5)   # RGBA

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["color"] = grid_color
        axis._axinfo["grid"]["linewidth"] = 0.5 
        axis._axinfo['tick']['inward_factor']  = 0.2  # Inward tick length (relative units)
        axis._axinfo['tick']['outward_factor'] = 0.0 


    return F_clipped, surf

# ============================================================
# 3. Load and process F(t) data for beta = 0.3 and 0.5 (type1/type2)
# ============================================================

# --- Data loading ---
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

mat_results_free3 = load_mat("results_free3")
data_type2       = mat_results_free3["data"]
mat_results_free5 = load_mat("results_free5")
data_type1       = mat_results_free5["data"]

data_npz = np.load(DATA_DIR / "data2.npz")
results_d4_type2 = data_npz['results_d4_type2']
results_d8_type2 = data_npz['results_d8_type2']
results_d4_type1 = data_npz['results_d4_type1']
results_d8_type1 = data_npz['results_d8_type1']

# --- Preprocessing ---
scale2 = 2 / 2.54
M, N_cell = real_values_all_type1.shape

BETA_05 = 0.5
BETA_03 = 0.3

real_all_1 = cell_to_list_2d(real_values_all_type1)
imag_all_1 = cell_to_list_2d(imag_values_all_type1)
real_all_2 = cell_to_list_2d(real_values_all_type2)
imag_all_2 = cell_to_list_2d(imag_values_all_type2)

# β=0.5
X_all_s_05 = [[np.real(real_all_1[i][j]) * scale2 for j in range(N_cell)] for i in range(M)]
Y_all_s_05 = [[np.real(imag_all_1[i][j]) * scale2 for j in range(N_cell)] for i in range(M)]

abs_mean_exp_05 = np.zeros((M, N_cell))
std_abs_05 = np.zeros((M, N_cell))
for i in range(M):
    for j in range(N_cell):
        curX = X_all_s_05[i][j]
        curY = Y_all_s_05[i][j]
        cur_abs = np.sqrt(curX**2 + curY**2)
        abs_mean_exp_05[i, j] = np.mean(cur_abs)
        std_abs_05[i, j] = np.std(cur_abs, ddof=0)

abs_m_05 = abs_mean_exp_05.T.reshape(-1)
lAbs_05  = std_abs_05.T.reshape(-1)
uAbs_05  = std_abs_05.T.reshape(-1)

x_exp_05 = tt_type1.T.reshape(-1)
x_th_05  = tt_type11_full.T.reshape(-1)

X_t_05 = X_expdata_type11_full.T.reshape(-1)
Y_t_05 = Y_expdata_type11_full.T.reshape(-1)
Abs_th_05 = np.sqrt(np.real(X_t_05)**2 + np.real(Y_t_05)**2)

F_m_05 = -np.log(np.real(abs_m_05)) / 4.0
std_F_05 = (1/4.0) * (1.0 / np.real(abs_m_05)) * np.real(std_abs_05.T.reshape(-1))
lF_05 = std_F_05.copy()
uF_05 = std_F_05.copy()

F_th_05 = -np.log(np.real(Abs_th_05)) / 4.0
x_F_05 = np.real(data_type1).astype(float).ravel()

F_d4_05 = (np.real(results_d4_type1).astype(float) * BETA_05 / 4.0).ravel()
F_d8_05 = (np.real(results_d8_type1).astype(float) * BETA_05 / 8.0).ravel()

# Thermodynamic-limit curve at beta = 0.5
data_limit = x_th_05.copy()
F_limit_th_05 = 0.5 * ln2 * np.ones_like(data_limit)
for x0 in zeros_list:
    F_limit_th_05 += 0.5 * ln2 * np.exp(-(data_limit - x0)**2 / (2 * sigma**2))

# β=0.3
X_all_s_03 = [[np.real(real_all_2[i][j]) * scale2 for j in range(N_cell)] for i in range(M)]
Y_all_s_03 = [[np.real(imag_all_2[i][j]) * scale2 for j in range(N_cell)] for i in range(M)]

abs_mean_exp_03 = np.zeros((M, N_cell))
std_abs_03 = np.zeros((M, N_cell))
for i in range(M):
    for j in range(N_cell):
        curX = X_all_s_03[i][j]
        curY = Y_all_s_03[i][j]
        cur_abs = np.sqrt(curX**2 + curY**2)
        abs_mean_exp_03[i, j] = np.mean(cur_abs)
        std_abs_03[i, j] = np.std(cur_abs, ddof=0)

abs_m_03 = abs_mean_exp_03.T.reshape(-1)
lAbs_03  = std_abs_03.T.reshape(-1)
uAbs_03  = std_abs_03.T.reshape(-1)

x_exp_03 = tt_type1.T.reshape(-1)
x_th_03  = tt_type11_full.T.reshape(-1)

X_t_03 = X_expdata_type21_full.T.reshape(-1)
Y_t_03 = Y_expdata_type21_full.T.reshape(-1)
Abs_th_03 = np.sqrt(np.real(X_t_03)**2 + np.real(Y_t_03)**2)

F_m_03 = -np.log(np.real(abs_m_03)) / 4.0
std_F_03 = (1/4.0) * (1.0 / np.real(abs_m_03)) * np.real(std_abs_03.T.reshape(-1))
lF_03 = std_F_03.copy()
uF_03 = std_F_03.copy()

F_th_03 = -np.log(np.real(Abs_th_03)) / 4.0
x_F_03 = np.real(data_type2).astype(float).ravel()

F_d4_03 = (np.real(results_d4_type2).astype(float) * BETA_03 / 4.0).ravel()
F_d8_03 = (np.real(results_d8_type2).astype(float) * BETA_03 / 8.0).ravel()

# ============================================================
# 4. Load and process type3 data (F(beta))
# ============================================================

mat_type3 = loadmat(DATA_DIR / "type3_0925_8.mat")
real_values_all_raw = mat_type3["real_values_all"]
imag_values_all_raw = mat_type3["imag_values_all"]

mat_free_type3 = loadmat(DATA_DIR / "results_freereal_new.mat")
data_beta = np.array(mat_free_type3["real_part"]).squeeze()

data_npz_type3 = np.load(DATA_DIR / "data.npz")
result_d4_beta = data_npz_type3["result_d4"].squeeze()
result_d8_beta = data_npz_type3["result_d8"].squeeze()

x_points    = np.arange(1, 19) * 0.05   # Beta sampling points
BETA_VALUES = x_points
N_POINTS    = x_points.size

real_values_all_list = cell_to_list_of_arrays(real_values_all_raw)
imag_values_all_list = cell_to_list_of_arrays(imag_values_all_raw)

F_exp_type3     = np.full(N_POINTS, np.nan)
std_F_exp_type3 = np.full(N_POINTS, np.nan)

for i in range(N_POINTS):
    X_i = np.array(real_values_all_list[i], dtype=float) * scale2
    Y_i = np.array(imag_values_all_list[i], dtype=float) * scale2
    abs_i = np.sqrt(X_i**2 + Y_i**2)
    if abs_i.size > 1:
        valid = abs_i > 0
        if np.any(valid):
            F_samples = -np.log(abs_i[valid]) / 4.0
            F_exp_type3[i] = F_samples.mean()
            std_F_exp_type3[i] = F_samples.std(ddof=1)

lF_exp_type3 = std_F_exp_type3
uF_exp_type3 = std_F_exp_type3

data_beta = np.array(data_beta, dtype=float)
F_d4_type3 = result_d4_beta.astype(float) * data_beta / 4.0
F_d8_type3 = result_d8_beta.astype(float) * data_beta / 8.0

# Spike-like curve for d -> infinity
real_part_new = 0.1 + 0.0005 * np.arange(1, 1601)
LN2 = np.log(2.0)
y_main_new = (1 - real_part_new) * LN2
x0_peak = 0.5
sigma_peak = 0.001
amp_peak = 0.5 * LN2
y_main_new = y_main_new + amp_peak * np.exp(-(real_part_new - x0_peak) ** 2 / (2 * sigma_peak**2))

# ============================================================
# 5. Create the figure: 2 rows x 3 columns (same GridSpec, upper half 60%)
# ============================================================
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
    height_ratios=[3.8, 2],   # Upper 60%, lower 40%
    wspace=0.10,            # Smaller horizontal spacing than before
    hspace=0.1
)

# First row: 3D panels a, b, c
ax_a3 = fig.add_subplot(gs[0, 0], projection='3d')
ax_b3 = fig.add_subplot(gs[0, 1], projection='3d')
ax_c3 = fig.add_subplot(gs[0, 2], projection='3d')

# Second row: 2D panels d, e, f
ax_d2 = fig.add_subplot(gs[1, 0])
ax_e2 = fig.add_subplot(gs[1, 1])
ax_f2 = fig.add_subplot(gs[1, 2])

# ---- First row: 3D plots ----
F_N16,  surf_a = plot_L2_surface_on_axis(ax_a3, N=16)
ax_a3.set_zlabel(r'$\mathcal{F}_1$', fontsize=FONT_3D_LABEL,labelpad=-9)
#ax_b3.zaxis.labelpad = -5   # Try values between -5 and 5 and tune by eye

# 2. Fine-tune the position further: the closer x is to 0, the more it hugs the axis
ax_b3.zaxis.label.set_position((0.02, 0.5)) 
F_N256, surf_b = plot_L2_surface_on_axis(ax_b3, N=256)
F_inf = compute_F1_gauss(beta_grid, t_grid)
F_inf_clip, surf_c = plot_inf_L2_panel_c(ax_c3, beta_grid, t_grid, F_inf)

# Remove z-axis titles for panels b and c
ax_b3.set_zlabel("")
ax_c3.set_zlabel("")

# Small titles for a, b, c
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

# 2) z-axis ticks for panel A (ax_a3): 0, 0.2, 0.4, 0.6, 0.8
ax_b3.set_zticks([0.0, 0.5, 1])
ax_b3.set_zticklabels(['0','0.5','1'],
                      fontsize=FONT_3D_TICK)

# 3) z-axis ticks for panel B (ax_b3): 0, 0.5, 1.0, 1.5
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
    fontsize=10  # Or another font size if you prefer
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


# ------------------------------------------------
# (D) β = 0.3,  F1(t)
# ------------------------------------------------
F_07_ln2 = 0.7 * ln2

# --------- D: Three lines for beta = 0.3, drawn with add_colored_line using gradient colors ----------
# Horizontal line at 0.7 ln 2: construct an array with y equal to F_07_ln2 everywhere
y_line_D = np.full_like(x_F_03, F_07_ln2, dtype=float)
# Limit curve: dash-dot line (-.)
line_limit_D = add_colored_line(
    ax_d2,
    x_F_03, y_line_D,
    linestyle='-.',
    linewidth=LINE_WIDTH_D,
)

# d = 8: long dashed line (--)
line_d8_D = add_colored_line(
    ax_d2,
    x_F_03, F_d8_03,
    linestyle='--',
    linewidth=LINE_WIDTH_D,
)

# d = 4: solid line
line_d4_D = add_colored_line(
    ax_d2,
    x_F_03, F_d4_03,
    linestyle='-',
    linewidth=LINE_WIDTH_D,
)


# Experiment: keep the original scatter points + error bars, without connecting lines
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

# ------------------------------------------------
# (E) β = 0.5,  F1(t)
# ------------------------------------------------

# d -> infinity: thermodynamic-limit curve, shown as a solid line '-'
line_limit_E = add_colored_line(
    ax_e2, data_limit, F_limit_th_05,
    linestyle='-.',
    linewidth=LINE_WIDTH_D,
)

zeros_t = np.array([
    14.134725142,   # t1
    21.022039639,   # t2
    25.010857580,   # t3
    30.424876126,   # t4
    32.935061588,   # t5
])

delta_t = 0.1
for t0 in zeros_t:
    mask_local = (data_limit >= t0 - delta_t) & (data_limit <= t0 + delta_t)
    if np.any(mask_local):
        add_colored_line(
            ax_e2,
            data_limit[mask_local],
            F_limit_th_05[mask_local],
            linestyle='-',          # Force a solid line here
            linewidth=LINE_WIDTH_D * 1.2,
        )

line_d8_E = add_colored_line(
    ax_e2, x_F_05, F_d8_05,
    linestyle='--',               # Long dashed line
    linewidth=LINE_WIDTH_D,
)

line_d4_E = add_colored_line(
    ax_e2, x_F_05, F_d4_05,
    linestyle='-',               # Dash-dot line
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

# Unified F(t) axis range
for ax in (ax_d2, ax_e2,ax_f2):
    ax.set_ylim(0.0, 1.85)

xmin_F = min(x_th_03.min(), x_th_05.min()) * 0.95
xmax_F = max(x_th_03.max(), x_th_05.max()) * 1.02
for ax in (ax_d2, ax_e2):
    ax.set_xlim(xmin_F, xmax_F)

# Mark the Riemann-zero locations at the lower right of panel E
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

legend_F_vals = np.linspace(VMIN_F, VMAX_F, 6)   # Take 6 points

# ========== Legend handles (using gradient dashed lines) ==========
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

# Experiment: keep using your original hollow-circle markers
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
            numpoints=1,   # Draw only one point, centered
            xerr_size=0,   # Do not draw horizontal error bars
            yerr_size=0.7  # Control the vertical error-bar length (adjust visually)
        ),
    },
    loc='upper center',
    bbox_to_anchor=(0.5, 0.85),
    frameon=False,
    ncol=2,
    handlelength=1.8,    # Make the line in the legend longer
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

# ------------------------------------------------
# (F) type3: F1(β)
# ------------------------------------------------
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
# ---------- F: F1(beta) ----------
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
# Overwrite the region within +-0.1 around beta = 0.5 as a solid line
beta0 = 0.5
delta_beta = 0.01
mask_beta0 = (real_part_new >= beta0 - delta_beta) & (real_part_new <= beta0 + delta_beta)

if np.any(mask_beta0):
    add_colored_line(
        ax_f2,
        real_part_new[mask_beta0],
        y_main_new[mask_beta0],
        linestyle='-',                  # Force a solid line here
        linewidth=LINE_WIDTH_D * 1.2,   # Slightly thicker
    )


# Experiment: keep the original points + error bars, without connecting lines
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


# Dashed line at x = 0.5
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
# Remove the y-axis label and y ticks for the F panel
ax_f2.set_ylabel("")
ax_f2.set_yticklabels([])

ax_f2.set_xlim(0.0, 1)

ax_d2.text(
    0.5, 0.9, r"$\beta=0.3$",
    transform=ax_d2.transAxes,
    fontsize=10,
    ha="center",   # Shift slightly to the right for a more compact appearance
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
    left=0.01,   # Overall distance from the left edge
    right=0.99,  # Overall distance from the right edge (leave some room for the right-side colorbar)
    bottom=0.01, # Overall distance from the bottom edge
    top=0.99,    # Overall distance from the top edge
    wspace=0.01, # Horizontal spacing between plots
    hspace=0.02  # Vertical spacing between plots
)

# ============================================================
# 6. Single colorbar shared by all subplots
# ============================================================

pos_f2   = ax_f2.get_position()
right_f2 = pos_f2.x1

# Width of the colorbar and horizontal gap between the 3D plots and the colorbar
cbar_width = 0.015
gap        = 0.01   # Leave a little empty space to the right of the 3D plots before placing the colorbar

# First narrow the three 3D plots horizontally to make room for the vertical bar at the upper right
new_right_3d = right_f2 - cbar_width - gap
i=0
for ax in (ax_a3, ax_b3, ax_c3):
    i=i+1.4
    pos = ax.get_position()
    new_width = new_right_3d - pos.x0
    ax.set_position([pos.x0-cbar_width*i, pos.y0, pos.width, pos.height])

# Then take the vertical extent of all axes to compute the height of the upper-half colorbar
all_axes = [ax_a3, ax_b3, ax_c3, ax_d2, ax_e2, ax_f2]
pos_all  = [ax.get_position() for ax in all_axes]
y0 = min(p.y0 for p in pos_all)
y1 = max(p.y1 for p in pos_all)

# The colorbar occupies only the upper half of the total height
h = (y1 - y0 -0.08) / 2.0
cbar_bottom = y0 + h

# Key point: the right boundary of the colorbar equals the right boundary right_f2 of the lower-right plot
cbar_left = right_f2 - 3*cbar_width

cax = fig.add_axes([cbar_left, cbar_bottom+0.02, cbar_width, h-0.04])

import matplotlib.colors as colors
cbar = fig.colorbar(
    surf_c,   # Use the surface from panel c as the mappable
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
bottom_color = CMAP_F(NORM_F(0.0))   # Color corresponding to VMIN_F = 0
cbar.ax.add_patch(
    patches.Rectangle(
        (0, 0), 1, 0.02,              # (x0, y0, width, height) in axes coordinates
        transform=cbar.ax.transAxes,  # Use 0-1 axes coordinates
        color=bottom_color,
        linewidth=0,
        alpha=0.7,
        clip_on=False,
    )
)

# Make the black border lie above the patch
cbar.outline.set_zorder(3)
# ============================================================
# 7. Output
# ============================================================
plt.savefig("Fig4.pdf", bbox_inches="tight")