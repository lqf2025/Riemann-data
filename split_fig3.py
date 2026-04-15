import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

DATA_DIR = Path('data')
EXP_DIR = DATA_DIR / 'exp'
THEORY_DIR = DATA_DIR / 'theory'

exp_03 = np.load(EXP_DIR / 'tscan_beta03_exp.npz')
exp_05 = np.load(EXP_DIR / 'tscan_beta05_exp.npz')
exp_beta = np.load(EXP_DIR / 'beta_scan_exp.npz')

theo_03 = np.load(THEORY_DIR / 'tscan_beta03_theory.npz')
theo_05 = np.load(THEORY_DIR / 'tscan_beta05_theory.npz')
theo_beta = np.load(THEORY_DIR / 'beta_scan_theory.npz')

# -------------------- style --------------------
LINE_WIDTH_THEO = 1.55
LINE_WIDTH_TICK = 1.0
MARKER_SIZE_A = 4.3
MARKER_SIZE_ZERO = 3.5
CAP_SIZE = 2.6
LINE_WIDTH_ERR = 0.95
LINE_WIDTH_EXP_POINT = 0.95

FONT_NAME = 'Arial'
FONT_SIZE_AXES_LABEL = 12.6
FONT_SIZE_TICK_LABEL = 11.2
FONT_SIZE_TITLE = 12.8
FONT_SIZE_LEGEND = 11.0
FONT_SIZE_GROUP = 14.8
LETTER_FONT_SIZE = 13.8
FONT_SIZE_INPANEL = 12.7

plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, bm} \boldmath'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.it'] = 'Computer Modern:italic'
mpl.rcParams['mathtext.bf'] = 'Computer Modern:bold'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': FONT_NAME,
    'mathtext.fontset': 'stix',
    'axes.linewidth': LINE_WIDTH_TICK,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3.4,
    'ytick.major.size': 3.4,
    'xtick.major.width': 0.85,
    'ytick.major.width': 0.85,
})


def hex2rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


COLOR_THEO_BLUE = hex2rgb('#0072B2')
COLOR_EXP_ORANGE = hex2rgb('#D55E00')
COLOR_SHADE_CYAN = hex2rgb('#DFF3FF')
COLOR_LOOP_BLUE = hex2rgb('#005EA8')
COLOR_ZERO_GREY = hex2rgb('#808080')

col_t = {
    'A_th': COLOR_THEO_BLUE,
    'A_exp': COLOR_EXP_ORANGE,
    'A_shade': COLOR_SHADE_CYAN,
    'ErrBar_A': COLOR_EXP_ORANGE,
    'X_exp': COLOR_EXP_ORANGE,
    'XY_th': COLOR_LOOP_BLUE,
    'Zero': COLOR_ZERO_GREY,
}

rows_zeros = exp_05['rows_zeros']

# -------------------- top-panel data --------------------
abs_mean_exp_05 = exp_05['abs_mean']
var_abs_05 = exp_05['abs_var']
abs_m_05 = abs_mean_exp_05.T.reshape(-1)
lAbs_05 = np.sqrt(var_abs_05.T.reshape(-1))
uAbs_05 = lAbs_05.copy()
x_exp_05 = exp_05['t_flat']
x_th_05 = theo_05['t_flat']
X_t_05 = theo_05['curve_x_full'].T.reshape(-1)
Y_t_05 = theo_05['curve_y_full'].T.reshape(-1)
Abs_th_05 = np.sqrt(np.real(X_t_05) ** 2 + np.real(Y_t_05) ** 2)

abs_mean_exp_03 = exp_03['abs_mean']
var_abs_03 = exp_03['abs_var']
abs_m_03 = abs_mean_exp_03.T.reshape(-1)
lAbs_03 = np.sqrt(var_abs_03.T.reshape(-1))
uAbs_03 = lAbs_03.copy()
x_exp_03 = exp_03['t_flat']
x_th_03 = theo_03['t_flat']
X_t_03 = theo_03['curve_x_full'].T.reshape(-1)
Y_t_03 = theo_03['curve_y_full'].T.reshape(-1)
Abs_th_03 = np.sqrt(np.real(X_t_03) ** 2 + np.real(Y_t_03) ** 2)

X_theo_beta = theo_beta['curve_x']
Y_theo_beta = theo_beta['curve_y']
abs_theo_beta = np.sqrt(X_theo_beta ** 2 + Y_theo_beta ** 2)

BETA_VALUES = exp_beta['beta']
X_mean_exp_beta = exp_beta['x_mean']
Y_mean_exp_beta = exp_beta['y_mean']
abs_mean_exp_beta = exp_beta['abs_mean']
err_X_beta = np.sqrt(exp_beta['x_var'])
err_Y_beta = np.sqrt(exp_beta['y_var'])
err_abs_beta = np.sqrt(exp_beta['abs_var'])

# -------------------- bottom-panel data --------------------
X_mean_s_03 = exp_03['x_mean']
Y_mean_s_03 = exp_03['y_mean']
std_X_03 = np.sqrt(exp_03['x_var'])
std_Y_03 = np.sqrt(exp_03['y_var'])
X_theo_full_xy_03 = theo_03['xy_x_full']
Y_theo_full_xy_03 = theo_03['xy_y_full']

X_mean_s_05 = exp_05['x_mean']
Y_mean_s_05 = exp_05['y_mean']
std_X_05 = np.sqrt(exp_05['x_var'])
std_Y_05 = np.sqrt(exp_05['y_var'])
X_theo_full_xy_05 = theo_05['xy_x_full']
Y_theo_full_xy_05 = theo_05['xy_y_full']

X_data_03_exp = X_mean_s_03[rows_zeros, :]
X_data_03_th = np.real(X_theo_full_xy_03[rows_zeros, 200:601])
Y_data_03_exp = Y_mean_s_03[rows_zeros, :]
Y_data_03_th = np.real(Y_theo_full_xy_03[rows_zeros, 200:601])

X_data_05_exp = X_mean_s_05[rows_zeros, :]
X_data_05_th = np.real(X_theo_full_xy_05[rows_zeros, 200:601])
Y_data_05_exp = Y_mean_s_05[rows_zeros, :]
Y_data_05_th = np.real(Y_theo_full_xy_05[rows_zeros, 200:601])

std_X_03_sub = std_X_03[rows_zeros, :]
std_Y_03_sub = std_Y_03[rows_zeros, :]
std_X_05_sub = std_X_05[rows_zeros, :]
std_Y_05_sub = std_Y_05[rows_zeros, :]


def style_axes(ax):
    ax.set_facecolor('white')
    ax.tick_params(labelsize=FONT_SIZE_TICK_LABEL)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(top=False, right=False)


def add_panel_label(fig, ax, label, dx=0.026, dy=0.018):
    pos = ax.get_position()
    fig.text(
        pos.x0 - dx,
        pos.y1 + dy,
        label,
        fontsize=LETTER_FONT_SIZE,
        fontweight='bold',
        ha='left',
        va='top',
    )


# ==========================================================
# Figure 1: top row only (a,b,c)
# ==========================================================
fig1 = plt.figure(figsize=(7.8, 1.92))
fig1.patch.set_facecolor('white')

gs1 = fig1.add_gridspec(
    1, 3,
    width_ratios=[1.52, 1.52, 0.84],
    left=0.068, right=0.995,
    bottom=0.22, top=0.92,
    wspace=0.22,
)

ax_A = fig1.add_subplot(gs1[0, 0])
ax_B = fig1.add_subplot(gs1[0, 1], sharey=ax_A)
ax_C = fig1.add_subplot(gs1[0, 2])

for ax in [ax_A, ax_B, ax_C]:
    style_axes(ax)

xmin_A = min(x_th_03.min(), x_th_05.min()) - 0.4
xmax_A = max(x_th_03.max(), x_th_05.max()) + 0.4

ax_A.fill_between(x_th_03, Abs_th_03, 0.0, facecolor=col_t['A_shade'], alpha=0.9, linewidth=0)
h_th_A, = ax_A.plot(x_th_03, Abs_th_03, color=col_t['A_th'], linewidth=LINE_WIDTH_THEO)
h_exp_A = ax_A.errorbar(
    x_exp_03, abs_m_03, yerr=[lAbs_03, uAbs_03],
    fmt='o', ecolor=col_t['ErrBar_A'], elinewidth=LINE_WIDTH_ERR,
    capsize=CAP_SIZE, capthick=LINE_WIDTH_ERR,
    markerfacecolor='none', markeredgecolor=col_t['A_exp'],
    markeredgewidth=LINE_WIDTH_EXP_POINT, markersize=MARKER_SIZE_A,
)
ax_A.set_xlim(xmin_A, xmax_A)
ax_A.set_ylim(0, 0.63)
ax_A.set_xticks([10, 15, 20, 25, 30, 35])
ax_A.set_yticks([0, 0.25, 0.5])
ax_A.set_ylabel(r'$|\mathcal{L}|$', fontsize=FONT_SIZE_AXES_LABEL, labelpad=2.0)
ax_A.set_xlabel(r'$t$', fontsize=FONT_SIZE_AXES_LABEL, labelpad=1.8)
ax_A.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax_A.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax_A.text(0.50, 0.93, r'$\beta=0.3$', transform=ax_A.transAxes,
          fontsize=FONT_SIZE_TITLE + 0.3, fontweight='bold', ha='center', va='bottom')
ax_A.legend(
    [h_th_A, h_exp_A],
    [r'$\mathcal{L}_{\mathrm{theo}}$', r'$\mathcal{L}_{\mathrm{exp}}$'],
    loc='upper left', bbox_to_anchor=(0.02, 0.86), frameon=False, ncol=2,
    fontsize=FONT_SIZE_LEGEND, columnspacing=0.35, handlelength=1.7,
    handletextpad=0.5, borderaxespad=0.0,
)

ax_B.fill_between(x_th_05, Abs_th_05, 0.0, facecolor=col_t['A_shade'], alpha=0.9, linewidth=0)
ax_B.plot(x_th_05, Abs_th_05, color=col_t['A_th'], linewidth=LINE_WIDTH_THEO)
ax_B.errorbar(
    x_exp_05, abs_m_05, yerr=[lAbs_05, uAbs_05],
    fmt='o', ecolor=col_t['ErrBar_A'], elinewidth=LINE_WIDTH_ERR,
    capsize=CAP_SIZE, capthick=LINE_WIDTH_ERR,
    markerfacecolor='none', markeredgecolor=col_t['A_exp'],
    markeredgewidth=LINE_WIDTH_EXP_POINT, markersize=MARKER_SIZE_A,
)
ax_B.set_xlim(xmin_A, xmax_A)
ax_B.set_ylim(0, 0.63)
ax_B.set_xticks([10, 15, 20, 25, 30, 35])
ax_B.set_yticks([0, 0.25, 0.5])
ax_B.tick_params(axis='y', labelleft=False)
ax_B.set_xlabel(r'$t$', fontsize=FONT_SIZE_AXES_LABEL, labelpad=1.8)
ax_B.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax_B.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax_B.text(0.50, 0.93, r'$\beta=0.5$', transform=ax_B.transAxes,
          fontsize=FONT_SIZE_TITLE + 0.3, fontweight='bold', ha='center', va='bottom')

idx_beta_main = slice(1, 18)
ax_C.fill_between(BETA_VALUES[idx_beta_main], abs_theo_beta[idx_beta_main], 0.0,
                  facecolor=col_t['A_shade'], alpha=0.9, linewidth=0)
ax_C.plot(BETA_VALUES[idx_beta_main], abs_theo_beta[idx_beta_main], '-',
          color=col_t['A_th'], linewidth=LINE_WIDTH_THEO)
ax_C.errorbar(
    BETA_VALUES[idx_beta_main], abs_mean_exp_beta[idx_beta_main],
    yerr=err_abs_beta[idx_beta_main],
    fmt='o', ecolor=col_t['ErrBar_A'], elinewidth=LINE_WIDTH_ERR,
    capsize=CAP_SIZE, capthick=LINE_WIDTH_ERR,
    markerfacecolor='none', markeredgecolor=col_t['X_exp'],
    markeredgewidth=LINE_WIDTH_EXP_POINT, markersize=MARKER_SIZE_A,
)
ax_C.axvline(0.5, ymin=0, ymax=0.82, color=col_t['Zero'], linestyle=':', linewidth=0.9)
ax_C.set_xlim(0, 1)
ax_C.set_ylim(0, 0.18)
ax_C.set_xticks([0.1, 0.5, 0.9])
ax_C.set_yticks([0, 0.08, 0.16])
ax_C.set_xlabel(r'$\beta$', fontsize=FONT_SIZE_AXES_LABEL, labelpad=1.8)
ax_C.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax_C.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax_C.text(0.52, 0.93, r'$t=14.134725$', transform=ax_C.transAxes,
          fontsize=FONT_SIZE_TITLE + 0.2, fontweight='bold', ha='center', va='bottom')

add_panel_label(fig1, ax_A, 'a')
add_panel_label(fig1, ax_B, 'b')
add_panel_label(fig1, ax_C, 'c')

fig1.savefig('Fig3_top_rightf2.pdf', bbox_inches='tight', pad_inches=0.01)


# ==========================================================
# Figure 2: bottom panels with f on the right
# ==========================================================
fig2 = plt.figure(figsize=(8.55, 3.38))
fig2.patch.set_facecolor('white')

outer = fig2.add_gridspec(
    1, 2,
    width_ratios=[4.10, 1.62],
    left=0.07, right=0.992,
    bottom=0.14, top=0.91,
    wspace=0.14,
)

left = outer[0, 0].subgridspec(2, 3, wspace=0.19, hspace=0.30)
right = outer[0, 1].subgridspec(1, 1)

ax_d = [fig2.add_subplot(left[0, j]) for j in range(3)]
ax_e = [fig2.add_subplot(left[1, j], sharex=ax_d[j], sharey=ax_d[j]) for j in range(3)]
ax_f = fig2.add_subplot(right[0, 0])

for ax in ax_d + ax_e + [ax_f]:
    style_axes(ax)

xlims_cols = [(-0.12, 0.46), (-0.22, 0.54), (-0.12, 0.55)]
ylim_phase = (-0.42, 0.36)
col_titles = [r'$t\in[12,16]$', r'$t\in[23,27]$', r'$t\in[31,35]$']

for j in range(3):
    ax = ax_d[j]
    ax.plot(X_data_03_th[j, :], Y_data_03_th[j, :], '-', linewidth=1.85, color=col_t['XY_th'], zorder=1)
    ax.errorbar(
        X_data_03_exp[j, :], Y_data_03_exp[j, :],
        yerr=std_Y_03_sub[j, :], xerr=std_X_03_sub[j, :],
        fmt='o', ecolor=col_t['ErrBar_A'], elinewidth=LINE_WIDTH_ERR,
        capsize=CAP_SIZE, capthick=LINE_WIDTH_ERR,
        markerfacecolor='none', markeredgecolor=col_t['X_exp'],
        markeredgewidth=LINE_WIDTH_ERR, markersize=MARKER_SIZE_ZERO, zorder=3,
    )
    ax.axvline(0, linestyle=':', color=col_t['Zero'], linewidth=0.8, zorder=0)
    ax.axhline(0, linestyle=':', color=col_t['Zero'], linewidth=0.8, zorder=0)
    ax.set_xlim(*xlims_cols[j])
    ax.set_ylim(*ylim_phase)
    ax.set_xticks([0.0, 0.2, 0.4])
    ax.set_yticks([-0.25, 0, 0.25])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.set_box_aspect(0.82)
    ax.text(0.50, 0.95, col_titles[j], transform=ax.transAxes,
            ha='center', va='top', fontsize=FONT_SIZE_INPANEL, fontweight='bold')
    if j == 0:
        ax.text(0.06, 0.95, r'$\beta=0.3$', transform=ax.transAxes,
                ha='left', va='top', fontsize=FONT_SIZE_GROUP, fontweight='bold')
    if j > 0:
        ax.tick_params(axis='y', labelleft=False)

    ax = ax_e[j]
    ax.plot(X_data_05_th[j, :], Y_data_05_th[j, :], '-', linewidth=1.85, color=col_t['XY_th'], zorder=1)
    ax.errorbar(
        X_data_05_exp[j, :], Y_data_05_exp[j, :],
        yerr=std_Y_05_sub[j, :], xerr=std_X_05_sub[j, :],
        fmt='o', ecolor=col_t['ErrBar_A'], elinewidth=LINE_WIDTH_ERR,
        capsize=CAP_SIZE, capthick=LINE_WIDTH_ERR,
        markerfacecolor='none', markeredgecolor=col_t['X_exp'],
        markeredgewidth=LINE_WIDTH_ERR, markersize=MARKER_SIZE_ZERO, zorder=3,
    )
    ax.axvline(0, linestyle=':', color=col_t['Zero'], linewidth=0.8, zorder=0)
    ax.axhline(0, linestyle=':', color=col_t['Zero'], linewidth=0.8, zorder=0)
    ax.set_xlim(*xlims_cols[j])
    ax.set_ylim(*ylim_phase)
    ax.set_xticks([0.0, 0.2, 0.4])
    ax.set_yticks([-0.25, 0, 0.25])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.set_box_aspect(0.82)
    ax.text(0.50, 0.95, col_titles[j], transform=ax.transAxes,
            ha='center', va='top', fontsize=FONT_SIZE_INPANEL, fontweight='bold')
    if j == 0:
        ax.text(0.06, 0.95, r'$\beta=0.5$', transform=ax.transAxes,
                ha='left', va='top', fontsize=FONT_SIZE_GROUP, fontweight='bold')
    if j > 0:
        ax.tick_params(axis='y', labelleft=False)

ax_d[0].set_ylabel(r'$\langle\sigma_y\rangle$', fontsize=FONT_SIZE_AXES_LABEL, labelpad=1.8)
ax_e[0].set_ylabel(r'$\langle\sigma_y\rangle$', fontsize=FONT_SIZE_AXES_LABEL, labelpad=1.8)
for ax in ax_e:
    ax.set_xlabel(r'$\langle\sigma_x\rangle$', fontsize=FONT_SIZE_AXES_LABEL, labelpad=1.8)

idx_beta_zoom = slice(6, 13)
ax_f.plot(X_theo_beta[idx_beta_zoom], Y_theo_beta[idx_beta_zoom], '-',
          linewidth=1.75, color=col_t['XY_th'], zorder=1)
ax_f.errorbar(
    X_mean_exp_beta[idx_beta_zoom], Y_mean_exp_beta[idx_beta_zoom],
    xerr=err_X_beta[idx_beta_zoom], yerr=err_Y_beta[idx_beta_zoom],
    fmt='o', ecolor=col_t['ErrBar_A'], elinewidth=LINE_WIDTH_ERR,
    capsize=CAP_SIZE, capthick=LINE_WIDTH_ERR,
    markerfacecolor='none', markeredgecolor=col_t['X_exp'],
    markeredgewidth=LINE_WIDTH_ERR, markersize=MARKER_SIZE_ZERO, zorder=3,
)
ax_f.axvline(0, linestyle=':', color=col_t['Zero'], linewidth=0.8, zorder=0)
ax_f.axhline(0, linestyle=':', color=col_t['Zero'], linewidth=0.8, zorder=0)

x_min = min(X_theo_beta[idx_beta_zoom].min(), X_mean_exp_beta[idx_beta_zoom].min())
x_max = max(X_theo_beta[idx_beta_zoom].max(), X_mean_exp_beta[idx_beta_zoom].max())
ax_f.set_xlim(x_min - 0.013, x_max + 0.013)
ax_f.set_ylim(-0.01, 0.045)
ax_f.set_xticks([0, 0.05])
ax_f.set_yticks([0, 0.03])
ax_f.set_xlabel(r'$\langle\sigma_x\rangle$', fontsize=FONT_SIZE_AXES_LABEL, labelpad=1.8)
ax_f.set_ylabel(r'$\langle\sigma_y\rangle$', fontsize=FONT_SIZE_AXES_LABEL, labelpad=1.8)
ax_f.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax_f.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax_f.set_box_aspect(1.42)
ax_f.text(0.53, 0.95, r'$\beta\in[0.35,0.65]$', transform=ax_f.transAxes,
          ha='center', va='top', fontsize=FONT_SIZE_GROUP - 0.6, fontweight='bold')

add_panel_label(fig2, ax_d[0], 'a', dx=0.028, dy=0.018)
add_panel_label(fig2, ax_e[0], 'b', dx=0.028, dy=0.018)
add_panel_label(fig2, ax_f, 'c', dx=0.022, dy=0.018)

fig2.savefig('Fig3_bottom_rightf2.pdf', bbox_inches='tight', pad_inches=0.01)
