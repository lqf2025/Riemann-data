import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpmath import zeta, pi, log, sqrt, exp
from matplotlib.axes import Axes
from scipy.special import loggamma
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties
plt.rcParams['axes.formatter.use_locale'] = True
plt.rcParams.update({
    'xtick.direction': 'in',          # 刻度线方向朝内
    'ytick.direction': 'in'          # 刻度线方向朝内 
})
class CommaOffsetOnly(mticker.ScalarFormatter):
    """只重写 offset 文本，用逗号分组；主刻度保持 ScalarFormatter 默认行为。"""
    def __init__(self, decimals=0, show_sign=False, **kwargs):
        super().__init__(useOffset=True, **kwargs)
        self.set_scientific(False)    # 不用科学计数法
        self.set_powerlimits((0, 0))  # 等价于 style='plain'
        self.decimals = decimals
        self.show_sign = show_sign

    def __call__(self, x, pos=None):
        # 主刻度标签：完全沿用父类
        return super().__call__(x, pos)

    def get_offset(self):
        off = getattr(self, 'offset', 0.0)
        if not off:
            return ''
        # 整数 -> 千分位；否则保留指定小数位
        if self.decimals == 0 and float(off).is_integer():
            s = f'{int(off):,}'
        else:
            s = f'{off:,.{self.decimals}f}'
        if self.show_sign and off > 0:
            s = '+' + s
        return s


# Font configuration aligned with Fig4.py.
# The final active math fontset is STIX, preventing external Computer Modern font lookup warnings.
plt.rcParams["text.usetex"] = False

LINE_WIDTH_tick = 1.0
LINE_WIDTH_MAIN = 2.35
LINE_WIDTH_ZERO = 1.25
LINE_WIDTH_INSET = 1.25
MARKER_SIZE_INSET = 4.0

# Font sizes are tuned to match the lighter density of Fig4.py and Fig5.py.
FIG_SIZE = (13.3, 6.55)

FS_LABEL = 19
FS_TICK = 18
FS_LEGEND = 18.5
FS_PANEL = 20
FS_INSET_LABEL = 14
FS_INSET_XLABEL = 11
FS_INSET_TICK = 11.5
FS_OFFSET = 17

LEGEND_COLUMN_SPACING = 0.65
LEGEND_COLUMN_SPACING_C = 0.75
LEGEND_HANDLE_LENGTH = 1.35
LEGEND_HANDLE_TEXT_PAD = 0.28
INSET_YLABEL_PAD = 1.0
INSET_XLABEL_PAD = 2
INSET_XLABEL_PAD_A = 0.5
INSET_POS_TOP_A = [0.70, 0.82, 0.29, 0.27]
INSET_POS_TOP = [0.70, 0.74, 0.29, 0.30]
INSET_POS_BOTTOM = [0.71, 0.52, 0.28, 0.46]

FONT_NAME = "Arial"

plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Arial"
mpl.rcParams["mathtext.it"] = "Computer Modern:italic"
mpl.rcParams["mathtext.bf"] = "Computer Modern:bold"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "mathtext.fontset": "stix",
    "axes.linewidth": LINE_WIDTH_tick,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.major.width": 0.85,
    "ytick.major.width": 0.85,
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
})

# Normal-weight font for offset text such as +6,595,000.
prop = FontProperties(family=FONT_NAME, size=FS_OFFSET, weight="normal", style="normal")


def style_main_axis(ax):
    ax.tick_params(axis="both", which="major", labelsize=FS_TICK, width=0.85, length=3.2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(LINE_WIDTH_tick)
    ax.spines["bottom"].set_linewidth(LINE_WIDTH_tick)


def style_inset_axis(ax):
    ax.set_facecolor((1, 1, 1, 0.94))
    ax.tick_params(axis="both", which="major", labelsize=FS_INSET_TICK, width=0.75, length=2.6)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)


fig = plt.figure(figsize=FIG_SIZE)
fig.patch.set_facecolor("white")
color1, color2, color3 = ["#D55E00", "#0072B2", "#682487"]
gs = plt.GridSpec(2, 2)
ax_main = fig.add_subplot(gs[0, 0])
data=np.load("GLA1.npz", allow_pickle=True)
Z_acc=data['Z_acc']
Z_app=data['Z_app']
zeros_t_acc=data['zeros_t_acc']
zeros_t_app=data['zeros_t_app']
zero_diffs=data['zero_diffs']
t_vals = np.linspace(420, 450, 1000000)
ax_main.plot(
    t_vals, Z_app,
    label=r'$\mathcal{G}(\frac{1}{2},t)$',
    color=color2, linewidth=LINE_WIDTH_MAIN, solid_capstyle="round"
)
ax_main.plot(
    t_vals, Z_acc,
    linestyle=(0, (5, 4)),
    label=r'$\frac{Z\left(\frac{1}{2}+it\right)}{2\mathcal{Z}\left(\frac{1}{2},H_0\right)}$',
    color=color1, linewidth=LINE_WIDTH_MAIN, solid_capstyle="round"
)
ymin, ymax = ax_main.get_ylim()
# for z in zeros_t_acc:
#     ax_main.vlines(
#     z,
#     ymin=ymin,  # bottom of the plot
#     ymax=0,                       # up to y = 0
#     color=color1,
#     linestyle=(0, (3, 3)),
#     alpha=0.5,
#     linewidth=1.5
#     )
#     #ax_main3.axvline(z, color=color1, linestyle=":", alpha=0.5,linewidth=2, ymax=0)
# for z in zeros_t_app:
#     #ax_main3.axvline(z, color=color2, linestyle="-.", alpha=0.5,linewidth=2, ymax=0)
#     ax_main.vlines(
#     z,
#     ymin=ymin,  # bottom of the plot
#     ymax=0,                       # up to y = 0
#     color=color2,
#     linestyle=(0, (6, 2, 1, 2)),
#     alpha=0.5,
#     linewidth=1.5
#     )
for z in zeros_t_acc:
    ax_main.vlines(
        z,
        ymin=ymin,
        ymax=0,
        color=color1,                # 深红，物理期刊常用
        linestyle=(0, (5, 3)),          # 长虚线
        alpha=0.6,
        linewidth=LINE_WIDTH_ZERO,
        zorder=2
    )

for z in zeros_t_app:
    ax_main.vlines(
        z,
        ymin=ymin,
        ymax=0,
        color=color2,                # 深蓝，稳重不刺眼
        linestyle=(0, (3, 2, 1, 2)),    # 点-划线
        alpha=0.8,
        linewidth=LINE_WIDTH_ZERO,
        zorder=3
    )

ax_main.set_ylim(ymin,ymax)
#ax_main.set_title("Riemann Z-function Z(t) on Re(s)=1/2, 420 ≤ t ≤ 460")
#ax_main.set_title(r'Riemann $Z$-function $Z(t)$ on $\Re(s)=1/2$, $420 \leq t \leq 460$')
ax_main.set_xlabel(r'$t$',fontsize=FS_LABEL, labelpad=2)
ax_main.set_ylabel('GLA',fontsize=FS_LABEL, labelpad=2)
ax_main.legend(
    fontsize=FS_LEGEND, frameon=False, bbox_to_anchor=(0.27, 0.94),
    loc='center', ncol=2, columnspacing=LEGEND_COLUMN_SPACING,
    handlelength=LEGEND_HANDLE_LENGTH,
    handletextpad=LEGEND_HANDLE_TEXT_PAD, borderaxespad=0.0
)
ax_main.set_xlim(419.5, 450.5)
ax_main.set_title('a', x=-0.04, y=1.005, fontsize=FS_PANEL, fontweight='bold')
style_main_axis(ax_main)
#ax_main.grid(True)
ax_main.set_yticks([-0.5, 0, 0.5])
ax_main.set_yticklabels(["-0.5", "0", "0.5"])
# Create inset plot for differences
ax_inset = ax_main.inset_axes(INSET_POS_TOP_A)  # [left, bottom, width, height]
min_len = min(len(zeros_t_acc), len(zeros_t_app))
ax_inset.plot(
    range(1, min_len + 1), zero_diffs,
    marker='o', markersize=MARKER_SIZE_INSET, markeredgewidth=0,
    linestyle='-', linewidth=LINE_WIDTH_INSET, color=color3
)
ax_inset.set_xlabel("Zero Index", fontsize=FS_INSET_XLABEL, labelpad=INSET_XLABEL_PAD_A)
ax_inset.set_ylabel(r'$\delta t$', fontsize=FS_INSET_LABEL, labelpad=INSET_YLABEL_PAD)
style_inset_axis(ax_inset)
ax_inset.set_yticks([-0.1, 0, 0.1])
ax_inset.set_yticklabels(["-0.1", "0", "0.1"])

ax_main2 = fig.add_subplot(gs[0, 1])
data=np.load("GLA2.npz", allow_pickle=True)
Z_acc=data['Z_acc']
Z_app=data['Z_app']
zeros_t_acc=data['zeros_t_acc']
zeros_t_app=data['zeros_t_app']
zero_diffs=data['zero_diffs']
ax_main2.set_xlim(6595000-0.2,6595010.2)
t_vals = np.linspace(6595000, 6595010, 1000000)
ax_main2.plot(
    t_vals, Z_app,
    label=r'$\mathcal{G}(\frac{1}{2},t)$',
    color=color2, linewidth=LINE_WIDTH_MAIN, solid_capstyle="round"
)
ax_main2.plot(
    t_vals, Z_acc,
    linestyle=(0, (5, 4)),
    label=r'$\frac{Z\left(\frac{1}{2}+it\right)}{2\mathcal{Z}\left(\frac{1}{2},H_0\right)}$',
    color=color1, linewidth=LINE_WIDTH_MAIN, solid_capstyle="round"
)
ymin, ymax = ax_main2.get_ylim()
for z in zeros_t_acc:
    ax_main2.vlines(
        z,
        ymin=ymin,
        ymax=0,
        color=color1,                # 深红，物理期刊常用
        linestyle=(0, (5, 3)),          # 长虚线
        alpha=0.6,
        linewidth=LINE_WIDTH_ZERO,
        zorder=2
    )

for z in zeros_t_app:
    ax_main2.vlines(
        z,
        ymin=ymin,
        ymax=0,
        color=color2,                # 深蓝，稳重不刺眼
        linestyle=(0, (3, 2, 1, 2)),    # 点-划线
        alpha=0.8,
        linewidth=LINE_WIDTH_ZERO,
        zorder=3
    )
ax_main2.set_ylim(ymin,ymax)
#ax_main.set_title("Riemann Z-function Z(t) on Re(s)=1/2, 420 ≤ t ≤ 460")
#ax_main.set_title(r'Riemann $Z$-function $Z(t)$ on $\Re(s)=1/2$, $420 \leq t \leq 460$')
ax_main2.set_xlabel(r'$t$',fontsize=FS_LABEL, labelpad=2)
ax_main2.set_ylabel('GLA',fontsize=FS_LABEL, labelpad=2)
legend=ax_main2.legend(
    fontsize=FS_LEGEND, frameon=False, bbox_to_anchor=(0.34, 0.92),
    loc='center', ncol=2, columnspacing=LEGEND_COLUMN_SPACING,
    facecolor='white', handlelength=LEGEND_HANDLE_LENGTH,
    handletextpad=LEGEND_HANDLE_TEXT_PAD, borderaxespad=0.0
)
ax_main2.set_title('b', x=-0.04, y=1.005, fontsize=FS_PANEL, fontweight='bold')
style_main_axis(ax_main2)
ax_main2.set_yticks([0, 0.05])
ax_main2.set_yticklabels(["0", "0.05"])
fmt = CommaOffsetOnly(decimals=0, show_sign=True) 
ax_main2.xaxis.set_major_formatter(fmt)
offset_text2 = ax_main2.get_xaxis().get_offset_text()
offset_text2.set_fontproperties(prop)
offset_text2.set_fontweight('normal')
offset_text2.set_fontstyle('normal')

#ax_main.grid(True)
# Create inset plot for differences
ax_inset = ax_main2.inset_axes(INSET_POS_TOP_A)  # [left, bottom, width, height]
min_len = min(len(zeros_t_acc), len(zeros_t_app))
ax_inset.plot(
    range(1, min_len + 1), zero_diffs,
    marker='o', markersize=MARKER_SIZE_INSET, markeredgewidth=0,
    linestyle='-', linewidth=LINE_WIDTH_INSET, color=color3
)
ax_inset.set_xlabel("Zero Index", fontsize=FS_INSET_XLABEL, labelpad=INSET_XLABEL_PAD)
#ax_inset.set_ylabel(r'|t_\mathrm{acc} - t_\mathrm{app}|', fontsize=6)
ax_inset.set_ylabel(r'$\delta t$', fontsize=FS_INSET_LABEL, labelpad=INSET_YLABEL_PAD)
style_inset_axis(ax_inset)
ax_inset.set_yticks([-0.002, 0,0.002])
ax_inset.set_yticklabels(["-2e-3", "0", "2e-3"])

#ax_inset.grid(True)

ax_main3 = fig.add_subplot(gs[1, :])
data=np.load("GLA3.npz", allow_pickle=True)
Z_acc=data['Z_acc']
Z_app=data['Z_app']
zeros_t_acc=data['zeros_t_acc']
zeros_t_app=data['zeros_t_app']
#print(len(zeros_t_acc),len(zeros_t_app))
#zero_diffs=data['zero_diffs']
diff=[]
zeros_t_appfix=[]
for i in zeros_t_acc:
    idx = np.argmin(np.abs(zeros_t_app - i))
    diff.append(i-zeros_t_app[idx])
    zeros_t_appfix.append(zeros_t_app[idx])
ax_main3.set_xlim(267653395648-0.2,267653395648+12.2)
t_vals = np.linspace(267653395648, 267653395648+12, 1000000)
ax_main3.plot(
    t_vals, Z_app,
    label=r'$\mathcal{G}(\frac{1}{2},t)$',
    color=color2, linewidth=LINE_WIDTH_MAIN, solid_capstyle="round"
)
ax_main3.plot(
    t_vals, Z_acc,
    linestyle=(0, (5, 4)),
    label=r'$\frac{Z\left(\frac{1}{2}+it\right)}{2\mathcal{Z}\left(\frac{1}{2},H_0\right)}$',
    color=color1, linewidth=LINE_WIDTH_MAIN, solid_capstyle="round"
)
ymin, ymax = ax_main3.get_ylim()
for z in zeros_t_acc:
    ax_main3.vlines(
        z,
        ymin=ymin,
        ymax=0,
        color=color1,                # 深红，物理期刊常用
        linestyle=(0, (5, 3)),          # 长虚线
        alpha=0.6,
        linewidth=LINE_WIDTH_ZERO,
        zorder=2
    )

for z in zeros_t_appfix:
    ax_main3.vlines(
        z,
        ymin=ymin,
        ymax=0,
        color=color2,                # 深蓝，稳重不刺眼
        linestyle=(0, (3, 2, 1, 2)),    # 点-划线
        alpha=0.8,
        linewidth=LINE_WIDTH_ZERO,
        zorder=3
    )
ax_main3.set_ylim(ymin,ymax)
#ax_main.set_title("Riemann Z-function Z(t) on Re(s)=1/2, 420 ≤ t ≤ 460")
#ax_main.set_title(r'Riemann $Z$-function $Z(t)$ on $\Re(s)=1/2$, $420 \leq t \leq 460$')
ax_main3.set_xlabel(r'$t$',fontsize=FS_LABEL, labelpad=2)
ax_main3.set_ylabel('GLA',fontsize=FS_LABEL, labelpad=2)
ax_main3.legend(
    fontsize=FS_LEGEND, frameon=False, bbox_to_anchor=(0.41, 0.93),
    loc='center', ncol=2, columnspacing=LEGEND_COLUMN_SPACING_C,
    handlelength=LEGEND_HANDLE_LENGTH,
    handletextpad=LEGEND_HANDLE_TEXT_PAD, borderaxespad=0.0
)
ax_main3.set_title('c', x=-0.02, y=1.005, fontsize=FS_PANEL, fontweight='bold')
style_main_axis(ax_main3)

fmt = CommaOffsetOnly(decimals=0, show_sign=True) 
ax_main3.xaxis.set_major_formatter(fmt)
offset_text3 = ax_main3.get_xaxis().get_offset_text()
offset_text3.set_fontproperties(prop)
offset_text3.set_fontweight('normal')
offset_text3.set_fontstyle('normal')

ax_main3.set_yticks([0, 0.01,0.02])
ax_main3.set_yticklabels(["0", "0.01", "0.02"])
#ax_main.grid(True)
# Create inset plot for differences
ax_inset = ax_main3.inset_axes(INSET_POS_BOTTOM)  # [left, bottom, width, height]
min_len = min(len(zeros_t_acc), len(zeros_t_app))
ax_inset.plot(
    range(1, min_len + 1), diff,
    marker='o', markersize=MARKER_SIZE_INSET, markeredgewidth=0,
    linestyle='-', linewidth=LINE_WIDTH_INSET, color=color3
)
ax_inset.set_xlabel("Zero Index", fontsize=FS_INSET_XLABEL, labelpad=INSET_XLABEL_PAD)
#ax_inset.set_ylabel(r'|t_\mathrm{acc} - t_\mathrm{app}|', fontsize=6)
ax_inset.set_ylabel(r'$\delta t$', fontsize=FS_INSET_LABEL, labelpad=INSET_YLABEL_PAD)
style_inset_axis(ax_inset)
ax_inset.set_yticks([-0.0001, 0, 0.0001])

# Set the x-tick labels using LaTeX for scientific notation
ax_inset.set_yticklabels(["-1e-4", "0", "1e-4"])

plt.subplots_adjust(left=0.070,   # 图像左边距（0~1之间，越大间距越大）
                    right=0.995,  # 图像右边距（0~1之间，越小间距越大）
                    top=0.955,    # 图像上边距（0~1之间，越小间距越大）
                    bottom=0.085,
                    wspace=0.18,
                    hspace=0.28)
# Save the plot
#plt.show()
plt.savefig('Fig2.pdf', bbox_inches="tight", pad_inches=0.01)