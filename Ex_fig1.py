import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties

plt.rcParams['axes.formatter.use_locale'] = True
plt.rcParams.update({
    'xtick.direction': 'in',          # Tick marks point inward
    'ytick.direction': 'in'           # Tick marks point inward
})

class CommaOffsetOnly(mticker.ScalarFormatter):
    """Only override the offset text with comma grouping; keep the main tick labels using the default ScalarFormatter behavior."""
    def __init__(self, decimals=0, show_sign=False, **kwargs):
        super().__init__(useOffset=True, **kwargs)
        self.set_scientific(False)    # Disable scientific notation
        self.set_powerlimits((0, 0))  # Equivalent to style='plain'
        self.decimals = decimals
        self.show_sign = show_sign

    def __call__(self, x, pos=None):
        # Main tick labels: fully inherit the parent class behavior
        return super().__call__(x, pos)

    def get_offset(self):
        off = getattr(self, 'offset', 0.0)
        if not off:
            return ''
        # Integer -> thousands separator; otherwise keep the specified decimal places
        if self.decimals == 0 and float(off).is_integer():
            s = f'{int(off):,}'
        else:
            s = f'{off:,.{self.decimals}f}'
        if self.show_sign and off > 0:
            s = '+' + s
        return s


plt.rcParams["text.usetex"] = True
prop = FontProperties(family='Computer Modern:bold', size=16, weight='bold')
# mpl.use('pgf')

# Set the LaTeX preamble to use XeLaTeX with Arial font
mpl.rcParams['text.latex.preamble'] = r"""
\usepackage{amsmath}
\usepackage{fontspec}
\setmainfont{Arial}
"""

# mpl.rcParams['text.latex.preamble'] = r"""
# \usepackage{amsmath}
# \usepackage{fontspec}
# \setmainfont{Arial} % Use Arial for text
# \renewcommand{\rmdefault}{phv}  % Arial font for math
# """

plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, bm} \boldmath"
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'       # Set regular text in math expressions to Arial
mpl.rcParams['mathtext.it'] = 'Computer Modern:italic'  # Set italic math text to Computer Modern italic
mpl.rcParams['mathtext.bf'] = 'Computer Modern:bold'

plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

fig = plt.figure(figsize=(14, 6))
color1, color2, color3 = ["#D4562E", "#4485C7", "#682487"]
gs = plt.GridSpec(2, 2)

ax_main = fig.add_subplot(gs[0, 0])

data = np.load("GLA1.npz", allow_pickle=True)
Z_acc = data['Z_acc']
Z_app = data['Z_app']
zeros_t_acc = data['zeros_t_acc']
zeros_t_app = data['zeros_t_app']
zero_diffs = data['zero_diffs']

t_vals = np.linspace(420, 450, 1000000)
plt.plot(t_vals, Z_app, label=r'$\mathcal{G} (\frac{1}{2},t)$', color=color2, linewidth=2.4)
plt.plot(t_vals, Z_acc, linestyle=(0, (5, 4)), label=r'$\frac{Z\left(\frac{1}{2}+it\right)}{2{Z}\left(\frac{1}{2},H_0\right)}$', color=color1, linewidth=2.4)

ymin, ymax = ax_main.get_ylim()

# for z in zeros_t_acc:
#     ax_main.vlines(
#     z,
#     ymin=ymin,  # bottom of the plot
#     ymax=0,     # up to y = 0
#     color=color1,
#     linestyle=(0, (3, 3)),
#     alpha=0.5,
#     linewidth=1.5
#     )
#     # ax_main3.axvline(z, color=color1, linestyle=":", alpha=0.5, linewidth=2, ymax=0)

# for z in zeros_t_app:
#     # ax_main3.axvline(z, color=color2, linestyle="-.", alpha=0.5, linewidth=2, ymax=0)
#     ax_main.vlines(
#     z,
#     ymin=ymin,  # bottom of the plot
#     ymax=0,     # up to y = 0
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
        color=color1,                # Dark red, commonly used in physics journals
        linestyle=(0, (5, 3)),       # Long dashed line
        alpha=0.6,
        linewidth=1.3,               # Slightly thicker
        zorder=2
    )

for z in zeros_t_app:
    ax_main.vlines(
        z,
        ymin=ymin,
        ymax=0,
        color=color2,                # Dark blue, stable and not overly harsh
        linestyle=(0, (3, 2, 1, 2)), # Dash-dot style
        alpha=0.8,
        linewidth=1.3,
        zorder=3
    )

ax_main.set_ylim(ymin, ymax)
# ax_main.set_title("Riemann Z-function Z(t) on Re(s)=1/2, 420 ≤ t ≤ 460")
# ax_main.set_title(r'Riemann $Z$-function $Z(t)$ on $\Re(s)=1/2$, $420 \leq t \leq 460$')
ax_main.set_xlabel(r'$t$', fontsize=18, labelpad=2)
ax_main.set_ylabel('GLA', fontsize=18, labelpad=2)
plt.legend(fontsize=14.5, frameon=False, bbox_to_anchor=(0.22, 0.89), loc='center', ncol=2, columnspacing=0.5)
ax_main.set_xlim(419.5, 450.5)
ax_main.set_title('a', x=-0.04, y=1.005, fontsize=22)
ax_main.tick_params(axis='both', which='major', labelsize=18)
ax_main.spines['right'].set_visible(False)
ax_main.spines['top'].set_visible(False)
# ax_main.grid(True)
ax_main.set_yticks([-0.5, 0, 0.5])
ax_main.set_yticklabels([r'$-0.5$', r'$0$', r'$0.5$'])
ax_main.axhline(y=0, linewidth=2, linestyle="--", color='#7A7A7A')

# Create inset plot for differences
ax_inset = ax_main.inset_axes([0.74, 0.73, 0.25, 0.25])  # [left, bottom, width, height]
min_len = min(len(zeros_t_acc), len(zeros_t_app))
ax_inset.plot(range(1, min_len + 1), zero_diffs, marker='.', linestyle='-', color=color3)
ax_inset.set_xlabel("Zero Index", fontsize=11.5, labelpad=2)
ax_inset.set_ylabel(r'$\delta t$', fontsize=11.5)
ax_inset.tick_params(axis='both', which='major', labelsize=11.5)
ax_inset.spines['right'].set_visible(False)
ax_inset.spines['top'].set_visible(False)
ax_inset.set_yticks([-0.1, 0, 0.1])
ax_inset.set_yticklabels([r'$-0.1$', r'$0$', r'$0.1$'])

ax_main2 = fig.add_subplot(gs[0, 1])

data = np.load("GLA2.npz", allow_pickle=True)
Z_acc = data['Z_acc']
Z_app = data['Z_app']
zeros_t_acc = data['zeros_t_acc']
zeros_t_app = data['zeros_t_app']
zero_diffs = data['zero_diffs']

ax_main2.set_xlim(6595000 - 0.2, 6595010.2)
t_vals = np.linspace(6595000, 6595010, 1000000)
plt.plot(t_vals, Z_app, label=r'$\mathcal{G} (\frac{1}{2},t)$', color=color2, linewidth=2.4)
plt.plot(t_vals, Z_acc, linestyle=(0, (5, 4)), label=r'$\frac{Z\left(\frac{1}{2}+it\right)}{2{Z}\left(\frac{1}{2},H_0\right)}$', color=color1, linewidth=2.4)

ymin, ymax = ax_main2.get_ylim()

for z in zeros_t_acc:
    ax_main2.vlines(
        z,
        ymin=ymin,
        ymax=0,
        color=color1,                # Dark red, commonly used in physics journals
        linestyle=(0, (5, 3)),       # Long dashed line
        alpha=0.6,
        linewidth=1.3,               # Slightly thicker
        zorder=2
    )

for z in zeros_t_app:
    ax_main2.vlines(
        z,
        ymin=ymin,
        ymax=0,
        color=color2,                # Dark blue, stable and not overly harsh
        linestyle=(0, (3, 2, 1, 2)), # Dash-dot style
        alpha=0.8,
        linewidth=1.3,
        zorder=3
    )

ax_main2.set_ylim(ymin, ymax)
# ax_main.set_title("Riemann Z-function Z(t) on Re(s)=1/2, 420 ≤ t ≤ 460")
# ax_main.set_title(r'Riemann $Z$-function $Z(t)$ on $\Re(s)=1/2$, $420 \leq t \leq 460$')
ax_main2.set_xlabel(r'$t$', fontsize=18, labelpad=2)
ax_main2.set_ylabel('GLA', fontsize=18, labelpad=2)
legend = ax_main2.legend(fontsize=14.5, frameon=False, bbox_to_anchor=(0.33, 0.86), loc='center', ncol=2, columnspacing=0.5, facecolor='white')
ax_main2.set_title('b', x=-0.04, y=1.005, fontsize=22)
ax_main2.tick_params(axis='both', which='major', labelsize=18)
ax_main2.spines['right'].set_visible(False)
ax_main2.spines['top'].set_visible(False)
ax_main2.set_yticks([0, 0.05])
ax_main2.set_yticklabels([r'$0$', r'$0.05$'])

fmt = CommaOffsetOnly(decimals=0, show_sign=True)
ax_main2.xaxis.set_major_formatter(fmt)
ax_main2.get_xaxis().get_offset_text().set_fontproperties(prop)

# ax_main.grid(True)
plt.axhline(y=0, linewidth=2, linestyle="--", color='#7A7A7A')

# Create inset plot for differences
ax_inset = ax_main2.inset_axes([0.74, 0.73, 0.25, 0.25])  # [left, bottom, width, height]
min_len = min(len(zeros_t_acc), len(zeros_t_app))
ax_inset.plot(range(1, min_len + 1), zero_diffs, marker='.', linestyle='-', color=color3)
ax_inset.set_xlabel("Zero Index", fontsize=11.5, labelpad=2)
# ax_inset.set_ylabel(r'|t_\mathrm{acc} - t_\mathrm{app}|', fontsize=6)
ax_inset.set_ylabel(r'$\delta t$', fontsize=11.5)
ax_inset.tick_params(axis='both', which='major', labelsize=11.5)
ax_inset.set_yticks([-0.002, 0, 0.002])
ax_inset.set_yticklabels([r'$-2\times 10^{-3}$', r'$0$', r'$2\times10^{-3}$'])
ax_inset.spines['right'].set_visible(False)
ax_inset.spines['top'].set_visible(False)

ax_main3 = fig.add_subplot(gs[1, :])

data = np.load("GLA3.npz", allow_pickle=True)
Z_acc = data['Z_acc']
Z_app = data['Z_app']
zeros_t_acc = data['zeros_t_acc']
zeros_t_app = data['zeros_t_app']
# print(len(zeros_t_acc), len(zeros_t_app))
# zero_diffs = data['zero_diffs']

diff = []
zeros_t_appfix = []

for i in zeros_t_acc:
    idx = np.argmin(np.abs(zeros_t_app - i))
    diff.append(i - zeros_t_app[idx])
    zeros_t_appfix.append(zeros_t_app[idx])

ax_main3.set_xlim(267653395648 - 0.2, 267653395648 + 12.2)
t_vals = np.linspace(267653395648, 267653395648 + 12, 1000000)
plt.plot(t_vals, Z_app, label=r'$\mathcal{G} (\frac{1}{2},t)$', color=color2, linewidth=2.4)
plt.plot(t_vals, Z_acc, linestyle=(0, (5, 4)), label=r'$\frac{Z\left(\frac{1}{2}+it\right)}{2{Z}\left(\frac{1}{2},H_0\right)}$', color=color1, linewidth=2.4)

ymin, ymax = ax_main3.get_ylim()

for z in zeros_t_acc:
    ax_main3.vlines(
        z,
        ymin=ymin,
        ymax=0,
        color=color1,                # Dark red, commonly used in physics journals
        linestyle=(0, (5, 3)),       # Long dashed line
        alpha=0.6,
        linewidth=1.3,               # Slightly thicker
        zorder=2
    )

for z in zeros_t_appfix:
    ax_main3.vlines(
        z,
        ymin=ymin,
        ymax=0,
        color=color2,                # Dark blue, stable and not overly harsh
        linestyle=(0, (3, 2, 1, 2)), # Dash-dot style
        alpha=0.8,
        linewidth=1.3,
        zorder=3
    )

ax_main3.set_ylim(ymin, ymax)
# ax_main.set_title("Riemann Z-function Z(t) on Re(s)=1/2, 420 ≤ t ≤ 460")
# ax_main.set_title(r'Riemann $Z$-function $Z(t)$ on $\Re(s)=1/2$, $420 \leq t \leq 460$')
ax_main3.set_xlabel(r'$t$', fontsize=18, labelpad=2)
ax_main3.set_ylabel('GLA', fontsize=18, labelpad=2)
ax_main3.legend(fontsize=14.5, frameon=False, bbox_to_anchor=(0.35, 0.85), loc='center', ncol=2, columnspacing=1)
ax_main3.set_title('c', x=-0.02, y=1.005, fontsize=22)
ax_main3.tick_params(axis='both', which='major', labelsize=18)
ax_main3.spines['right'].set_visible(False)
ax_main3.spines['top'].set_visible(False)

fmt = CommaOffsetOnly(decimals=0, show_sign=True)
ax_main3.xaxis.set_major_formatter(fmt)
ax_main3.get_xaxis().get_offset_text().set_fontproperties(prop)

ax_main3.set_yticks([0, 0.01, 0.02])
ax_main3.set_yticklabels([r'$0$', r'$0.01$', r'$0.02$'])
# ax_main.grid(True)
plt.axhline(y=0, linewidth=2, linestyle="--", color='#7A7A7A')

# Create inset plot for differences
ax_inset = ax_main3.inset_axes([0.745, 0.53, 0.25, 0.45])  # [left, bottom, width, height]
min_len = min(len(zeros_t_acc), len(zeros_t_app))
ax_inset.plot(range(1, min_len + 1), diff, marker='.', linestyle='-', color=color3)
ax_inset.set_xlabel("Zero Index", fontsize=11.5, labelpad=2)
# ax_inset.set_ylabel(r'|t_\mathrm{acc} - t_\mathrm{app}|', fontsize=6)
ax_inset.set_ylabel(r'$\delta t$', fontsize=11.5)
ax_inset.tick_params(axis='both', which='major', labelsize=11.5)
ax_inset.set_yticks([-0.0001, 0, 0.0001])
ax_inset.spines['right'].set_visible(False)
ax_inset.spines['top'].set_visible(False)

# Set the y-tick labels using LaTeX for scientific notation
ax_inset.set_yticklabels([r'$-10^{-4}$', r'$0$', r'$10^{-4}$'])

plt.subplots_adjust(
    left=0.065,    # Left margin of the figure (between 0 and 1; larger means more space)
    right=0.995,   # Right margin of the figure (between 0 and 1; smaller means more space)
    top=0.947,     # Top margin of the figure (between 0 and 1; smaller means more space)
    bottom=0.095,
    wspace=0.15,
    hspace=0.25
)

# Save the plot
# plt.show()
plt.savefig('Ex_fig1.pdf')