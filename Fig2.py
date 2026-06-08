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


plt.rcParams["text.usetex"] = True

# Font sizes: use normal (non-bold) text, with slightly larger sizes for readability.
FS_LABEL = 20
FS_TICK = 19
FS_LEGEND = 16
FS_PANEL = 20
FS_INSET_LABEL = 13
FS_INSET_TICK = 12.5
FS_OFFSET = 18

# Normal-weight font for offset text such as +6,595,000.
prop = FontProperties(size=FS_OFFSET, weight='normal', style='normal')

# Do not load bm or use \boldmath; mathematical labels should remain normal weight.
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
fig = plt.figure(figsize=(14,6))
color1,color2,color3=["#D4562E","#4485C7","#682487"]
gs = plt.GridSpec(2, 2)
ax_main = fig.add_subplot(gs[0, 0])
data=np.load("GLA1.npz", allow_pickle=True)
Z_acc=data['Z_acc']
Z_app=data['Z_app']
zeros_t_acc=data['zeros_t_acc']
zeros_t_app=data['zeros_t_app']
zero_diffs=data['zero_diffs']
t_vals = np.linspace(420, 450, 1000000)
plt.plot(t_vals, Z_app, label=r'$\mathcal{G}(\frac{1}{2},t)$', color=color2,linewidth=2.4)
plt.plot(t_vals, Z_acc, linestyle=(0, (5, 4)),label=r'$\frac{Z\left(\frac{1}{2}+it\right)}{2{Z}\left(\frac{1}{2},H_0\right)}$', color=color1,linewidth=2.4)
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
        linewidth=1.3,                  # 稍加粗
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
        linewidth=1.3,
        zorder=3
    )

ax_main.set_ylim(ymin,ymax)
#ax_main.set_title("Riemann Z-function Z(t) on Re(s)=1/2, 420 ≤ t ≤ 460")
#ax_main.set_title(r'Riemann $Z$-function $Z(t)$ on $\Re(s)=1/2$, $420 \leq t \leq 460$')
ax_main.set_xlabel(r'$t$',fontsize=FS_LABEL, labelpad=2)
ax_main.set_ylabel('GLA',fontsize=FS_LABEL, labelpad=2)
plt.legend(fontsize=FS_LEGEND,frameon=False,bbox_to_anchor=(0.22, 0.89), loc='center',ncol=2,columnspacing=0.5)
ax_main.set_xlim(419.5, 450.5)
ax_main.set_title(r'\textbf{a}', x=-0.04, y=1.005, fontsize=FS_PANEL, fontweight='bold')
ax_main.tick_params(axis='both', which='major', labelsize=FS_TICK)
ax_main.spines['right'].set_visible(False)
ax_main.spines['top'].set_visible(False)
#ax_main.grid(True)
ax_main.set_yticks([-0.5, 0, 0.5])
ax_main.set_yticklabels([r'$-0.5$', r'$0$', r'$0.5$'])
ax_main.axhline(y=0,linewidth=2,linestyle="--",color='#7A7A7A')
# Create inset plot for differences
ax_inset = ax_main.inset_axes([0.74, 0.73, 0.25, 0.25])  # [left, bottom, width, height]
min_len = min(len(zeros_t_acc), len(zeros_t_app))
ax_inset.plot(range(1, min_len + 1), zero_diffs, marker='.', linestyle='-', color=color3)
ax_inset.set_xlabel("Zero Index", fontsize=FS_INSET_LABEL, labelpad=2)
ax_inset.set_ylabel(r'$\delta t$', fontsize=FS_INSET_LABEL)
ax_inset.tick_params(axis='both', which='major', labelsize=FS_INSET_TICK)
ax_inset.spines['right'].set_visible(False)
ax_inset.spines['top'].set_visible(False)
ax_inset.set_yticks([-0.1, 0, 0.1])
ax_inset.set_yticklabels([r'$-0.1$', r'$0$', r'$0.1$'])

ax_main2 = fig.add_subplot(gs[0, 1])
data=np.load("GLA2.npz", allow_pickle=True)
Z_acc=data['Z_acc']
Z_app=data['Z_app']
zeros_t_acc=data['zeros_t_acc']
zeros_t_app=data['zeros_t_app']
zero_diffs=data['zero_diffs']
ax_main2.set_xlim(6595000-0.2,6595010.2)
t_vals = np.linspace(6595000, 6595010, 1000000)
plt.plot(t_vals, Z_app, label=r'$\mathcal{G}(\frac{1}{2},t)$', color=color2,linewidth=2.4)
plt.plot(t_vals, Z_acc, linestyle=(0, (5, 4)), label=r'$\frac{Z\left(\frac{1}{2}+it\right)}{2{Z}\left(\frac{1}{2},H_0\right)}$', color=color1,linewidth=2.4)
ymin, ymax = ax_main2.get_ylim()
for z in zeros_t_acc:
    ax_main2.vlines(
        z,
        ymin=ymin,
        ymax=0,
        color=color1,                # 深红，物理期刊常用
        linestyle=(0, (5, 3)),          # 长虚线
        alpha=0.6,
        linewidth=1.3,                  # 稍加粗
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
        linewidth=1.3,
        zorder=3
    )
ax_main2.set_ylim(ymin,ymax)
#ax_main.set_title("Riemann Z-function Z(t) on Re(s)=1/2, 420 ≤ t ≤ 460")
#ax_main.set_title(r'Riemann $Z$-function $Z(t)$ on $\Re(s)=1/2$, $420 \leq t \leq 460$')
ax_main2.set_xlabel(r'$t$',fontsize=FS_LABEL, labelpad=2)
ax_main2.set_ylabel('GLA',fontsize=FS_LABEL, labelpad=2)
legend=ax_main2.legend(fontsize=FS_LEGEND,frameon=False,bbox_to_anchor=(0.33, 0.86), loc='center',ncol=2,columnspacing=0.5,facecolor='white')
ax_main2.set_title(r'\textbf{b}', x=-0.04, y=1.005, fontsize=FS_PANEL, fontweight='bold')
ax_main2.tick_params(axis='both', which='major', labelsize=FS_TICK)
ax_main2.spines['right'].set_visible(False)
ax_main2.spines['top'].set_visible(False)
ax_main2.set_yticks([0, 0.05])
ax_main2.set_yticklabels([r'$0$', r'$0.05$'])
fmt = CommaOffsetOnly(decimals=0, show_sign=True) 
ax_main2.xaxis.set_major_formatter(fmt)
offset_text2 = ax_main2.get_xaxis().get_offset_text()
offset_text2.set_fontproperties(prop)
offset_text2.set_fontweight('normal')
offset_text2.set_fontstyle('normal')

#ax_main.grid(True)
plt.axhline(y=0,linewidth=2,linestyle="--",color='#7A7A7A')
# Create inset plot for differences
ax_inset = ax_main2.inset_axes([0.74, 0.73, 0.25, 0.25])  # [left, bottom, width, height]
min_len = min(len(zeros_t_acc), len(zeros_t_app))
ax_inset.plot(range(1, min_len + 1), zero_diffs, marker='.', linestyle='-', color=color3)
ax_inset.set_xlabel("Zero Index", fontsize=FS_INSET_LABEL, labelpad=2)
#ax_inset.set_ylabel(r'|t_\mathrm{acc} - t_\mathrm{app}|', fontsize=6)
ax_inset.set_ylabel(r'$\delta t$', fontsize=FS_INSET_LABEL)
ax_inset.tick_params(axis='both', which='major', labelsize=FS_INSET_TICK)
ax_inset.set_yticks([-0.002, 0,0.002])
ax_inset.set_yticklabels([r'$-2\times 10^{-3}$', r'$0$', r'$2\times10^{-3}$'])
ax_inset.spines['right'].set_visible(False)
ax_inset.spines['top'].set_visible(False)

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
plt.plot(t_vals, Z_app, label=r'$\mathcal{G}(\frac{1}{2},t)$', color=color2,linewidth=2.4)
plt.plot(t_vals, Z_acc, linestyle=(0, (5, 4)), label=r'$\frac{Z\left(\frac{1}{2}+it\right)}{2{Z}\left(\frac{1}{2},H_0\right)}$', color=color1,linewidth=2.4)
ymin, ymax = ax_main3.get_ylim()
for z in zeros_t_acc:
    ax_main3.vlines(
        z,
        ymin=ymin,
        ymax=0,
        color=color1,                # 深红，物理期刊常用
        linestyle=(0, (5, 3)),          # 长虚线
        alpha=0.6,
        linewidth=1.3,                  # 稍加粗
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
        linewidth=1.3,
        zorder=3
    )
ax_main3.set_ylim(ymin,ymax)
#ax_main.set_title("Riemann Z-function Z(t) on Re(s)=1/2, 420 ≤ t ≤ 460")
#ax_main.set_title(r'Riemann $Z$-function $Z(t)$ on $\Re(s)=1/2$, $420 \leq t \leq 460$')
ax_main3.set_xlabel(r'$t$',fontsize=FS_LABEL, labelpad=2)
ax_main3.set_ylabel('GLA',fontsize=FS_LABEL, labelpad=2)
ax_main3.legend(fontsize=FS_LEGEND,frameon=False,bbox_to_anchor=(0.35, 0.85), loc='center',ncol=2,columnspacing=1)
ax_main3.set_title(r'\textbf{c}', x=-0.02, y=1.005, fontsize=FS_PANEL, fontweight='bold')
ax_main3.tick_params(axis='both', which='major', labelsize=FS_TICK)
ax_main3.spines['right'].set_visible(False)
ax_main3.spines['top'].set_visible(False)

fmt = CommaOffsetOnly(decimals=0, show_sign=True) 
ax_main3.xaxis.set_major_formatter(fmt)
offset_text3 = ax_main3.get_xaxis().get_offset_text()
offset_text3.set_fontproperties(prop)
offset_text3.set_fontweight('normal')
offset_text3.set_fontstyle('normal')

ax_main3.set_yticks([0, 0.01,0.02])
ax_main3.set_yticklabels([r'$0$',r'$0.01$', r'$0.02$'])
#ax_main.grid(True)
plt.axhline(y=0,linewidth=2,linestyle="--",color='#7A7A7A')
# Create inset plot for differences
ax_inset = ax_main3.inset_axes([0.745, 0.53, 0.25, 0.45])  # [left, bottom, width, height]
min_len = min(len(zeros_t_acc), len(zeros_t_app))
ax_inset.plot(range(1, min_len + 1), diff, marker='.', linestyle='-', color=color3)
ax_inset.set_xlabel("Zero Index", fontsize=FS_INSET_LABEL, labelpad=2)
#ax_inset.set_ylabel(r'|t_\mathrm{acc} - t_\mathrm{app}|', fontsize=6)
ax_inset.set_ylabel(r'$\delta t$', fontsize=FS_INSET_LABEL)
ax_inset.tick_params(axis='both', which='major', labelsize=FS_INSET_TICK)
ax_inset.set_yticks([-0.0001, 0, 0.0001])
ax_inset.spines['right'].set_visible(False)
ax_inset.spines['top'].set_visible(False)

# Set the x-tick labels using LaTeX for scientific notation
ax_inset.set_yticklabels([r'$-10^{-4}$', r'$0$', r'$10^{-4}$'])

plt.subplots_adjust(left=0.065,   # 图像左边距（0~1之间，越大间距越大）
                    right=0.995,  # 图像右边距（0~1之间，越小间距越大）
                    top=0.947,    # 图像上边距（0~1之间，越小间距越大）
                    bottom=0.095,
                    wspace=0.15,
                    hspace=0.25)
# Save the plot
#plt.show()
plt.savefig('Fig2.pdf')