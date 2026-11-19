"""Schematic overview of the kinematic feature groups (synthetic signals).

Panels follow the order and naming of the Methods section:
1 Magnitude, 2 Variability, 3 Rhythm, 4 Coordination,
5 Proximal-distal, 6 Smoothness, 7 Handshape.

Usage in a notebook:
    from feature_overview import plot_feature_overview
    fig = plot_feature_overview(out="feature_overview.pdf")   # or .png
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

INK, ACC, ACC2, GREY = "#333333", "#17800E", "#6D0E0E", "#BBBBBB"


def _title(ax, num, name, sub):
    ax.set_title(f"{num}  {name}\n", loc="left", fontweight="bold", fontsize=9, pad=2)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=7.5, color="0.35", va="bottom")


def _speed(t, peaks):
    return sum(a * np.exp(-((t - m) / w) ** 2) for a, m, w in peaks) + 0.06


def _P(a):
    return np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])


def plot_feature_overview(out=None):
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig = plt.figure(figsize=(13, 6.8))
    gs = fig.add_gridspec(2, 4, hspace=0.6, wspace=0.3)
    t = np.linspace(0, 2, 600)
    PEAKS = [(1.0, .35, .11), (.65, .85, .09), (.85, 1.35, .10), (.45, 1.75, .08)]
    sp = _speed(t, PEAKS)
    pk = [p[1] for p in PEAKS]
    ph = [np.interp(p, t, sp) for p in pk]

    # 1 Magnitude --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, sp, color=INK, lw=1.6)
    ax.plot([0, 2.0], [sp.mean()]*2, color="0.45", lw=1, ls=":")
    ax.text(2.05, sp.mean(), "global\nmean speed", fontsize=7, color="0.4", va="center")
    ax.scatter(pk, ph, color=ACC, s=16, zorder=3)
    ax.plot([0, 2.0], [np.mean(ph)]*2, color=ACC, lw=.9, ls="--")
    ax.text(2.05, np.mean(ph), "mean peak\nspeed", fontsize=7, color=ACC, va="center")
    # area under the speed curve = distance travelled (path length)
    ax.fill_between(t, sp, color=ACC2, alpha=.18, lw=0)
    ax.text(2.05, 1.0, "shaded area\n= distance\ntravelled", fontsize=7, color=ACC2, va="center")
    ax.set_ylim(0, 1.2); ax.set_xlim(0, 2.75); ax.set_xticks([0, 1, 2]); ax.set_yticks([]); ax.set_xlabel("time (s)"); ax.set_ylabel("arm speed")
    _title(ax, "1", "Magnitude", "size and vigour of the movement")

    # 2 Variability ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.axhspan(sp.mean() - sp.std(), sp.mean() + sp.std(), color=GREY, alpha=.3, lw=0)
    ax.plot(t, sp, color=INK, lw=1.6)
    ax.text(2.05, .12, "SD of\nspeed", fontsize=7, color="0.4")
    ax.scatter(pk, ph, color=ACC, s=16, zorder=3)
    ax.annotate("", xy=(2.3, max(ph)), xytext=(2.3, min(ph)),
                arrowprops=dict(arrowstyle="<->", color=ACC, lw=1.2))
    for h in (max(ph), min(ph)):
        ax.plot([0.3, 2.35], [h, h], color=ACC, lw=.6, ls=":")
    ax.text(2.38, (max(ph) + min(ph)) / 2, "SD of\npeak\nheight", fontsize=7, color=ACC, va="center")
    ax.set_ylim(0, 1.2); ax.set_xlim(0, 2.95); ax.set_xticks([0, 1, 2]); ax.set_yticks([]); ax.set_xlabel("time (s)"); ax.set_ylabel("arm speed")
    _title(ax, "2", "Variability", "how much the movement varies within an item")

    # 3 Rhythm (two panels) ------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, sp, color=INK, lw=1.6)
    for p in pk:
        ax.plot([p, p], [-.12, -.04], color=ACC, lw=1.4)
    ax.text(1, -.22, "peaks per second", ha="center", color=ACC, fontsize=7)
    tr = np.interp(.6, t, sp)
    ax.annotate("", xy=(.6, tr), xytext=(.6, ph[0]), arrowprops=dict(arrowstyle="<->", color=ACC2, lw=1.2))
    ax.text(.68, 1.02, "modulation depth\n(peak − trough)", color=ACC2, fontsize=7)
    ax.set_ylim(-.28, 1.35); ax.set_yticks([]); ax.set_xlabel("time (s)"); ax.set_ylabel("arm speed")
    _title(ax, "3", "Rhythm", "temporal structure: peaks …")

    ax = fig.add_subplot(gs[0, 3])
    f = np.linspace(.5, 6, 500)
    P = np.exp(-((f - 1.4) / .7) ** 2) + .45 * np.exp(-((f - 3.2) / .6) ** 2) + .04
    for (lo, hi, l), c in zip([(1, 2, "slow\n1–2 Hz"), (2, 3, "medium\n2–3 Hz"), (3, 5, "fast\n3–5 Hz")],
                              ["#ECECEC", "#DCDCDC", "#CCCCCC"]):
        ax.axvspan(lo, hi, color=c, lw=0)
        ax.text((lo + hi) / 2, 1.13, l, ha="center", va="center", fontsize=7)
    ax.plot(f, P, color=INK, lw=1.6)
    m = (f >= 3) & (f <= 5)
    cen = (f[m] * P[m]).sum() / P[m].sum()
    ax.fill_between(f[m], P[m], color=ACC, alpha=.25, lw=0)
    ax.plot([cen, cen], [0, .95], color=ACC, lw=1, ls="--")
    ax.text(cen + .08, .8, "centroid", color=ACC, fontsize=7)
    ax.text(4.5, .24, "band\npower", color=ACC, fontsize=7, ha="center")
    ax.set_ylim(0, 1.25); ax.set_xlim(.5, 6); ax.set_yticks([])
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("power")
    _title(ax, "", "", "… and rhythmic ranges (power, centroid, spread)")

    # 4 Coordination -----------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    tt = np.linspace(0, 2, 400)
    ax.plot(tt, np.sin(2 * np.pi * tt) + 2.6, color=ACC, lw=1.6)
    ax.plot(tt, np.sin(2 * np.pi * tt + .2) + 2.6, color=INK, lw=1.2, ls="--")
    ax.plot(tt, np.sin(2 * np.pi * tt), color=ACC, lw=1.6)
    ax.plot(tt, .5 * np.sin(2 * np.pi * 1.7 * tt + 1), color=INK, lw=1.2, ls="--")
    ax.text(2.12, 2.6, "coupled", fontsize=7, va="center")
    ax.text(2.12, 0, "uncoupled", fontsize=7, va="center")
    ax.plot([], [], color=ACC, label="left wrist"); ax.plot([], [], color=INK, ls="--", label="right wrist")
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    ax.set_xlim(0, 2.7); ax.set_ylim(-1.3, 4.6); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("time (s)"); ax.set_ylabel("vertical position")
    _title(ax, "4", "Coordination", "how body parts move in relation to each other")

    # 5 Proximal-distal ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 1]); ax.set_aspect("equal"); ax.axis("off")

    def arm(S, a1, a2, a3, c, ls="-", lw=2.4):
        S = np.array(S); E = S + _P(a1); W = E + .85 * _P(a2); H = W + .3 * _P(a3)
        ax.plot(*zip(S, E, W), color=c, lw=lw, ls=ls, solid_capstyle="round")
        ax.plot(*zip(W, H), color=c, lw=lw + 1.5, ls=ls, solid_capstyle="round")
        ax.scatter(*np.array([S, E, W]).T, color=c, s=14, zorder=3)
        return W

    arm((0, 0), -95, -35, -35, GREY, "--"); arm((0, 0), -60, 0, 0, ACC)
    ax.add_patch(FancyArrowPatch((0.25, -1.05), (0.75, -0.75), connectionstyle="arc3,rad=.35",
                                 arrowstyle="->", mutation_scale=10, color=ACC))
    ax.text(.3, -2.45, "proximal: the arm is\ncarried from the shoulder", fontsize=7, color=ACC, ha="center")
    arm((2.6, 0), -60, 0, -60, GREY, "--"); W = arm((2.6, 0), -60, 0, 40, ACC2)
    ax.add_patch(FancyArrowPatch((W[0] + .25, W[1] - .35), (W[0] + .3, W[1] + .25),
                                 connectionstyle="arc3,rad=.45", arrowstyle="->", mutation_scale=10, color=ACC2))
    ax.text(3.0, -2.45, "distal: the joint\nmoves on its own", fontsize=7, color=ACC2, ha="center")
    for x0 in (0, 2.6):
        ax.text(x0, .25, "shoulder", fontsize=7, color="0.4", ha="center")
    ax.set_xlim(-.6, 4.4); ax.set_ylim(-2.85, .6)
    _title(ax, "5", "Proximal–distal", "which joints the movement comes from")

    # 6 Smoothness ---------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    t1 = np.linspace(0, 1, 500)
    ax.plot(t1, np.exp(-((t1 - .5) / .18) ** 2), color=ACC, lw=1.8, label="smooth: one submovement")
    ax.plot(t1, sum(a * np.exp(-((t1 - m) / w) ** 2) for a, m, w in [(.75, .25, .07), (.9, .48, .07), (.6, .72, .06)]),
            color=INK, lw=1.6, label="intermittent: several submovements")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_ylim(0, 1.55); ax.set_yticks([]); ax.set_xlabel("time (s)"); ax.set_ylabel("speed")
    _title(ax, "6", "Smoothness", "continuous vs stop-and-go movement")

    # 7 Handshape ----------------------------------------------------------
    ax = fig.add_subplot(gs[1, 3]); ax.set_aspect("equal"); ax.axis("off")
    wrist = np.array([0, 0])
    mcp = np.array([[-.55, .55], [-.3, 1.0], [0, 1.08], [.28, 1.02], [.52, .88]])
    ang, L = [125, 100, 90, 80, 62], [.75, 1.0, 1.08, .98, .78]
    tips = np.array([m + l * _P(a) for m, a, l in zip(mcp, ang, L)])
    ax.add_patch(plt.Polygon([[-.6, .5], [-.35, 1.05], [.55, .95], [.55, .1], [-.35, -.05]], color="#EEEEEE"))
    for m, tp in zip(mcp, tips):
        ax.plot(*zip(wrist, m), color=GREY, lw=1)
        ax.plot(*zip(m, tp), color=INK, lw=2.2, solid_capstyle="round")
    ax.scatter(*tips.T, color=INK, s=12, zorder=3)
    ax.scatter(*mcp.T, color="white", edgecolor=INK, s=14, zorder=3)
    ax.scatter(*wrist, color=INK, s=16)
    ax.annotate("", xy=tips[0], xytext=tips[1], arrowprops=dict(arrowstyle="<->", color=ACC, lw=1.2))
    ax.text(-1.0, 1.55, "thumb–index\ndistance", color=ACC, fontsize=7, ha="right")
    for i in range(1, 4):
        ax.plot(*zip(tips[i], tips[i + 1]), color=ACC2, lw=1, ls=":")
    ax.text(.75, 2.05, "fingertip spread", color=ACC2, fontsize=7)
    ax.plot(*zip(tips[0], mcp[4]), color="0.5", lw=1, ls="--")
    ax.text(-1.75, -.15, "thumb opposition\n(thumb tip –\nlittle-finger base)", fontsize=6.5, color="0.4")
    ax.annotate("", xy=tips[4], xytext=mcp[4], arrowprops=dict(arrowstyle="<->", color="0.3", lw=1))
    ax.text(1.05, 1.2, "knuckle–tip\n(extension)", fontsize=7, color="0.35")
    ax.text(.1, -.25, "wrist", fontsize=7, color="0.35")
    ax.set_xlim(-1.7, 1.9); ax.set_ylim(-.4, 2.3)
    _title(ax, "7", "Handshape", "shape of the hand, averaged over the item")

    if out:
        fig.savefig(out, dpi=300, bbox_inches="tight")
    return fig
