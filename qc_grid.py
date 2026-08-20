"""
qc_grid.py  —  visual sanity check for processed Sapiens kinematics files.

Draws N random processed skeletons (torso-centred, trunk-normalised) either as
  (a) a synchronised grid ANIMATION  -> one .mp4, or
  (b) a static TRAJECTORY contact sheet -> one .png  (fast first look).

What to look for:
  * every skeleton sits centred on (0,0) with mid-shoulder above mid-hip;
  * the trunk length printed in each title should be ~1.0 after normalisation.
    A title in RED = trunk deviates >30% from 1  -> normalisation/centering
    suspect for that file (wrong per-pcn median, wrong person, bad flip, ...);
  * one skeleton much bigger/smaller than the rest = scale problem;
  * upside-down skeleton = flip problem;
  * skeleton that teleports mid-clip = identity swap / wrong person.

Only gross body keypoints are drawn (fingers live in a wrist-local, palm-scaled
frame, so they don't belong on the same axes).
"""

import os, glob, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # file output, no display needed
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

# ----------------------------------------------------------------------------- config
PROCESSED_DIR = "Sapiens_Processed"          # folder with *_kinematics.csv
PATTERN       = "*_kinematics.csv"
N_FILES       = 50
SEED          = 42
MODE          = "both"                        # "animation" | "contact" | "both"
MAX_FRAMES    = 200                           # cap anim length (resampled per file)
OUT_ANIM      = "qc_grid_animation.mp4"
OUT_CONTACT   = "qc_trajectories.png"
TRUNK_TOL     = 0.30                          # title turns red if |trunk-1| > this
# -----------------------------------------------------------------------------

# skeleton as pairs of keypoint stems; only pairs whose BOTH points exist are drawn
EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("neck", "nose"),
]
POINTS = ["nose", "neck", "left_shoulder", "right_shoulder",
          "left_elbow", "right_elbow", "left_wrist", "right_wrist",
          "left_hip", "right_hip"]

GROUP_COLORS = {}   # filled on the fly

def group_of(fid):
    if "GtST" in fid or "GP" in fid:
        return fid.split("_")[0]
    if "LP" in fid or "IP" in fid:
        return "_".join(fid.split("_")[:2])
    if "ViCom" in fid:
        return "ViCom"
    return "other"

def color_of(group):
    if group not in GROUP_COLORS:
        palette = plt.cm.tab10.colors
        GROUP_COLORS[group] = palette[len(GROUP_COLORS) % len(palette)]
    return GROUP_COLORS[group]

def load_one(path):
    df = pd.read_csv(path)
    fid = str(df["FileID"].iloc[0]) if "FileID" in df.columns else os.path.basename(path)
    # keypoint arrays; NaN where a point is absent
    xy = {}
    for p in POINTS:
        cx, cy = f"{p}_x", f"{p}_y"
        if cx in df.columns and cy in df.columns:
            xy[p] = (df[cx].to_numpy(float), df[cy].to_numpy(float))
    # trunk length in the *normalised* frame — should be ~1.0
    trunk = np.nan
    if all(k in xy for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")):
        msx = 0.5 * (xy["left_shoulder"][0] + xy["right_shoulder"][0])
        msy = 0.5 * (xy["left_shoulder"][1] + xy["right_shoulder"][1])
        mhx = 0.5 * (xy["left_hip"][0] + xy["right_hip"][0])
        mhy = 0.5 * (xy["left_hip"][1] + xy["right_hip"][1])
        trunk = float(np.nanmedian(np.hypot(msx - mhx, msy - mhy)))
    return fid, xy, trunk

def pick_files():
    files = sorted(glob.glob(os.path.join(PROCESSED_DIR, PATTERN)))
    if not files:
        raise SystemExit(f"No files matched {PROCESSED_DIR}/{PATTERN}")
    random.seed(SEED)
    return random.sample(files, min(N_FILES, len(files)))

def shared_limit(samples):
    """symmetric axis limit from a robust percentile of all plotted coords."""
    allv = []
    for _, xy, _ in samples:
        for (x, y) in xy.values():
            allv.append(np.nanpercentile(np.abs(x), 99))
            allv.append(np.nanpercentile(np.abs(y), 99))
    lim = np.nanpercentile(allv, 90) * 1.15
    return max(lim, 1.5)

# ----------------------------------------------------------------------------- contact sheet
def make_contact_sheet(samples, lim):
    n = len(samples)
    ncol = int(np.ceil(np.sqrt(n * 1.4)))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.0, nrow * 2.2))
    axes = np.array(axes).reshape(-1)
    for ax in axes: ax.axis("off")
    for k, (fid, xy, trunk) in enumerate(samples):
        ax = axes[k]; ax.axis("on")
        g = group_of(fid); c = color_of(g)
        # wrist trajectories (the roaming endpoints) as lines
        for w in ("left_wrist", "right_wrist"):
            if w in xy:
                ax.plot(xy[w][0], xy[w][1], lw=0.6, alpha=0.7, color=c)
        # static-ish reference points: shoulders, hips, nose (median position)
        for p in ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "nose"):
            if p in xy:
                ax.plot(np.nanmedian(xy[p][0]), np.nanmedian(xy[p][1]),
                        "o", ms=3, color="k", alpha=0.6)
        ax.axhline(0, color="grey", lw=0.4); ax.axvline(0, color="grey", lw=0.4)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        bad = (not np.isfinite(trunk)) or abs(trunk - 1) > TRUNK_TOL
        ax.set_title(f"{fid[:16]}\ntrunk={trunk:.2f}", fontsize=6,
                     color="crimson" if bad else "black")
    fig.suptitle("Wrist trajectories (torso-centred, trunk-normalised).  "
                 "Red title = trunk far from 1.0", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_CONTACT, dpi=130)
    plt.close(fig)
    print("wrote", OUT_CONTACT)

# ----------------------------------------------------------------------------- animation
def make_animation(samples, lim):
    n = len(samples)
    ncol = int(np.ceil(np.sqrt(n * 1.4)))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.9, nrow * 2.1))
    axes = np.array(axes).reshape(-1)

    # precompute a per-file frame index map onto a common timeline of MAX_FRAMES
    prepared = []
    for k, (fid, xy, trunk) in enumerate(samples):
        L = max((len(v[0]) for v in xy.values()), default=1)
        idx = (np.linspace(0, L - 1, MAX_FRAMES)).astype(int)   # resample to common T
        prepared.append((fid, xy, trunk, idx))

    artists = []
    for k, ax in enumerate(axes):
        ax.axis("off")
        if k >= n:
            artists.append(None); continue
        fid, xy, trunk, idx = prepared[k]
        ax.axis("on")
        g = group_of(fid); c = color_of(g)
        ax.axhline(0, color="grey", lw=0.3); ax.axvline(0, color="grey", lw=0.3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        bad = (not np.isfinite(trunk)) or abs(trunk - 1) > TRUNK_TOL
        ax.set_title(f"{fid[:14]}  t={trunk:.2f}", fontsize=6,
                     color="crimson" if bad else "black")
        bones = [ax.plot([], [], "-", lw=1.3, color=c)[0] for _ in EDGES]
        joints, = ax.plot([], [], "o", ms=2.5, color="k")
        artists.append((xy, idx, bones, joints))

    def update(f):
        out = []
        for a in artists:
            if a is None: continue
            xy, idx, bones, joints = a
            fi = idx[f]
            for e, (p1, p2) in zip(bones, EDGES):
                if p1 in xy and p2 in xy:
                    e.set_data([xy[p1][0][fi], xy[p2][0][fi]],
                               [xy[p1][1][fi], xy[p2][1][fi]])
                else:
                    e.set_data([], [])
            jx = [xy[p][0][fi] for p in POINTS if p in xy]
            jy = [xy[p][1][fi] for p in POINTS if p in xy]
            joints.set_data(jx, jy)
            out += bones + [joints]
        return out

    fig.suptitle("Sapiens processed skeletons — synchronised.  "
                 "Red title = trunk far from 1.0", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    ani = FuncAnimation(fig, update, frames=MAX_FRAMES, interval=50, blit=True)
    try:
        ani.save(OUT_ANIM, writer=FFMpegWriter(fps=20, bitrate=2400))
        print("wrote", OUT_ANIM)
    except Exception as e:
        alt = OUT_ANIM.replace(".mp4", ".gif")
        print("ffmpeg failed (%s); writing gif instead" % e)
        ani.save(alt, writer=PillowWriter(fps=15)); print("wrote", alt)
    plt.close(fig)

# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    files = pick_files()
    samples = [load_one(f) for f in files]
    lim = shared_limit(samples)
    print(f"{len(samples)} files | shared axis limit ±{lim:.2f} trunk-lengths")
    n_bad = sum(1 for _, _, t in samples if (not np.isfinite(t)) or abs(t - 1) > TRUNK_TOL)
    print(f"{n_bad} file(s) with trunk >{int(TRUNK_TOL*100)}% off 1.0 (flagged red)")
    if MODE in ("contact", "both"):
        make_contact_sheet(samples, lim)
    if MODE in ("animation", "both"):
        make_animation(samples, lim)
