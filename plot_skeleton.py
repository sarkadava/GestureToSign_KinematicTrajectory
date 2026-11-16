"""Plot the skeleton of the first frame of one recording.

Usage (in your processing notebook, where clean_and_process_df and TOKEEP exist):

    df = clean_and_process_df(csv, TOKEEP)          # raw, flipped, interpolated
    plot_skeleton(df, frame=0, out="skeleton_first_frame.png")

Use the cleaned data *before* centering/normalization: after
center_fingers_on_wrist the finger keypoints are wrist-relative and would be
drawn around the origin instead of on the hand.
"""
import re
import matplotlib.pyplot as plt

BODY_EDGES = [
    ("nose", "neck"),
    ("neck", "left_shoulder"), ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
]


def _joint_rank(name):
    """Order finger joints from base to tip: *_third_joint (knuckle) < 2 < 3 < 4 (tip)."""
    if name.endswith("third_joint"):
        return 1
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 99


def _finger_chains(kpts):
    """Group finger keypoints per side and finger, ordered base -> tip."""
    chains = {}
    for k in kpts:
        if ("finger" in k or "thumb" in k) and k.startswith(("left_", "right_")):
            side = k.split("_")[0]
            finger = re.sub(r"(_third_joint|\d+)$", "", k)
            chains.setdefault((side, finger), []).append(k)
    return {key: sorted(v, key=_joint_rank) for key, v in chains.items()}


def plot_skeleton(df, frame=0, out=None, color="#444444", ax=None):
    row = df.iloc[frame]
    kpts = sorted({c[:-2] for c in df.columns if c.endswith("_x") and f"{c[:-2]}_y" in df.columns})
    xy = {k: (row[f"{k}_x"], row[f"{k}_y"]) for k in kpts}

    edges = [e for e in BODY_EDGES if e[0] in xy and e[1] in xy]
    for (side, _), chain in _finger_chains(kpts).items():
        wrist = f"{side}_wrist"
        if wrist in xy:
            edges.append((wrist, chain[0]))
        edges += list(zip(chain[:-1], chain[1:]))

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 5))
    for a, b in edges:
        ax.plot([xy[a][0], xy[b][0]], [xy[a][1], xy[b][1]], color=color, lw=1.5)
    ax.scatter([p[0] for p in xy.values()], [p[1] for p in xy.values()],
               s=8, color=color, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    # If the figure appears upside down, the y axis still runs in image
    # coordinates: uncomment the next line.
    # ax.invert_yaxis()
    if out:
        plt.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    return ax
