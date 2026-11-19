"""Show the effect of the normalization steps on the body skeleton (no fingers).

Three panels, each overlaying the first frame of several recordings
(e.g. one per group):
  1. raw        - after clean_and_process_df (image coordinates)
  2. centered   - after add_midtorso_and_center (mid-torso at [0, 0])
  3. normalized - after size_normalize_by_trunk (divided by trunk length)

Usage, in the notebook where your pipeline functions are defined:

    files = {"Gesture": csv_gesture, "Learner_1": csv_l1,
             "Learner_2": csv_l2, "Sign": csv_sign}
    plot_normalization(files, TOKEEP, median_bodylen, out="normalization.png")
"""
import matplotlib.pyplot as plt

BODY_EDGES = [
    ("nose", "neck"),
    ("neck", "left_shoulder"), ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
]

GROUP_COLS = {"Gesture": "#878787", "Learner_1": "#77AB00",
              "Learner_2": "#00BCC2", "Sign": "#8B3FAB"}


def draw_body(ax, row, color, label=None):
    """Draw body segments of one frame (a pandas Series)."""
    first = True
    for a, b in BODY_EDGES:
        if all(f"{k}_{d}" in row.index for k in (a, b) for d in "xy"):
            ax.plot([row[f"{a}_x"], row[f"{b}_x"]], [row[f"{a}_y"], row[f"{b}_y"]],
                    color=color, lw=1.6, alpha=0.9, label=label if first else None)
            first = False
    if "midtorso_x" in row.index:          # adjust if your mid-torso column is named differently
        ax.scatter(row["midtorso_x"], row["midtorso_y"], s=18, color=color,
                   marker="x", zorder=3)


def plot_normalization(files, tokeep, median_bodylen, frame=0, out=None):
    steps = {"Raw": [], "Centered on mid-torso": [], "Size-normalized": []}

    for group, csv in files.items():
        raw = clean_and_process_df(csv, tokeep)
        raw = raw.loc[:, ~raw.columns.str.contains("_conf")]
        centered = add_midtorso_and_center(raw.copy())

        pcn = raw["FileID"].iloc[0].split("_")[0]
        trunk = median_bodylen.loc[median_bodylen["pcn_ID"] == pcn, "trunk_len"].iloc[0]
        _, normalized = size_normalize_by_trunk(centered.copy(), trunk_len=trunk)

        for name, d in zip(steps, (raw, centered, normalized)):
            steps[name].append((group, d.iloc[frame]))

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    for ax, (name, rows) in zip(axes, steps.items()):
        for group, row in rows:
            draw_body(ax, row, GROUP_COLS.get(group, "#444444"), label=group)
        if name != "Raw":
            ax.axhline(0, color="0.85", lw=0.8, zorder=0)
            ax.axvline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_title(name, fontsize=10)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        # if skeletons appear upside down, uncomment:
        # ax.invert_yaxis()
    axes[0].legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=300, bbox_inches="tight")
    return fig
