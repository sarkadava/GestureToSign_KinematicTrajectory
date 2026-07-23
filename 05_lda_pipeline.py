"""
=============================================================================
 A principled LDA pipeline for group classification from kinematic features
=============================================================================

Designed for data where:
  - rows are windows/trials NESTED within participants
  - participants are nested within groups (group is a BETWEEN-participant factor)
  - features are many and heavily collinear

Every stage prints what it found and why it matters. Figures go to ./lda_out/.

Run:  python lda_pipeline.py
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import linalg, stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA,
)
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- CONFIG ---
CSV        = os.path.join("Datasets", "sampled_all_250_cleanforPCA.csv")
LABEL_COL  = "groupID"
GROUP_COL  = "participantID"   # None if you truly have one row per participant
DURATION_COL = None            # e.g. "duration_s"; used to check count confounds
OUTDIR     = "lda_out"
N_SPLITS   = 5
N_PERM     = 1000              # permutations for significance tests
N_BOOT     = 500               # bootstrap resamples for loading stability
CORR_CUT   = 0.30              # dendrogram cut: 1 - |rho| < 0.30  =>  |rho| > 0.70
RANDOM_STATE = 1
# ---------------------------------------------------------------------------

os.makedirs(OUTDIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_STATE)


def banner(txt):
    print("\n" + "=" * 78)
    print(txt)
    print("=" * 78)


# ===========================================================================
# STAGE 1 -- LOAD AND HYGIENE
# ===========================================================================
banner("STAGE 1  Load and hygiene")

data = pd.read_csv(CSV)
print(f"raw shape: {data.shape}")

y = data[LABEL_COL].astype(str)
drop = [LABEL_COL] + ([GROUP_COL] if GROUP_COL else [])
X = data.drop(columns=drop).select_dtypes(include=[np.number]).copy()

if GROUP_COL and GROUP_COL in data.columns:
    groups = data[GROUP_COL].astype(str)
    n_part = groups.nunique()
    per_group = data.groupby(LABEL_COL)[GROUP_COL].nunique()
    print(f"participants: {n_part}   rows/participant: "
          f"{len(data)/n_part:.1f}   participants per class:\n{per_group}")
    if (len(data) / n_part) > 1.5:
        print("\n  >> Rows are NESTED in participants. All CV below is grouped.\n"
              "     Ungrouped CV here is pseudoreplication: the model can memorise\n"
              "     individual participants and be scored on the same people. The\n"
              "     size of the inflation scales with between-participant variance,\n"
              "     so run both and report the grouped number.")
else:
    groups = pd.Series(np.arange(len(data)).astype(str))
    print("\n  !! No participant column. Everything below ASSUMES rows are\n"
          "     independent. If they are not, every p-value here is wrong.\n")

# non-finite values
bad = ~np.isfinite(X.to_numpy()).all(axis=1)
if bad.any():
    print(f"dropping {bad.sum()} rows with NaN/inf")
    X, y, groups = X[~bad], y[~bad], groups[~bad]

# constant / near-constant features carry no information and destabilise the
# within-class covariance inversion
nzv = X.columns[X.std(ddof=1) < 1e-10]
if len(nzv):
    print(f"dropping {len(nzv)} zero-variance features: {list(nzv)}")
    X = X.drop(columns=nzv)

feat_names = list(X.columns)
classes = np.array(sorted(y.unique()))
p, g, N = X.shape[1], len(classes), len(X)
n_lds = min(p, g - 1)
print(f"\nfinal: N={N}  p={p}  g={g}  -> at most {n_lds} discriminant functions")
print(f"class counts:\n{y.value_counts()}")
chance = y.value_counts(normalize=True).max()
print(f"majority-class baseline = {chance:.3f}")

# --- count/duration confound check ------------------------------------------
if DURATION_COL and DURATION_COL in data.columns:
    dur = data.loc[X.index, DURATION_COL]
    count_feats = [c for c in feat_names if c.endswith("_n")]
    print("\ncorrelation of count features with trial duration:")
    for c in count_feats:
        r = stats.spearmanr(X[c], dur).statistic
        flag = "  <-- RED FLAG, normalise by duration" if abs(r) > .5 else ""
        print(f"   {c:42s} rho={r:+.2f}{flag}")
else:
    print("\n  ?  DURATION_COL not set. If your '_n' features are raw counts "
          "rather than\n     rates, they partly index trial length, not "
          "movement structure. Check this.")


# ===========================================================================
# STAGE 2 -- COLLINEARITY DIAGNOSTICS
# ===========================================================================
banner("STAGE 2  Collinearity diagnostics")

Xs = StandardScaler().fit_transform(X)
corr = pd.DataFrame(Xs, columns=feat_names).corr(method="spearman")

# condition number of the correlation matrix: >30 moderate, >100 severe
eig = np.linalg.eigvalsh(corr.to_numpy())
cond = np.sqrt(eig.max() / max(eig.min(), 1e-12))
print(f"condition number of correlation matrix: {cond:,.0f}")
print("   <30 fine | 30-100 moderate | >100 severe: raw LDA weights are "
      "uninterpretable")

# variance inflation factors, computed from the inverse correlation matrix
vif = pd.Series(np.diag(np.linalg.pinv(corr.to_numpy())), index=feat_names)
print(f"\nVIF > 10 for {int((vif > 10).sum())}/{p} features. Worst:")
print(vif.sort_values(ascending=False).head(8).round(1).to_string())

# hierarchical clustering of features on correlation distance
dist = 1.0 - corr.abs().to_numpy()
np.fill_diagonal(dist, 0.0)
dist = (dist + dist.T) / 2
Z = hierarchy.linkage(squareform(dist, checks=False), method="average")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
dn = hierarchy.dendrogram(Z, labels=feat_names, ax=axes[0],
                          orientation="left", color_threshold=CORR_CUT)
axes[0].axvline(CORR_CUT, color="crimson", ls="--")
axes[0].set_title(f"Feature clustering on 1-|rho|  (cut at {CORR_CUT})")
axes[0].tick_params(labelsize=7)

order = dn["leaves"]
im = axes[1].imshow(corr.to_numpy()[np.ix_(order, order)],
                    cmap="RdBu_r", vmin=-1, vmax=1)
axes[1].set_xticks(range(p)); axes[1].set_yticks(range(p))
axes[1].set_xticklabels([feat_names[i] for i in order], rotation=90, fontsize=6)
axes[1].set_yticklabels([feat_names[i] for i in order], fontsize=6)
axes[1].set_title("Spearman correlation, dendrogram order")
fig.colorbar(im, ax=axes[1], shrink=.7)
fig.tight_layout(); fig.savefig(f"{OUTDIR}/01_collinearity.png", dpi=150)
plt.close(fig)

cl = hierarchy.fcluster(Z, CORR_CUT, criterion="distance")
print(f"\n{p} features collapse to {cl.max()} correlation clusters at "
      f"|rho| > {1-CORR_CUT:.2f}")


# ===========================================================================
# STAGE 3 -- LDA ASSUMPTIONS
# ===========================================================================
banner("STAGE 3  Assumption checks")

# (a) within-class multivariate normality -- checked descriptively via
#     univariate skew/kurtosis. LDA is fairly robust for CLASSIFICATION;
#     violations matter mainly for the parametric tests in stage 6.
sk = pd.DataFrame(
    {c: stats.skew(Xs[(y == c).to_numpy()], axis=0) for c in classes},
    index=feat_names)
ku = pd.DataFrame(
    {c: stats.kurtosis(Xs[(y == c).to_numpy()], axis=0) for c in classes},
    index=feat_names)
worst = sk.abs().max(axis=1).sort_values(ascending=False)
print("worst within-class skew (|skew| > 2 is a concern):")
print(worst.head(5).round(2).to_string())
print(f"features with |skew| > 2 in some class: {int((sk.abs() > 2).any(axis=1).sum())}/{p}")
print(f"features with |excess kurtosis| > 7:    {int((ku.abs() > 7).any(axis=1).sum())}/{p}")
print("   -> if many, consider log/sqrt transforms of the offending features")


# (b) equality of covariance matrices -- Box's M
def box_m(Xarr, yarr):
    cls = np.unique(yarr)
    Nn, pp, gg = Xarr.shape[0], Xarr.shape[1], len(cls)
    pooled = np.zeros((pp, pp)); logdets = []; ns = []
    for c in cls:
        Xc = Xarr[yarr == c]
        nc = Xc.shape[0]; ns.append(nc)
        Sc = np.cov(Xc, rowvar=False)
        pooled += (nc - 1) * Sc
        logdets.append(np.linalg.slogdet(Sc)[1])
    pooled /= (Nn - gg)
    M = (Nn - gg) * np.linalg.slogdet(pooled)[1] - \
        sum((n - 1) * ld for n, ld in zip(ns, logdets))
    c1 = (sum(1 / (n - 1) for n in ns) - 1 / (Nn - gg)) * \
         (2 * pp**2 + 3 * pp - 1) / (6 * (pp + 1) * (gg - 1))
    chi2 = M * (1 - c1)
    df = pp * (pp + 1) * (gg - 1) / 2
    return M, chi2, df, stats.chi2.sf(chi2, df)


if min(y.value_counts()) > p + 2:
    M, chi2, df, pv = box_m(Xs, y.to_numpy())
    print(f"\nBox's M = {M:.1f}   chi2 = {chi2:.1f}  df = {df:.0f}  p = {pv:.3g}")
    print("   !! Box's M is hypersensitive: with N in the hundreds it rejects\n"
          "      almost always. Do NOT treat p<.001 here as fatal. The useful\n"
          "      test is empirical -- compare LDA against QDA below.")
else:
    print("\nBox's M skipped: at least one class has fewer rows than features+2")


# ===========================================================================
# STAGE 4 -- THREE WAYS TO HANDLE COLLINEARITY
# ===========================================================================
banner("STAGE 4  Collinearity strategies")

class CorrClusterSelect(BaseEstimator, TransformerMixin):
    """Cluster features on 1-|Spearman rho|, keep one medoid per cluster.

    UNSUPERVISED: never sees y, so it cannot leak class information. Fully
    deterministic given a threshold, which is what makes it reproducible --
    unlike stepwise/VIF-dropping, which is order-dependent.
    """
    def __init__(self, threshold=0.30):
        self.threshold = threshold

    def fit(self, X, y=None):
        Xa = np.asarray(X)
        c = pd.DataFrame(Xa).corr(method="spearman").to_numpy()
        c = np.nan_to_num(c, nan=0.0)
        d = 1.0 - np.abs(c)
        np.fill_diagonal(d, 0.0)
        d = (d + d.T) / 2
        lab = hierarchy.fcluster(
            hierarchy.linkage(squareform(d, checks=False), method="average"),
            self.threshold, criterion="distance")
        keep = []
        for k in np.unique(lab):
            idx = np.where(lab == k)[0]
            # medoid = member most representative of its own cluster
            keep.append(idx[np.argmax(np.abs(c[np.ix_(idx, idx)]).mean(axis=1))])
        self.support_ = np.sort(np.array(keep))
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]


STRATEGIES = {
    # (A) drop redundant features, keep original units -> most interpretable
    "cluster+LDA": Pipeline([
        ("sc", StandardScaler()),
        ("sel", CorrClusterSelect(CORR_CUT)),
        ("lda", LDA(solver="svd", n_components=n_lds)),
    ]),
    # (B) rotate into orthogonal components -> collinearity gone by construction
    "PCA+LDA": Pipeline([
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
        ("lda", LDA(solver="svd", n_components=n_lds)),
    ]),
    # (C) keep everything, regularise the covariance estimate instead
    "shrinkage LDA": Pipeline([
        ("sc", StandardScaler()),
        ("lda", LDA(solver="eigen", shrinkage="auto", n_components=n_lds)),
    ]),
    # (D) reference: does relaxing the equal-covariance assumption help?
    "QDA (ref)": Pipeline([
        ("sc", StandardScaler()),
        ("sel", CorrClusterSelect(CORR_CUT)),
        ("qda", QDA(reg_param=0.1)),
    ]),
}


# ===========================================================================
# STAGE 5 -- GROUPED CROSS-VALIDATION
# ===========================================================================
banner("STAGE 5  Grouped cross-validation")

cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)
splits = list(cv.split(X, y, groups))

results = {}
for name, pipe in STRATEGIES.items():
    fold = []
    for tr, te in splits:
        m = clone(pipe)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X.iloc[tr], y.iloc[tr])
            fold.append(balanced_accuracy_score(y.iloc[te], m.predict(X.iloc[te])))
    results[name] = np.array(fold)
    print(f"{name:16s} balanced acc = {np.mean(fold):.3f} "
          f"+/- {np.std(fold):.3f}   folds: "
          f"{np.round(fold, 3)}")

print(f"\nchance (balanced) = {1/g:.3f}")
print("Balanced accuracy, not raw accuracy: it is not inflated by class "
      "imbalance.")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.boxplot(results.values(), tick_labels=results.keys())
ax.axhline(1 / g, color="crimson", ls="--", label="chance")
ax.set_ylabel("balanced accuracy (grouped CV)")
ax.legend(); fig.tight_layout()
fig.savefig(f"{OUTDIR}/02_strategy_comparison.png", dpi=150); plt.close(fig)

BEST = max(results, key=lambda k: results[k].mean())
if "QDA" in BEST:
    print("\n  >> QDA wins: covariances differ enough between groups that the\n"
          "     LDA equal-covariance assumption is costing you. Report LDA for\n"
          "     interpretation but flag this.")
print(f"\nselected for interpretation: {BEST}")
best_pipe = clone(STRATEGIES[BEST])


# ===========================================================================
# STAGE 6 -- IS THE CLASSIFICATION REAL?  (permutation test)
# ===========================================================================
banner("STAGE 6  Permutation test on classification accuracy")

# Group is a BETWEEN-participant factor, so the null must shuffle labels at
# the PARTICIPANT level. Shuffling row labels would break the nesting and give
# a null that is far too easy to beat -> a meaningless p-value.
part_label = pd.DataFrame({"pid": groups.values, "lab": y.values}) \
               .drop_duplicates().set_index("pid")["lab"]
pids = part_label.index.to_numpy()


def cv_score(y_vec):
    sc = []
    for tr, te in StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True,
            random_state=RANDOM_STATE).split(X, y_vec, groups):
        m = clone(best_pipe)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X.iloc[tr], y_vec.iloc[tr])
            sc.append(balanced_accuracy_score(y_vec.iloc[te],
                                              m.predict(X.iloc[te])))
    return np.mean(sc)


obs = results[BEST].mean()
null = np.empty(N_PERM)
for i in range(N_PERM):
    perm = pd.Series(rng.permutation(part_label.to_numpy()), index=pids)
    null[i] = cv_score(groups.map(perm).reset_index(drop=True))
p_acc = (np.sum(null >= obs) + 1) / (N_PERM + 1)
print(f"observed = {obs:.3f}   permutation null mean = {null.mean():.3f}   "
      f"p = {p_acc:.4f}  ({N_PERM} perms)")

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.hist(null, bins=40, color="#9db4d0", edgecolor="w")
ax.axvline(obs, color="crimson", lw=2, label=f"observed {obs:.3f}")
ax.set_xlabel("balanced accuracy under participant-level label permutation")
ax.legend(); fig.tight_layout()
fig.savefig(f"{OUTDIR}/03_permutation_accuracy.png", dpi=150); plt.close(fig)


# ===========================================================================
# STAGE 7 -- WHICH DISCRIMINANT FUNCTIONS ARE INTERPRETABLE?
# ===========================================================================
banner("STAGE 7  Dimensionality: how many LDs are real?")

# reduce once with the UNSUPERVISED part of the chosen pipeline, so that
# permuting labels afterwards is a valid null
pre = Pipeline(best_pipe.steps[:-1]).fit(X)
Zred = pre.transform(X)
if "sel" in pre.named_steps:
    red_names = list(np.array(feat_names)[pre.named_steps["sel"].support_])
elif "pca" in pre.named_steps:
    red_names = [f"PC{i+1}" for i in range(Zred.shape[1])]
else:
    red_names = list(feat_names)
print(f"interpreting in a space of {Zred.shape[1]} variables: "
      f"{red_names[:6]}{' ...' if len(red_names) > 6 else ''}")


def lda_eigs(Zm, yv):
    """Generalised eigenvalues of B w.r.t. W -- the quantities LDA maximises."""
    cls = np.unique(yv); pp = Zm.shape[1]
    grand = Zm.mean(0)
    W = np.zeros((pp, pp)); B = np.zeros((pp, pp))
    for c in cls:
        Zc = Zm[yv == c]
        mc = Zc.mean(0); d = Zc - mc
        W += d.T @ d
        dm = (mc - grand)[:, None]
        B += len(Zc) * dm @ dm.T
    W += np.eye(pp) * 1e-8
    ev = linalg.eigh(B, W, eigvals_only=True)
    return np.sort(np.real(ev))[::-1][:min(pp, len(cls) - 1)]


yv = y.to_numpy()
lam = lda_eigs(Zred, yv)
canon = np.sqrt(lam / (1 + lam))
prop = lam / lam.sum()

# --- (a) classical Bartlett / Wilks' lambda sequential test ------------------
print("Bartlett sequential test (PARAMETRIC -- see caveat below)")
pr = Zred.shape[1]
for k in range(len(lam)):
    wilks = np.prod(1 / (1 + lam[k:]))
    chi2 = -(N - 1 - (pr + g) / 2) * np.log(wilks)
    df = (pr - k) * (g - 1 - k)
    print(f"   LD{k+1}..LD{len(lam)}:  Wilks' lambda = {wilks:.4f}  "
          f"chi2 = {chi2:.1f}  df = {df}  p = {stats.chi2.sf(chi2, df):.3g}")
print("   !! These p-values use N = number of ROWS. With rows nested in\n"
       "      participants, N is inflated and every p here is anticonservative.\n"
       "      Treat as descriptive only. The permutation version below is the\n"
       "      one to report.")

# --- (b) permutation test on each eigenvalue --------------------------------
print("\nParticipant-level permutation test on each discriminant")
null_lam = np.empty((N_PERM, len(lam)))
for i in range(N_PERM):
    perm = pd.Series(rng.permutation(part_label.to_numpy()), index=pids)
    null_lam[i] = lda_eigs(Zred, groups.map(perm).to_numpy())
for k in range(len(lam)):
    pk = (np.sum(null_lam[:, k] >= lam[k]) + 1) / (N_PERM + 1)
    verdict = "INTERPRETABLE" if pk < .05 else "do NOT interpret"
    print(f"   LD{k+1}: eigenvalue = {lam[k]:.3f}  canonical r = {canon[k]:.3f}  "
          f"{prop[k]*100:5.1f}% of trace   p = {pk:.4f}   {verdict}")

fig, axes = plt.subplots(1, len(lam), figsize=(4.2 * len(lam), 3.6), squeeze=False)
for k in range(len(lam)):
    a = axes[0, k]
    a.hist(null_lam[:, k], bins=35, color="#9db4d0", edgecolor="w")
    a.axvline(lam[k], color="crimson", lw=2)
    a.set_title(f"LD{k+1}  ({prop[k]*100:.1f}% of trace)")
    a.set_xlabel("eigenvalue under null")
fig.tight_layout(); fig.savefig(f"{OUTDIR}/04_permutation_eigenvalues.png", dpi=150)
plt.close(fig)


# ===========================================================================
# STAGE 8 -- ARE THE LOADINGS STABLE?
# ===========================================================================
banner("STAGE 8  Bootstrap stability of structure coefficients")

def structure_coefs(Zm, scores):
    out = np.empty((Zm.shape[1], scores.shape[1]))
    for j in range(Zm.shape[1]):
        for k in range(scores.shape[1]):
            out[j, k] = np.corrcoef(Zm[:, j], scores[:, k])[0, 1]
    return np.nan_to_num(out)

ref_lda = LDA(solver="svd", n_components=len(lam)).fit(Zred, yv)
ref_scores = ref_lda.transform(Zred)
ref_struct = structure_coefs(Zred, ref_scores)

boot = np.full((N_BOOT, Zred.shape[1], len(lam)), np.nan)
for b in range(N_BOOT):
    take = rng.choice(pids, size=len(pids), replace=True)   # resample PARTICIPANTS
    idx = np.concatenate([np.where(groups.values == pp_)[0] for pp_ in take])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LDA(solver="svd", n_components=len(lam)).fit(Zred[idx], yv[idx])
        s = structure_coefs(Zred[idx], m.transform(Zred[idx]))
    except Exception:
        continue
    # discriminant axes have arbitrary sign; align each to the reference
    for k in range(len(lam)):
        if np.dot(s[:, k], ref_struct[:, k]) < 0:
            s[:, k] *= -1
    boot[b] = s

for k in range(min(2, len(lam))):
    lo = np.nanpercentile(boot[:, :, k], 2.5, axis=0)
    hi = np.nanpercentile(boot[:, :, k], 97.5, axis=0)
    sign_ok = np.nanmean(np.sign(boot[:, :, k]) == np.sign(ref_struct[:, k]), axis=0)
    tab = pd.DataFrame({"struct": ref_struct[:, k], "lo": lo, "hi": hi,
                        "sign_consistency": sign_ok}, index=red_names)
    tab["stable"] = (tab.lo * tab.hi > 0) & (tab.sign_consistency > .95)
    tab = tab.reindex(tab.struct.abs().sort_values(ascending=False).index)
    print(f"\n--- LD{k+1} structure coefficients (95% bootstrap CI over participants)")
    print(tab.round(3).to_string())
    print(f"    {int(tab.stable.sum())}/{len(tab)} features stable. Interpret ONLY "
          f"those.")

    o = tab.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.30 * len(o) + 1.6))
    col = np.where(o.stable, "#2b6cb0", "#c8ccd4")
    ax.errorbar(o.struct, range(len(o)),
                xerr=[o.struct - o.lo, o.hi - o.struct],
                fmt="none", ecolor=col, lw=2)
    ax.scatter(o.struct, range(len(o)), color=col, zorder=3)
    ax.axvline(0, color="k", lw=.8)
    ax.set_yticks(range(len(o))); ax.set_yticklabels(o.index, fontsize=8)
    ax.set_xlabel(f"LD{k+1} structure coefficient")
    ax.set_title(f"LD{k+1}: blue = stable across participant bootstraps")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/05_loadings_LD{k+1}.png", dpi=150)
    plt.close(fig)


# ===========================================================================
# STAGE 9 -- FINAL FIGURES
# ===========================================================================
banner("STAGE 9  Final figures")

centroids = pd.DataFrame(ref_scores[:, :min(2, len(lam))],
                         columns=[f"LD{i+1}" for i in range(min(2, len(lam)))])
centroids["grp"] = yv
print("group centroids (use these to read the SIGN of every loading):")
print(centroids.groupby("grp").mean().round(2).to_string())

# out-of-fold confusion matrix
oof = cross_val_predict(clone(best_pipe), X, y, cv=splits, groups=groups)
cm = confusion_matrix(y, oof, labels=classes, normalize="true")
fig, ax = plt.subplots(figsize=(5.6, 5))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
for i in range(g):
    for j in range(g):
        ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                color="white" if cm[i, j] > .5 else "black", fontsize=9)
ax.set_xticks(range(g)); ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticks(range(g)); ax.set_yticklabels(classes)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title("Out-of-fold confusion matrix (row-normalised)")
fig.colorbar(im, shrink=.75); fig.tight_layout()
fig.savefig(f"{OUTDIR}/06_confusion.png", dpi=150); plt.close(fig)

# biplot
if len(lam) >= 2:
    fig, ax = plt.subplots(figsize=(8, 7))
    cols = plt.cm.tab10(np.linspace(0, 1, g))
    for c, cls_ in zip(cols, classes):
        m = yv == cls_
        ax.scatter(ref_scores[m, 0], ref_scores[m, 1], s=14, alpha=.30,
                   color=c, label=cls_)
    mag = np.hypot(ref_struct[:, 0], ref_struct[:, 1])
    sel = np.argsort(mag)[::-1][:8]
    scale = .85 * np.abs(ref_scores[:, :2]).max()
    for j in sel:
        dx, dy = ref_struct[j, 0] * scale, ref_struct[j, 1] * scale
        ax.arrow(0, 0, dx, dy, color="k", lw=1.3, head_width=.12,
                 length_includes_head=True, zorder=5)
        ax.text(dx * 1.1, dy * 1.1, red_names[j], fontsize=7.5, ha="center")
    ax.axhline(0, color="grey", lw=.5); ax.axvline(0, color="grey", lw=.5)
    ax.set_xlabel(f"LD1 ({prop[0]*100:.1f}%)")
    ax.set_ylabel(f"LD2 ({prop[1]*100:.1f}%)")
    ax.legend(markerscale=2.5); fig.tight_layout()
    fig.savefig(f"{OUTDIR}/07_biplot.png", dpi=150); plt.close(fig)

banner("DONE")
print(f"figures in ./{OUTDIR}/")
print(f"""
REPORTABLE SUMMARY
  strategy              {BEST}
  balanced accuracy     {obs:.3f} (grouped {N_SPLITS}-fold CV), chance {1/g:.3f}
  permutation p         {p_acc:.4f} ({N_PERM} participant-level permutations)
  discriminants kept    see stage 7
  loadings              interpret only rows marked stable in stage 8
""")
