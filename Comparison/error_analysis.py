"""Error analysis A–D, H, I of Plans/PatchCore_error_analysis_plan.md (all local, from saved scores).

    python Comparison/error_analysis.py --out Results/patchcore/error_analysis

A  error_table.csv        per test file: score, margin, prediction, error per seed, errors of other methods
B  error_by_date.csv / error_by_label.csv / error_rate_by_date.png
C  calibration.json       error rate by margin decile; rank-normalization (H2) re-evaluation
D  aggregation_study.csv  every window->file rule, threshold chosen on val (H1)
H  method_overlap.csv     which files each method gets wrong
I  bootstrap.json         date-level bootstrap CI per method + paired differences

Conventions: positive class = Noise (score s = d_pd - d_noise). For reporting, PD is the event of
interest, so a "missed PD" is a PD file called Noise and a "false alarm" is a Noise file called PD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
METHODS = {
    "PatchCore": "Results/patchcore/final_k8_s{seed}/eval",
    "EfficientAD": "Results/efficientad/effad_v2_k8_s{seed}/eval",
    "One-Class SVM": "Results/svm/ocsvm_v2_k8_s{seed}/eval",
}
BASELINE = ("kNN-5 (256-D)", "Results/patchcore/baselines_v2/knn5")
AGGS = ["mean", "median", "max", "min", "p90"]
INK, ACCENT, WARN = "#1a1a1a", "#3d6fa8", "#b4461f"


def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    v = np.unique(s)
    cand = np.concatenate([[v[0] - 1e-9], (v[:-1] + v[1:]) / 2, [v[-1] + 1e-9]])
    f1s = [f1_score(y, (s > t).astype(int), average="macro", zero_division=0) for t in cand]
    return float(cand[int(np.argmax(f1s))])


def load_eval(path: str) -> tuple[pd.DataFrame, str]:
    d = REPO_ROOT / path
    files = pd.read_csv(d / "file_scores.csv", dtype={"date": str})
    agg = json.loads((d / "val_selection.json").read_text())["aggregation"]
    return files, f"S_{agg}"


def decide(files: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add prediction and error flag using the val-chosen threshold."""
    val, test = files[files.split == "val"], files[files.split == "test"].copy()
    tau = best_threshold((val.cls == "Noise").to_numpy(int), val[col].to_numpy())
    test["score"] = test[col]
    test["margin"] = test[col] - tau
    test["pred"] = np.where(test[col] > tau, "Noise", "PD")
    test["error"] = test.pred != test.cls
    test["error_kind"] = np.where(~test.error, "correct",
                                  np.where(test.cls == "PD", "missed PD", "false alarm"))
    test.attrs["tau"] = tau
    return test


# ----------------------------------------------------------------- A, B, H

def build_tables(out: Path) -> pd.DataFrame:
    base = decide(*load_eval(METHODS["PatchCore"].format(seed=42)))
    table = base[["sample_id", "date", "group", "label", "cls", "score", "margin", "pred",
                  "error", "error_kind"]].copy()
    for s in SEEDS:
        t = decide(*load_eval(METHODS["PatchCore"].format(seed=s))).set_index("sample_id")
        table[f"pc_err_s{s}"] = table.sample_id.map(t.error)
        table[f"pc_score_s{s}"] = table.sample_id.map(t.score)
    table["pc_err_count"] = table[[f"pc_err_s{s}" for s in SEEDS]].sum(axis=1)
    for name, pat in list(METHODS.items())[1:]:
        t = decide(*load_eval(pat.format(seed=42))).set_index("sample_id")
        table[f"{name}_err"] = table.sample_id.map(t.error)
    bl = decide(*load_eval(BASELINE[1])).set_index("sample_id")
    table["knn5_err"] = table.sample_id.map(bl.error)
    table.to_csv(out / "error_table.csv", index=False)

    by_date = (table.groupby("date").agg(files=("sample_id", "size"), n_pd=("cls", lambda c: int((c == "PD").sum())),
                                         errors=("error", "sum"), missed_pd=("error_kind", lambda k: int((k == "missed PD").sum())),
                                         false_alarms=("error_kind", lambda k: int((k == "false alarm").sum())))
               .assign(error_rate=lambda d: d.errors / d.files).sort_values("errors", ascending=False))
    by_date.to_csv(out / "error_by_date.csv")
    by_label = (table.groupby(["cls", "label"]).agg(files=("sample_id", "size"), errors=("error", "sum"))
                .assign(error_rate=lambda d: d.errors / d.files))
    by_label.to_csv(out / "error_by_label.csv")

    cols = ["error", "EfficientAD_err", "One-Class SVM_err", "knn5_err"]
    names = ["PatchCore", "EfficientAD", "SVM", "kNN-5"]
    ov = pd.DataFrame([[int((table[a] & table[b]).sum()) for b in cols] for a in cols], index=names, columns=names)
    ov.to_csv(out / "method_overlap.csv")

    fig, ax = plt.subplots(figsize=(9, 3.4))
    d = by_date[by_date.files >= 3].sort_values("error_rate", ascending=False)
    ax.bar(range(len(d)), d.error_rate, color=[WARN if r > 0.3 else ACCENT for r in d.error_rate])
    ax.set_xticks(range(len(d)), [f"{i}\n(n={n})" for i, n in zip(d.index, d.files)], fontsize=7, rotation=90)
    ax.set_ylabel("error rate"); ax.set_title("PatchCore seed 42: test error rate per date (dates with ≥3 files)",
                                              fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "error_rate_by_date.png", dpi=200, facecolor="white"); plt.close(fig)
    return table


# --------------------------------------------------------------------- C

def calibration(out: Path, table: pd.DataFrame) -> dict:
    res = {"margin_deciles": {}, "rank_normalization": {}}
    q = pd.qcut(table.margin.abs(), 10, labels=False, duplicates="drop")
    for d in sorted(set(q)):
        m = q == d
        res["margin_deciles"][int(d)] = {"n": int(m.sum()), "error_rate": round(float(table.error[m].mean()), 3),
                                         "median_abs_margin": round(float(table.margin[m].abs().median()), 4)}

    # H2: rank-normalize each bank distance against its own VAL distribution, then re-evaluate.
    for seed in SEEDS:
        w = pd.read_csv(REPO_ROOT / f"Results/patchcore/final_k8_s{seed}/window_scores.csv", dtype={"date": str})
        ref = w[w.split == "val"]
        out_rows = []
        for split in ("val", "test"):
            part = w[w.split == split].copy()
            for c in ("d_pd", "d_noise"):
                part[f"r_{c}"] = np.searchsorted(np.sort(ref[c].to_numpy()), part[c].to_numpy()) / len(ref)
            part["s_rank"] = part.r_d_pd - part.r_d_noise
            out_rows.append(part)
        z = pd.concat(out_rows)
        f = z.groupby(["sample_id", "split", "cls"], as_index=False).agg(S_raw=("s", "mean"), S_rank=("s_rank", "mean"))
        entry = {}
        for name, col in (("raw", "S_raw"), ("rank_normalized", "S_rank")):
            v, t = f[f.split == "val"], f[f.split == "test"]
            tau = best_threshold((v.cls == "Noise").to_numpy(int), v[col].to_numpy())
            y = (t.cls == "Noise").to_numpy(int)
            entry[name] = {
                "test_auroc": round(float(roc_auc_score(y, t[col])), 4),
                "macro_f1_at_val_threshold": round(float(f1_score(y, (t[col] > tau).astype(int), average="macro")), 4),
                "macro_f1_at_test_optimal": round(float(f1_score(
                    y, (t[col] > best_threshold(y, t[col].to_numpy())).astype(int), average="macro")), 4)}
        res["rank_normalization"][f"seed{seed}"] = entry

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for kind, color in (("correct", ACCENT), ("missed PD", WARN), ("false alarm", "#e0a54a")):
        sub = table[table.error_kind == kind]
        ax.hist(sub.margin, bins=40, alpha=.65, label=f"{kind} (n={len(sub)})", color=color)
    ax.axvline(0, color="k", ls="--", lw=1, label="decision threshold")
    ax.set_xlabel("margin  (file score − val threshold);  > 0 predicted Noise")
    ax.set_ylabel("files"); ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("PatchCore seed 42: where errors sit relative to the threshold", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(out / "margin_hist.png", dpi=200, facecolor="white"); plt.close(fig)
    (out / "calibration.json").write_text(json.dumps(res, indent=2))
    return res


# --------------------------------------------------------------------- D

def aggregation_study(out: Path) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        w = pd.read_csv(REPO_ROOT / f"Results/patchcore/final_k8_s{seed}/window_scores.csv", dtype={"date": str})
        g = w.groupby(["sample_id", "split", "cls"], as_index=False)
        rules = {a: g.s.agg(a if a != "p90" else (lambda x: x.quantile(0.9))).rename(columns={"s": "S"}) for a in AGGS}
        # count-of-PD-like-windows rules: how many of the 28 windows fall on the PD side of 0
        for k in (1, 3, 5, 8):
            c = g.s.agg(lambda x, k=k: -float((x < 0).sum() >= k)).rename(columns={"s": "S"})
            rules[f"count>={k}_PD_windows"] = c
        for name, f in rules.items():
            v, t = f[f.split == "val"], f[f.split == "test"]
            yv, yt = (v.cls == "Noise").to_numpy(int), (t.cls == "Noise").to_numpy(int)
            tau = best_threshold(yv, v.S.to_numpy())
            pred = (t.S.to_numpy() > tau).astype(int)
            missed = int(((yt == 0) & (pred == 1)).sum()); false_alarm = int(((yt == 1) & (pred == 0)).sum())
            rows.append({"seed": seed, "rule": name,
                         "val_auroc": round(float(roc_auc_score(yv, v.S)), 4),
                         "val_macro_f1": round(float(f1_score(yv, (v.S.to_numpy() > tau).astype(int), average="macro")), 4),
                         "test_auroc": round(float(roc_auc_score(yt, t.S)), 4),
                         "test_macro_f1": round(float(f1_score(yt, pred, average="macro")), 4),
                         "test_missed_pd": missed, "test_false_alarms": false_alarm})
    df = pd.DataFrame(rows)
    df.to_csv(out / "aggregation_study.csv", index=False)
    return df


# --------------------------------------------------------------------- I

def bootstrap(out: Path, n: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    scores = {}
    for name, pat in METHODS.items():
        f, col = load_eval(pat.format(seed=42))
        t = f[f.split == "test"]
        scores[name] = t.set_index("sample_id")[[col, "cls", "date"]].rename(columns={col: "S"})
    f, col = load_eval(BASELINE[1])
    t = f[f.split == "test"]
    scores[BASELINE[0]] = t.set_index("sample_id")[[col, "cls", "date"]].rename(columns={col: "S"})

    dates = scores["PatchCore"].date.unique()
    draws = {k: [] for k in scores}
    diffs = []
    for _ in range(n):
        pick = rng.choice(dates, size=len(dates), replace=True)
        idx = np.concatenate([scores["PatchCore"].index[scores["PatchCore"].date == d].to_numpy() for d in pick])
        vals = {}
        for k, df in scores.items():
            sub = df.loc[idx]
            y = (sub.cls == "Noise").to_numpy(int)
            if y.min() == y.max():
                vals = {}; break
            vals[k] = roc_auc_score(y, sub.S)
        if not vals:
            continue
        for k, v in vals.items():
            draws[k].append(v)
        diffs.append({k: vals["PatchCore"] - vals[k] for k in vals if k != "PatchCore"})
    res = {k: {"auroc_median": round(float(np.median(v)), 4),
               "ci95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]}
           for k, v in draws.items()}
    dd = pd.DataFrame(diffs)
    res["paired_differences_vs_PatchCore"] = {
        k: {"median": round(float(dd[k].median()), 4),
            "ci95": [round(float(dd[k].quantile(.025)), 4), round(float(dd[k].quantile(.975)), 4)],
            "P(PatchCore better)": round(float((dd[k] > 0).mean()), 3)} for k in dd.columns}
    res["note"] = "date-level bootstrap, seed-42 runs, 2000 resamples; paired = same resampled dates"
    (out / "bootstrap.json").write_text(json.dumps(res, indent=2))
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "Results/patchcore/error_analysis")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    table = build_tables(args.out)
    print("== A/B/H ==")
    print("errors (seed 42):", int(table.error.sum()), "of", len(table),
          "|", table.error_kind.value_counts().drop("correct").to_dict())
    print("consistent across 3 seeds:", int((table.pc_err_count == 3).sum()),
          "| exactly one seed:", int((table.pc_err_count == 1).sum()))
    print("\ntop error dates:\n", pd.read_csv(args.out / "error_by_date.csv").head(5).to_string(index=False))
    print("\nerror overlap:\n", pd.read_csv(args.out / "method_overlap.csv", index_col=0).to_string())

    cal = calibration(args.out, table)
    print("\n== C: rank normalization (H2) ==")
    for s, e in cal["rank_normalization"].items():
        print(f"  {s}: raw AUROC {e['raw']['test_auroc']} F1@val-thr {e['raw']['macro_f1_at_val_threshold']}"
              f"  ->  rank AUROC {e['rank_normalized']['test_auroc']} F1@val-thr {e['rank_normalized']['macro_f1_at_val_threshold']}")

    agg = aggregation_study(args.out)
    print("\n== D: aggregation rules (median over 3 seeds, sorted by val AUROC) ==")
    print(agg.groupby("rule")[["val_auroc", "val_macro_f1", "test_auroc", "test_macro_f1", "test_missed_pd",
                               "test_false_alarms"]].median().sort_values("val_auroc", ascending=False).to_string())

    bs = bootstrap(args.out)
    print("\n== I: date-level bootstrap 95% CI ==")
    for k, v in bs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
