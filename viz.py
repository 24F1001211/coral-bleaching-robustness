"""Charts for the experiment results (shared by report.py and the live app).

Colour use: blue / orange / aqua are the first three slots of a validated categorical
palette (safe for colour-blind readers). Heat maps use a single-hue blue ramp.
Every chart has a legend or direct labels, and the full numbers are in the table view.
"""
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "text.color": INK,
})


# ----------------------------------------------------------------------------- loading
def load_runs(runs_dir):
    runs = {}
    for d in sorted(Path(runs_dir).glob("*")):
        f = d / "results.json"
        if not f.exists():
            continue
        r = {"dir": d, "results": json.loads(f.read_text())}
        for key, fn in [("probes", "probes.json"), ("colour", "colour_baseline.json")]:
            p = d / fn
            r[key] = json.loads(p.read_text()) if p.exists() else None
        runs[d.name] = r
    return runs


def class_names(run):
    return ["healthy", "bleached", "dead"] if run["results"]["args"]["scheme"] == "3class" else ["healthy", "unhealthy"]


def _kind(eval_name):
    return "cross-dataset" if "cross-dataset" in eval_name else "in-domain"


def _style(ax, grid_axis="y"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _wrap(s, w=22):
    return textwrap.fill(s, w)


# ----------------------------------------------------------------------------- charts
def fig_overview(runs):
    """In-domain vs cross-dataset: the headline chart. One row per run."""
    names = list(runs)
    metrics = [("macro_f1", "Macro-F1 (higher is better)"),
               ("mae", "Class error, MAE (lower is better)"),
               ("ece", "Calibration error, ECE (lower is better)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, max(3.0, 0.8 * len(names) + 1.9)), sharey=True)
    y = np.arange(len(names))
    h = 0.36
    for ax, (key, title) in zip(axes, metrics):
        allv = []
        for off, kind, color in [(-h / 2, "in-domain", BLUE), (h / 2, "cross-dataset", ORANGE)]:
            vals = np.array([
                np.mean([m[key] for k, m in runs[n]["results"]["eval"].items() if _kind(k) == kind] or [np.nan])
                for n in names])
            allv += list(vals[~np.isnan(vals)])
            ax.barh(y + off, np.nan_to_num(vals), height=h - 0.05, color=color, edgecolor=SURFACE, linewidth=2)
            for yy, v in zip(y + off, vals):
                if not np.isnan(v):
                    ax.text(v + 0.012, yy, f"{v:.2f}", va="center", fontsize=9, color=INK2)
        top = 1.0 if key == "macro_f1" else max(0.05, max(allv or [0.05]))
        ax.set_xlim(0, top * 1.18)
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
        _style(ax, "x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([_wrap(n, 20) for n in names], fontsize=9, color=INK2)
    axes[0].invert_yaxis()
    fig.legend(handles=[Patch(color=BLUE, label="In-domain test (same dataset as training)"),
                        Patch(color=ORANGE, label="Cross-dataset (unseen dataset)")],
               loc="upper right", frameon=False, ncol=2, fontsize=9, bbox_to_anchor=(0.99, 1.0))
    H = fig.get_figheight()
    fig.text(0.01, 1 - 0.28 / H, "Does accuracy survive a different dataset?", ha="left", va="center",
             fontsize=13, fontweight="bold")
    fig.text(0.01, 1 - 0.58 / H, "Each bar is the mean over that run's evaluation sets of that kind; per-dataset values are in the table.",
             ha="left", va="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.8 / H])
    return fig


def fig_confusion(name, run):
    sets = run["results"]["eval"]
    classes = class_names(run)
    n = len(sets)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.4), squeeze=False)
    for ax, (k, m) in zip(axes[0], sets.items()):
        cm = np.array(m["confusion"], float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=10,
                        color="#ffffff" if norm[i, j] > 0.5 else INK)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=9, color=INK2)
        ax.set_yticklabels(classes, fontsize=9, color=INK2)
        ax.set_xlabel("Predicted", fontsize=9, color=MUTED)
        ax.set_ylabel("True", fontsize=9, color=MUTED)
        ax.set_title(f"{_wrap(k, 28)}\nacc {m['accuracy']:.2f} · F1 {m['macro_f1']:.2f}", fontsize=9.5, color=INK, loc="left")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"Confusion matrices: {name}   (cell = count and share of the true class)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def _grouped(ax, groups, series, colors, labels):
    x = np.arange(len(groups))
    w = 0.8 / len(series)
    for i, (vals, c) in enumerate(zip(series, colors)):
        pos = x + (i - (len(series) - 1) / 2) * w
        ax.bar(pos, vals, width=w - 0.04, color=c, edgecolor=SURFACE, linewidth=2)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([_wrap(g, 22) for g in groups], fontsize=9, color=INK2)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Macro-F1", fontsize=9, color=MUTED)
    _style(ax, "y")
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors, labels)], frameon=False, fontsize=9,
              loc="lower left", ncol=len(series), bbox_to_anchor=(0.0, 1.0), borderaxespad=0.3)


def fig_probes(name, run):
    pr = run["probes"]
    groups = list(pr)
    series = [[pr[g][k]["macro_f1"] for g in groups] for k in ("normal", "grayscale", "patch_shuffle")]
    fig, ax = plt.subplots(figsize=(max(6.5, 2.8 * len(groups)), 4.4))
    _grouped(ax, groups, series, [BLUE, ORANGE, AQUA],
             ["Normal", "Grayscale (colour removed)", "Patch-shuffled (shape destroyed)"])
    fig.suptitle(f"Stress tests: {name}", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.01, "Big drop in grayscale = relies on colour. Small drop when shuffled = relies on colour statistics, not coral shape.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    return fig


def fig_colour(name, run):
    co = run["colour"]
    groups = list(co)
    cnn = [run["results"]["eval"][g]["macro_f1"] for g in groups]
    col = [co[g]["macro_f1"] for g in groups]
    fig, ax = plt.subplots(figsize=(max(6.0, 2.6 * len(groups)), 4.4))
    _grouped(ax, groups, [cnn, col], [BLUE, ORANGE], ["CNN", "Colour-only baseline (logistic regression)"])
    fig.suptitle(f"CNN vs colour-only baseline: {name}", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.01, "If the colour-only baseline is close to the CNN, the task is mostly solvable from colour alone.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    return fig


def results_table(runs):
    rows = []
    for n, r in runs.items():
        for k, m in r["results"]["eval"].items():
            rows.append({
                "Run": n, "Evaluation set": k, "Type": _kind(k), "Images": m["n"],
                "Accuracy": m["accuracy"], "Macro-F1": m["macro_f1"], "MAE": m["mae"], "ECE": m["ece"],
                "Colour-only F1": r["colour"][k]["macro_f1"] if r["colour"] and k in r["colour"] else np.nan,
                "Grayscale F1": r["probes"][k]["grayscale"]["macro_f1"] if r["probes"] and k in r["probes"] else np.nan,
                "Shuffled F1": r["probes"][k]["patch_shuffle"]["macro_f1"] if r["probes"] and k in r["probes"] else np.nan,
            })
    return pd.DataFrame(rows).round(3)


# ----------------------------------------------------------------------------- report
def save_all(runs_dir, out_dir):
    runs = load_runs(runs_dir)
    if not runs:
        raise SystemExit(f"No finished runs (results.json) found in {runs_dir}/. Train first.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sections = []

    def save(fig, fname, title, alt):
        fig.savefig(out / fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        sections.append((title, fname, alt))

    save(fig_overview(runs), "overview.png", "Overview: in-domain vs cross-dataset",
         "Bar charts of macro-F1, MAE and ECE for every run, in-domain versus cross-dataset")
    for n, r in runs.items():
        save(fig_confusion(n, r), f"confusion_{n}.png", f"Confusion matrices: {n}", f"Confusion matrices for run {n}")
        if r["probes"]:
            save(fig_probes(n, r), f"probes_{n}.png", f"Stress tests: {n}", f"Macro-F1 under grayscale and patch-shuffle for run {n}")
        if r["colour"]:
            save(fig_colour(n, r), f"colour_{n}.png", f"CNN vs colour-only: {n}", f"CNN versus colour-only baseline for run {n}")
        g = r["dir"] / "gradcam.png"
        if g.exists():
            (out / f"gradcam_{n}.png").write_bytes(g.read_bytes())
            sections.append((f"Grad-CAM attention: {n}", f"gradcam_{n}.png", f"Grad-CAM heat maps for run {n}"))

    df = results_table(runs)
    df.to_csv(out / "results_table.csv", index=False)
    body = "".join(f'<section><h2>{t}</h2><img src="{f}" alt="{a}"></section>' for t, f, a in sections)
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coral bleaching robustness report</title>
<style>body{{font:15px system-ui,-apple-system,"Segoe UI",sans-serif;background:#f9f9f7;color:#0b0b0b;margin:0;padding:24px;max-width:1100px;margin:auto}}
section{{background:#fcfcfb;border:1px solid #e1e0d9;border-radius:10px;padding:16px;margin:16px 0}}img{{max-width:100%}}
table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{border-bottom:1px solid #e1e0d9;padding:6px 8px;text-align:left}}th{{color:#52514e}}</style>
<h1>Coral bleaching robustness: results</h1>
<section><h2>Results table (all numbers)</h2>{df.to_html(index=False, na_rep="–", border=0)}</section>{body}</html>"""
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"report written to {out}/index.html ({len(sections)} charts)")
