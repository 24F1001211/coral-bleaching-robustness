"""Live demo + results dashboard.

  python app.py --runs runs          # opens http://127.0.0.1:7860 in your browser

Run it from the project folder (the runs store relative image paths).
Tab 1: webcam / upload / random test image -> prediction, Grad-CAM, stress tests, colour-only check.
Tab 2: results dashboard for every run in runs/.
"""
import argparse
import io
import json
import random
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import viz
from infer import Detector

RUNS = Path("runs")
_cache = {}

BADGE = {  # icon + label + colour: status is never colour alone
    "healthy": ("✔", "Healthy", "#0ca30c", "#ffffff"),
    "bleached": ("⚠", "Bleached", "#fab219", "#0b0b0b"),
    "dead": ("✖", "Dead", "#d03b3b", "#ffffff"),
    "unhealthy": ("⚠", "Unhealthy", "#fab219", "#0b0b0b"),
}


def list_runs():
    return [d.name for d in sorted(RUNS.glob("*")) if (d / "model.pt").exists()]


def get_detector(name):
    if name not in _cache:
        _cache[name] = Detector(RUNS / name)
    return _cache[name]


def _img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ----------------------------------------------------------------------------- live detection
def detect(img, run_name, show_cam=True):
    if img is None or not run_name:
        return "<i>Waiting for an image…</i>", None, None, ""
    det = get_detector(run_name)
    cls = det.classes
    p = det.probs(img)
    top = int(np.argmax(p))
    icon, text, bg, fg = BADGE.get(cls[top], ("●", cls[top], "#52514e", "#ffffff"))
    low = " <small>(low confidence)</small>" if p[top] < 0.6 else ""
    badge = (f'<div style="display:inline-flex;gap:10px;align-items:center;padding:10px 18px;border-radius:10px;'
             f'background:{bg};color:{fg};font:600 22px system-ui,sans-serif">{icon} {text} · {p[top]:.0%}{low}</div>')

    overlay = det.cam_overlay(img) if show_cam else None
    pg, ps, pc = det.probs(img, "grayscale"), det.probs(img, "patch_shuffle"), det.colour_probs(img)
    rows = [("CNN on this image", p), ("CNN, colour removed (grayscale)", pg),
            ("CNN, shape destroyed (patch-shuffled)", ps), ("Colour-only baseline (no CNN)", pc)]
    md = "| Test | Prediction | Confidence |\n|---|---|---|\n" + "\n".join(
        f"| {n} | {cls[int(np.argmax(q))]} | {q.max():.0%} |" for n, q in rows)

    notes = []
    if int(np.argmax(pg)) != top:
        notes.append("⚠ The prediction changes when colour is removed, so this decision leans on colour.")
    if int(np.argmax(ps)) == top:
        notes.append("ℹ The prediction survives patch-shuffling, which suggests colour statistics matter more than coral shape here.")
    if int(np.argmax(pc)) == top:
        notes.append("ℹ A model that sees only colour histograms gives the same answer.")
    if notes:
        md += "\n\n" + "\n\n".join(notes)
    return badge, overlay, {c: float(v) for c, v in zip(cls, p)}, md


def random_test_image(run_name):
    if not run_name:
        return None, ""
    items = json.loads((RUNS / run_name / "items.json").read_text())["eval"]
    set_name = random.choice(list(items))
    existing = [(p, l) for p, l in items[set_name] if Path(p).exists()]
    det = get_detector(run_name)
    if existing:
        path, label = random.choice(existing)
        return Image.open(path).convert("RGB"), f"True label: **{det.classes[label]}** · from *{set_name}*"
    local_imgs = list(Path("data").rglob("*.jpg")) + list(Path("data").rglob("*.png"))
    if local_imgs:
        p = random.choice(local_imgs)
        return Image.open(p).convert("RGB"), f"Sample local image from *{p.parent.name}*"
    return None, "Please upload an image to test."


# ----------------------------------------------------------------------------- dashboard
def render_overview():
    runs = viz.load_runs(RUNS)
    if not runs:
        return None, pd.DataFrame({"info": ["No finished runs in runs/. Train first (see README)."]}), gr.Dropdown(choices=[], value=None)
    names = list(runs)
    return _img(viz.fig_overview(runs)), viz.results_table(runs), gr.Dropdown(choices=names, value=names[0])


def render_run(name):
    runs = viz.load_runs(RUNS)
    if name not in runs:
        return None, None, None, None
    r = runs[name]
    g = r["dir"] / "gradcam.png"
    return (_img(viz.fig_confusion(name, r)),
            _img(viz.fig_probes(name, r)) if r["probes"] else None,
            _img(viz.fig_colour(name, r)) if r["colour"] else None,
            Image.open(g).convert("RGB") if g.exists() else None)


def build():
    with gr.Blocks(title="Coral bleaching robustness") as demo:
        gr.Markdown("# Do coral bleaching classifiers see bleaching, or just water colour?")
        with gr.Tab("Live detection"):
            with gr.Row():
                run_dd = gr.Dropdown(choices=list_runs(), value=(list_runs() or [None])[0], label="Model (run)")
                show_cam = gr.Checkbox(value=True, label="Show Grad-CAM (slower on CPU)")
                rand_btn = gr.Button("Random test image")
            truth = gr.Markdown("")
            with gr.Row():
                with gr.Column():
                    live = gr.Image(sources=["webcam"], streaming=True, type="pil",
                                    label="Live webcam (detects continuously)")
                    inp = gr.Image(sources=["upload"], type="pil", label="Or upload an image")
                with gr.Column():
                    badge = gr.HTML()
                    overlay = gr.Image(label="Grad-CAM: where the model looks", interactive=False)
                    lab = gr.Label(label="CNN class probabilities")
            report_md = gr.Markdown()
            outs = [badge, overlay, lab, report_md]
            live.stream(detect, [live, run_dd, show_cam], outs, stream_every=1.0, time_limit=600)
            ins = [inp, run_dd, show_cam]
            inp.upload(detect, ins, outs)
            rand_btn.click(random_test_image, [run_dd], [inp, truth]).then(detect, ins, outs)
        with gr.Tab("Results dashboard"):
            refresh = gr.Button("Refresh results")
            overview = gr.Image(label="Overview", interactive=False)
            table = gr.Dataframe(label="Table view (all numbers)", interactive=False)
            run2 = gr.Dropdown(choices=[], label="Inspect one run")
            with gr.Row():
                conf = gr.Image(label="Confusion matrices", interactive=False)
                probes = gr.Image(label="Stress tests", interactive=False)
            with gr.Row():
                colour = gr.Image(label="CNN vs colour-only", interactive=False)
                cam_img = gr.Image(label="Grad-CAM grid (saved by analyze.py)", interactive=False)
            dash_outs = [overview, table, run2]
            refresh.click(render_overview, None, dash_outs)
            run2.change(render_run, [run2], [conf, probes, colour, cam_img])
            demo.load(render_overview, None, dash_outs)
        demo.load(lambda: gr.Dropdown(choices=list_runs(), value=(list_runs() or [None])[0]), None, run_dd)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    RUNS = Path(a.runs)
    build().launch(server_name="127.0.0.1", server_port=a.port, inbrowser=not a.no_browser)
