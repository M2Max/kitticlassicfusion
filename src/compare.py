"""Confronto fra tutti i metodi gia' lanciati con run.py su uno split: tabella, F1 e matrici di confusione.

Uso: .venv/bin/python src/compare.py [--split val]
"""
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

from config import CLASSES, RESULTS
from evaluation.confusion import confusion_matrix, plot_confusion
from evaluation.iou import iou_3d, iou_bev
from evaluation.result_analysis import collect, load_run, save_fig, setup_style, summary_table

ORDER = ["L1", "C1", "FE", "FM", "FL-c", "L2"]               # ordine delle righe; metodi sconosciuti in fondo


def find_runs(split):
    found = {p.parent.name: p for p in (RESULTS / split).glob("*/run.pkl")}
    names = [m for m in ORDER if m in found] + sorted(m for m in found if m not in ORDER)
    return {m: load_run(found[m]) for m in names}

def check_same_frames(runs):
    # Il confronto ha senso solo se tutti i metodi hanno visto gli stessi frame
    frames = {m: [r["frame"] for r in recs] for m, recs in runs.items()}
    ref_name, ref = next(iter(frames.items()))
    for m, f in frames.items():
        if f != ref:
            raise SystemExit(f"{m} ha {len(f)} frame diversi da {ref_name} ({len(ref)}): rilancia con lo stesso split/--limit")

def comparison_rows(runs):
    rows = []
    for m, recs in runs.items():
        s3d = summary_table(collect(recs, iou_3d))
        sbev = summary_table(collect(recs, iou_bev))
        t = np.mean([r["time_s"] for r in recs]) if all("time_s" in r for r in recs) else np.nan
        for a, b in zip(s3d, sbev):
            rows.append({"method": m, **a, "F1_bev": b["F1"], "time_s": t})
    return rows

def save_table(rows, n_frames, out):
    with open(out / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = [f"Confronto su {n_frames} frame - difficolta' Moderate, soglie IoU 0.7 / 0.5 / 0.5", "",
             "| metodo | classe | TP | FP | FN | precision | recall | F1 3D | F1 BEV | s/frame |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['class']} | {r['TP']} | {r['FP']} | {r['FN']} | {r['precision']:.3f} | "
                     f"{r['recall']:.3f} | {r['F1']:.3f} | {r['F1_bev']:.3f} | {r['time_s']:.2f} |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

def plot_f1(rows, methods, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    x = np.arange(len(CLASSES))
    width = 0.8 / len(methods)
    for ax, key, name in ((axes[0], "F1", "IoU 3D"), (axes[1], "F1_bev", "IoU BEV")):
        for k, m in enumerate(methods):
            vals = [next(r[key] for r in rows if r["method"] == m and r["class"] == c) for c in CLASSES]
            ax.bar(x + (k - (len(methods) - 1) / 2) * width, vals, width=width, label=m, edgecolor="white")
        ax.set_xticks(x, CLASSES)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"F1 per classe ({name})")
    axes[0].set_ylabel("F1 (0-1)")
    axes[1].legend(loc="upper right", fontsize=8)
    save_fig(fig, out / "comparison_f1.png")

def plot_confusions(runs, out, per_row=3):
    rows = (len(runs) + per_row - 1) // per_row
    cols = min(len(runs), per_row)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows), squeeze=False)
    for ax, (m, recs) in zip(axes.ravel(), runs.items()):
        plot_confusion(ax, confusion_matrix(recs, iou_3d), m)
    for ax in axes.ravel()[len(runs):]:                     # caselle avanzate nell'ultima riga
        ax.axis("off")
    save_fig(fig, out / "comparison_confusion.png")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    runs = find_runs(args.split)
    if not runs:
        raise SystemExit(f"nessuna run in {RESULTS / args.split}: prima lancia src/run.py")
    check_same_frames(runs)
    out = RESULTS / args.split
    setup_style()
    rows = comparison_rows(runs)
    save_table(rows, len(next(iter(runs.values()))), out)
    plot_f1(rows, list(runs), out)
    plot_confusions(runs, out)
    print(f"\nConfronto salvato in {out.resolve()}")


if __name__ == "__main__":
    main()
