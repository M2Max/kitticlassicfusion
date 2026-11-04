"""Stesso frame visto da tutti i metodi: foto della camera e vista dall'alto, scatole colorate per esito.

Uso:
    .venv/bin/python src/compare_views.py --frames 8 15 42      # results/val/views/frame_<n>_camera.png e _bev.png
    .venv/bin/python src/compare_views.py --pick 6              # i 6 frame dove i metodi differiscono di piu'
    .venv/bin/python src/compare_views.py --video 200 --fps 4   # results/val/views/comparison.mp4
"""
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from compare import find_runs
from config import CLASSES, IOU_THRESH, RESULTS
from evaluation.iou import iou_3d
from evaluation.result_analysis import (BOX_EDGES, box_corners_3d, class_indices, draw_bev_box, hex_to_bgr,
                                        iou_table, open_writer, save_fig, score_frame, setup_style)
from kitti.data import load_calib, load_image, load_lidar
from kitti.transforms import lidar_to_rectcam

OUTCOME = {"TP": "#2CA02C", "FP": "#D62728", "FN": "#F2B701", "ignored": "#9A9A9A"}   # verde, rosso, giallo, grigio
PANEL = (620, 188)                                          # misura di ogni foto nella griglia (le foto KITTI variano)
BAR = 26                                                    # barra del titolo sopra ogni foto


def outcomes(rec):
    # Esito di ogni det (TP, FP, o "ignored" se copre un oggetto ignorato) e di ogni GT valida (TP o FN)
    det = ["ignored"] * len(rec["det_boxes"])
    gt = ["ignored"] * len(rec["gt_boxes"])
    for c in CLASSES:
        d, valid, ignored = class_indices(rec, c)
        f = {"dets": rec["det_boxes"][d], "valid": rec["gt_boxes"][valid],
             "iou_v": iou_table(rec["det_boxes"][d], rec["gt_boxes"][valid], iou_3d),
             "iou_i": iou_table(rec["det_boxes"][d], rec["gt_boxes"][ignored], iou_3d)}
        pairs, fp, fn = score_frame(f, IOU_THRESH[c])
        for i, j in pairs:
            det[d[i]], gt[valid[j]] = "TP", "TP"
        for i in fp:
            det[d[i]] = "FP"
        for j in fn:
            gt[valid[j]] = "FN"
    return det, gt

def counts(det, gt):
    return f"TP {det.count('TP')}  FP {det.count('FP')}  FN {gt.count('FN')}"

def draw_on_image(img, box, calib, color, thickness):
    pts = lidar_to_rectcam(calib, box_corners_3d(box))
    if (pts[:, 2] < 0.5).any():                             # qualche spigolo dietro la camera
        return
    uvw = np.c_[pts, np.ones(8)] @ calib["P2"].reshape(3, 4).T
    uv = np.round(uvw[:, :2] / uvw[:, 2:]).astype(int)
    for a, b in BOX_EDGES:
        cv2.line(img, tuple(uv[a]), tuple(uv[b]), color, thickness, cv2.LINE_AA)

def camera_panel(img, rec, calib, name):
    det, gt = outcomes(rec)
    panel = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for box, o in zip(rec["gt_boxes"], gt):
        if o == "FN":
            draw_on_image(panel, box, calib, hex_to_bgr(OUTCOME["FN"]), 2)
    for box, o in zip(rec["det_boxes"], det):
        draw_on_image(panel, box, calib, hex_to_bgr(OUTCOME[o]), 2)
    panel = cv2.resize(panel, PANEL, interpolation=cv2.INTER_AREA)
    bar = np.full((BAR, PANEL[0], 3), 20, np.uint8)
    cv2.putText(bar, f"{name}   {counts(det, gt)}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])

def camera_grid(frame, runs, cols=3):
    calib = load_calib(frame)
    img = load_image(frame)
    panels = [camera_panel(img, recs[frame], calib, m) for m, recs in runs.items()]
    panels += [np.zeros_like(panels[0])] * (-len(panels) % cols)   # caselle vuote nell'ultima riga
    rows = [np.hstack(panels[k:k + cols]) for k in range(0, len(panels), cols)]
    legend = np.full((BAR, rows[0].shape[1], 3), 20, np.uint8)
    text = f"frame {frame:06d}    green = correct (TP)   red = invented (FP)   yellow = missed (FN)   grey = on an ignored object"
    cv2.putText(legend, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack(rows + [legend])

def bev_grid(frame, runs, path, cols=3):
    points = load_lidar(frame)[:, :3]
    keep = (points[:, 0] > 0) & (points[:, 0] < 60) & (np.abs(points[:, 1]) < 25)
    pts = points[keep][:: max(1, int(keep.sum()) // 30000)]
    rows = (len(runs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5.6 * rows), squeeze=False)
    for ax, (m, recs) in zip(axes.ravel(), runs.items()):
        rec = recs[frame]
        det, gt = outcomes(rec)
        ax.scatter(-pts[:, 1], pts[:, 0], s=0.15, c="#B0B0B0", linewidths=0)
        for box, o in zip(rec["gt_boxes"], gt):
            color = {"TP": "#222222", "FN": OUTCOME["FN"], "ignored": "#9A9A9A"}[o]   # trovata, persa, ignorata
            draw_bev_box(ax, box, color=color, lw=1.4, ls=":" if o == "ignored" else "-")
        for box, o in zip(rec["det_boxes"], det):
            draw_bev_box(ax, box, color=OUTCOME[o], lw=1.6)
        ax.set_xlim(-25, 25)
        ax.set_ylim(0, 60)
        ax.set_aspect("equal")
        ax.grid(True, axis="both", alpha=0.3)
        ax.set_title(f"{m}   {counts(det, gt)}", fontsize=11)
    for ax in axes.ravel()[len(runs):]:
        ax.axis("off")
    legend = [Line2D([], [], color="#222222", lw=1.4, label="label, found"),
              Line2D([], [], color=OUTCOME["FN"], lw=1.4, label="label, missed (FN)"),
              Line2D([], [], color="#9A9A9A", lw=1.4, ls=":", label="label ignored by KITTI"),
              Line2D([], [], color=OUTCOME["TP"], lw=2, label="detection, correct (TP)"),
              Line2D([], [], color=OUTCOME["FP"], lw=2, label="detection, invented (FP)"),
              Line2D([], [], color=OUTCOME["ignored"], lw=2, label="detection on ignored label")]
    axes[0, 0].legend(handles=legend, loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle(f"Frame {frame:06d} from above (LiDAR points in grey)", fontweight="bold")
    save_fig(fig, path)

def pick_frames(runs, n):
    # Frame dove i metodi si comportano piu' diversamente: differenza fra il migliore e il peggiore in TP - FP
    frames = next(iter(runs.values())).keys()
    score = {}
    for f in frames:
        net = []
        for recs in runs.values():
            det, _ = outcomes(recs[f])
            net.append(det.count("TP") - det.count("FP"))
        score[f] = max(net) - min(net)
    return sorted(score, key=score.get, reverse=True)[:n]

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val")
    ap.add_argument("--frames", type=int, nargs="*", default=[])
    ap.add_argument("--pick", type=int, help="sceglie da solo N frame fra i primi 500")
    ap.add_argument("--video", type=int, help="video sui primi N frame dello split")
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()

    runs = {m: {r["frame"]: r for r in recs} for m, recs in find_runs(args.split).items()}
    out = RESULTS / args.split / "views"
    out.mkdir(parents=True, exist_ok=True)
    setup_style()
    frames = list(args.frames)
    if args.pick:
        first = {m: dict(list(recs.items())[:500]) for m, recs in runs.items()}
        frames += pick_frames(first, args.pick)
        print("frame scelti:", frames)
    for f in frames:
        cv2.imwrite(str(out / f"frame_{f:06d}_camera.png"), camera_grid(f, runs))
        bev_grid(f, runs, out / f"frame_{f:06d}_bev.png")
    if args.video:
        video_frames = list(next(iter(runs.values())))[:args.video]
        first = camera_grid(video_frames[0], runs)
        writer = open_writer(out / "comparison.mp4", args.fps, (first.shape[1], first.shape[0]))
        for f in video_frames:
            writer.write(camera_grid(f, runs))
        writer.release()
        print(f"video: {len(video_frames)} frame a {args.fps} fps")
    print(f"Salvato in {out}")


if __name__ == "__main__":
    main()
