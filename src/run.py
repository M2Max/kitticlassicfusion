"""Lancia un metodo su uno split KITTI: record, tabelle e grafici in results/<split>/<metodo>/.

Uso: .venv/bin/python src/run.py L1 [--split val] [--limit 50] [--recompute] [--workers 1] [--video]
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

from config import RESULTS
from detectors import lidar
from evaluation.iou import iou_3d
from evaluation.result_analysis import analyze_run, load_run, make_sensor_video, save_run
from kitti.data import frame_ids, load_calib
from kitti.ground_truth import gt_boxes

METHODS = {                                                 # metodo -> (detect, nuvola per grafici e video)
    "L1": (lidar.detect, lidar.cloud),
}


def run_frame(method, i):
    calib = load_calib(i)
    t0 = time.perf_counter()
    boxes, classes, scores = METHODS[method][0](i, calib)
    elapsed = time.perf_counter() - t0                      # tempo del solo detector
    gt_b, gt_c, gt_ign = gt_boxes(i, calib)
    return {"frame": i, "det_boxes": boxes, "det_classes": classes, "det_scores": scores,
            "gt_boxes": gt_b, "gt_classes": gt_c, "gt_ignored": gt_ign, "time_s": elapsed}

def run_method(method, frames, workers):
    records = []
    with ProcessPoolExecutor(workers) as pool:
        for rec in pool.map(run_frame, [method] * len(frames), frames, chunksize=8):   # stesso ordine di frames
            records.append(rec)
            if len(records) % 200 == 0 or len(records) == len(frames):
                print(f"  {method}: {len(records)}/{len(frames)} frame", flush=True)
    return records

def method_dir(split, method):
    return RESULTS / split / method

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=sorted(METHODS))
    ap.add_argument("--split", default="val", help="file in ImageSets/ (default: val)")
    ap.add_argument("--limit", type=int, help="solo i primi N frame dello split (per prove veloci)")
    ap.add_argument("--recompute", action="store_true", help="ignora run.pkl e rilancia il detector")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="processi in parallelo (time_s e' piu' fedele con 1)")
    ap.add_argument("--video", action="store_true", help="scrive anche sensor_view.mp4 (lento)")
    args = ap.parse_args()

    out = method_dir(args.split, args.method)
    run_file = out / "run.pkl"
    frames = frame_ids(args.split)[:args.limit]
    records = load_run(run_file) if run_file.exists() and not args.recompute else None
    if records is not None and [r["frame"] for r in records] != frames:
        print(f"{run_file} ha altri frame (es. un --limit diverso): rilancio")
        records = None
    if records is None:
        records = run_method(args.method, frames, args.workers)
        save_run(records, run_file)

    cloud = METHODS[args.method][1]
    analyze_run(records, iou_3d, out_dir=out, load_points=cloud)         # tabelle + grafici
    if args.video:
        make_sensor_video(records, cloud, out / "sensor_view.mp4", fps=10)


if __name__ == "__main__":
    main()
