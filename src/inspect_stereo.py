"""Stereo di un frame: foto sinistra, disparita' e nuvola stereo vista dall'alto sopra il LiDAR.

Uso: .venv/bin/python src/inspect_stereo.py 8 [15 42 ...]      -> results/inspect/stereo_<frame>.png
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS
from detectors.camera_classic import MAX_DEPTH, stereo_depth
from evaluation.result_analysis import COLORS, draw_bev_box, save_fig, setup_style
from kitti.data import load_calib, load_lidar, load_stereo_pair
from kitti.ground_truth import gt_boxes
from kitti.transforms import lidar_to_rectcam, rectcam_to_lidar
from stereo.disparity import compute_disparity
from stereo.pseudo_lidar import depth_to_points


def lidar_depth_check(depth, calib, lidar):
    # Profondita' stereo contro quella dei punti LiDAR che cadono sugli stessi pixel
    uvw = np.c_[lidar_to_rectcam(calib, lidar), np.ones(len(lidar))] @ calib["P2"].reshape(3, 4).T
    front = uvw[:, 2] > 1.0                                 # davanti alla camera
    u = (uvw[front, 0] / uvw[front, 2]).astype(int)
    v = (uvw[front, 1] / uvw[front, 2]).astype(int)
    z = uvw[front, 2]
    inside = (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
    stereo_z, lidar_z = depth[v[inside], u[inside]], z[inside]
    ok = np.isfinite(stereo_z)
    err = np.abs(stereo_z[ok] - lidar_z[ok])
    return {"pixel LiDAR": int(inside.sum()), "con stereo valido": f"{ok.mean():.0%}",
            "errore mediano (m)": f"{np.median(err):.2f}", "errore relativo mediano": f"{np.median(err / lidar_z[ok]):.1%}"}

def inspect(i, out_dir):
    calib = load_calib(i)
    left, right = load_stereo_pair(i)
    disp = compute_disparity(left, right)
    depth = stereo_depth(i, calib)                          # la stessa di C1, senza pixel volanti
    stereo = rectcam_to_lidar(calib, depth_to_points(depth, calib, MAX_DEPTH))
    lidar = load_lidar(i)[:, :3]

    fig = plt.figure(figsize=(20, 8.5))
    grid = fig.add_gridspec(2, 2, width_ratios=[2.2, 1])    # a sinistra foto e disparita', a destra dall'alto
    ax_img = fig.add_subplot(grid[0, 0])
    ax_img.imshow(left)
    ax_img.set_title(f"Frame {i:06d}: camera sinistra")
    ax_disp = fig.add_subplot(grid[1, 0])
    im = ax_disp.imshow(np.where(disp > 0, disp, np.nan), cmap="turbo")
    fig.colorbar(im, ax=ax_disp, orientation="horizontal", fraction=0.05, pad=0.02, label="disparita' (px)")
    ax_disp.set_title("Disparita' (bianco = non valida)")
    ax_bev = fig.add_subplot(grid[:, 1])
    keep = (lidar[:, 0] > 0) & (lidar[:, 0] < 70) & (np.abs(lidar[:, 1]) < 30)
    ax_bev.scatter(-lidar[keep, 1], lidar[keep, 0], s=0.2, c="#9A9A9A", linewidths=0, label="LiDAR")
    sub = stereo[:: max(1, len(stereo) // 60000)]           # al massimo ~60k punti, per velocita'
    ax_bev.scatter(-sub[:, 1], sub[:, 0], s=0.2, c="#DD8452", linewidths=0, alpha=0.5, label="stereo")
    for box, c in zip(*gt_boxes(i, calib)[:2]):
        draw_bev_box(ax_bev, box, color=COLORS.get(c, "#333333"), lw=1.5)
    ax_bev.set_xlim(-30, 30)
    ax_bev.set_ylim(0, 70)
    ax_bev.set_aspect("equal")
    ax_bev.set_xlabel("laterale (m)")
    ax_bev.set_ylabel("avanti (m)")
    ax_bev.set_title(f"Dall'alto: {len(stereo)} punti stereo,\n{len(lidar)} LiDAR, scatole = GT")
    ax_bev.legend(loc="upper right", markerscale=20)
    for ax in (ax_img, ax_disp):
        ax.axis("off")
    path = out_dir / f"stereo_{i:06d}.png"
    save_fig(fig, path)
    print(f"{path}  {lidar_depth_check(depth, calib, lidar)}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=int, nargs="+")
    args = ap.parse_args()
    out_dir = RESULTS / "inspect"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    for i in args.frames:
        inspect(i, out_dir)


if __name__ == "__main__":
    main()
