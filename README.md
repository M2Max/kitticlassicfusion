# Bayesian LiDAR-Stereo Occupancy Fusion for 3D Object Detection on KITTI

Detection 3D classica (senza machine learning) sul benchmark KITTI 3D Object Detection.
Alla nuvola LiDAR si aggiunge una nuvola ricavata dalla coppia stereo a colori con block matching
classico; le due modalità vengono fuse con un aggiornamento bayesiano in log-odds sulla griglia BEV
usata per il clustering. Clustering, box fitting, classificazione e valutazione (IoU 3D contro
`label_2`, difficoltà Moderate) restano quelli del progetto precedente. I risultati confronteranno
LiDAR da solo, stereo da solo e fusione.

## Stato

Baseline LiDAR-only copiata da `kittilidardetection`, **fusione da implementare**.
Su 50 frame dà detection e conteggi TP/FP/FN identici al repo di origine.
Le anomalie notate durante la copia (non corrette) sono in [NOTES.md](NOTES.md).

## Struttura

```
src/
  config.py            costanti condivise e percorso del dataset (KITTI_ROOT)
  main.py              run completa LiDAR-only + tabelle, grafici, video
  kitti/               lettura file, calibrazione LiDAR <-> camera, ground truth
  pipeline/            suolo (RANSAC), griglia BEV + clustering, box fitting, classi
  detectors/lidar.py   detector LiDAR-only (in futuro: stereo.py, fusion.py)
  evaluation/          IoU BEV/3D, matching, tabelle, grafici e video
tests/                 test geometrici e di valutazione (unittest)
```

Le parti nuove andranno in moduli nuovi: `src/stereo/` (block matching, disparità -> nuvola),
`src/fusion/` (occupancy log-odds sulla griglia di `pipeline/bev.py`) e i detector in `src/detectors/`.

## Dati

Il dataset non sta nel repo: il percorso si imposta con la variabile d'ambiente `KITTI_ROOT`
(default `data/kitti` nella radice del repo, anche come link simbolico).

```
$KITTI_ROOT/
  training/
    calib/      000000.txt ...   (P0-P3, R0_rect, Tr_velo_to_cam)
    label_2/    000000.txt ...
    velodyne/   000000.bin ...
  ImageSets/    train.txt val.txt trainval.txt test.txt   (non usati dal codice attuale)
```

Per la parte stereo serviranno anche le immagini a colori della coppia stereo (da
https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d, non ancora scaricate):

- `data_object_image_2.zip` (~12 GB) -> `training/image_2/` (camera sinistra, quella delle label)
- `data_object_image_3.zip` (~12 GB) -> `training/image_3/` (camera destra)

La calibrazione (`P2`, `P3`) è già in `training/calib/`.

## Uso

```bash
uv venv --python 3.14 && uv pip install -r requirements.txt
export KITTI_ROOT=/percorso/di/kitti
.venv/bin/python src/main.py
```

Da lanciare dalla radice del repo. La run completa (7480 frame, ~0.4 s a frame) richiede circa un'ora;
salva i record in `results/run.pkl` e scrive tabelle, grafici e `results/sensor_view.mp4`.
Se `results/run.pkl` esiste già, `main.py` lo ricarica invece di rilanciare il detector (`RECOMPUTE`).

Test (quelli di valutazione vengono saltati se il dataset non c'è):

```bash
.venv/bin/python -m unittest discover -s tests -v
```
