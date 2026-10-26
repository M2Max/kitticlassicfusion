# Note sulla copia da kittilidardetection

Codice copiato alla lettera dal commit `b79ae71` di `kittilidardetection`. Questo file elenca cosa
è cambiato e cosa sembra sospetto ma **non** è stato toccato.

## Modifiche fatte durante la copia

| File nuovo | Origine | Modifica |
|---|---|---|
| `src/config.py` | `src/config.py` | `DATA` ora è `KITTI_ROOT / "training"`, con `KITTI_ROOT` letto dalla variabile d'ambiente (default `<repo>/data/kitti`); aggiunto `import os` |
| `src/kitti/data.py` | `src/data.py` | nessuna |
| `src/kitti/transforms.py` | `src/transforms.py` | nessuna |
| `src/kitti/ground_truth.py` | `src/ground_truth.py` | import |
| `src/pipeline/floor.py` | `src/floor.py` | nessuna |
| `src/pipeline/bev.py` | `src/bev.py` | nessuna |
| `src/pipeline/boxes.py` | `src/boxes.py` | nessuna |
| `src/pipeline/classify.py` | `src/classify.py` | nessuna |
| `src/detectors/lidar.py` | `src/detection.py` | import |
| `src/evaluation/iou.py` | `src/iou.py` | rimossa `match` (scelta B: si usa solo il matching di `result_analysis`) |
| `src/evaluation/result_analysis.py` | `src/result_analysis.py` | `CLASSES`, `IOU_THRESH`, `NEIGHBOR` importate da `config` invece che ridefinite (valori identici) |
| `src/main.py` | `src/main.py` | import; rimosso `from evaluation import evaluate_frame, print_scores` (modulo non riportato, funzioni mai chiamate) |

Non riportati: `src/evaluation.py` (`evaluate_frame`/`print_scores`, mai chiamati da `main.py`,
duplicato del matching di `result_analysis`), `iou.match`, devkit C++/MATLAB, script di download,
dati, risultati, presentazione.

Ambiente: stesse versioni dirette del vecchio `.venv` (Python 3.14.2, numpy 2.5.3, scipy 1.18.1,
matplotlib 3.11.2, opencv-python-headless 5.0.0.93). Alcune dipendenze transitive installate da uv
sono più recenti (pillow 13.0.0 vs 12.3.0, fonttools 4.67.0 vs 4.66.1, python-dateutil 3.0.0 vs
2.9.0.post0): non toccano i numeri, al più il rendering dei grafici. `shapely` era nel vecchio
`requirements.txt` ma non è usato né installato: tolto.

## Cose sospette (non modificate)

| File:riga | Cosa | Perché è sospetto |
|---|---|---|
| `src/pipeline/bev.py:28` | `print(labels[i_r, i_theta])` in `label_points` | print di debug eseguito a ogni frame |
| `src/main.py:19` | `for i in range(7480)` | il training KITTI ha 7481 frame (000000-007480): l'ultimo non viene mai valutato; numero scritto a mano invece di `N_frames` / `n_frames()` |
| `src/config.py:4` | `N_frames = 7480` | mai usata |
| `src/kitti/data.py:20` | `n_frames()` | mai usata |
| `src/main.py:1`, `src/main.py:6` | `from config import CLASSES` poi `CLASSES, IOU_THRESH, NEIGHBOR` da `result_analysis` | import inutilizzati (e il primo `CLASSES` viene sovrascritto dal secondo) |
| `src/main.py:11-16` | `RUN_FILE` relativo alla cartella corrente, `RECOMPUTE = False` | se esiste un `results/run.pkl` vecchio i risultati non vengono ricalcolati anche se il codice è cambiato; il percorso dipende da dove si lancia |
| `src/main.py:31` | `make_sensor_video(records, ...)` su tutti i record a 1 fps | con 7480 frame il video dura ~2 ore |
| `src/evaluation/result_analysis.py:7` | docstring "video a 24 fps" | il default di `make_sensor_video` è `fps=1` (riga 410) |
| `src/evaluation/result_analysis.py:360` | `box_corners_bev` | identica a `src/evaluation/iou.py:4`: duplicato |
| `src/evaluation/result_analysis.py:90` | `np.argsort(ious, axis=None)[::-1]` in `greedy_match` | ordinamento non stabile: a parità di IoU l'ordine di abbinamento non è definito |
| `src/evaluation/iou.py:46` | `iou_bev` | mai usata dalla pipeline (ora usata solo dai test) |
| `src/pipeline/boxes.py:6` | parametro `criterion="area"` | ramo mai usato |
| `src/pipeline/floor.py:21` | `low = points[points[:, 2] < -1.5]` | soglia fissa legata all'altezza del sensore (~1.73 m); con meno di 3 punti `rng.choice` (riga 24) solleva eccezione |
| `src/pipeline/floor.py:37` | `points @ best_n + best_d` | se tutti i 500 tentativi danno piani degeneri `best_n` è `None` ed esplode |
| `src/pipeline/floor.py:23` | `for i in range(tries)` | variabile `i` non usata |
| `src/detectors/lidar.py:15` | commento "z <= 0" | il codice controlla `p[2] < 0.1` |
| `src/detectors/lidar.py:18` | `0 <= u / w < IMG_W` | controlla solo la colonna, non la riga del pixel; `IMG_W` fisso (`src/config.py:10`) mentre la larghezza varia di qualche pixel fra frame |
| `src/kitti/ground_truth.py:10` | `DontCare` scartate | le detection sulle regioni DontCare contano come FP (già dichiarato nel report del progetto 1) |
