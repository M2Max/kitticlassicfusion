"""Fusione mid: griglie di evidenza di LiDAR e stereo sulla griglia BEV, sommate in log-odds."""
import numpy as np

from pipeline.bev import count_grid, label_points, labeling
from pipeline.boxes import find_boxes

GROUND = 0.2                                                # sotto e' suolo, come la soglia del RANSAC
P_HIT, MAX_HIT = 0.6, 3                                     # ogni punto LiDAR nella casella vale logit(0.6), al massimo 3 volte
P_FREE = 0.4                                                # lo stereo vede il suolo e nessun oggetto: casella libera
STEREO_BAND = (0.3, 2.5)                                    # punti stereo di oggetto: fra queste altezze sopra il suolo
L_MIN = 0.3                                                 # log-odds minimo per dire "occupata"


def logit(p):
    return np.log(p / (1 - p))

def lidar_evidence(lidar, n, d):
    # Piu' punti nella casella, piu' e' credibile che sia occupata: 1 punto isolato e' spesso rumore
    obj = lidar[lidar @ n + d >= GROUND]
    counts, i_r, i_theta, valid = count_grid(obj)
    return logit(P_HIT) * np.minimum(counts, MAX_HIT), obj[valid], i_r[valid], i_theta[valid]

def stereo_evidence(stereo, n, d):
    # Lo stereo serve solo a smentire: suolo visto e nessun oggetto -> evidenza negativa
    height = stereo @ n + d
    obj_counts = count_grid(stereo[(height >= STEREO_BAND[0]) & (height < STEREO_BAND[1])])[0]
    ground_counts = count_grid(stereo[np.abs(height) < GROUND])[0]
    return np.where((obj_counts == 0) & (ground_counts > 0), logit(P_FREE), 0.0)

def fused_boxes(lidar, stereo, n, d):
    # Sensori indipendenti con prior 0.5: Bayes diventa una somma di log-odds. Scatole dai soli punti LiDAR
    L_lidar, points, i_r, i_theta = lidar_evidence(lidar, n, d)
    labeled, n_clusters = labeling(L_lidar + stereo_evidence(stereo, n, d) > L_MIN)
    return find_boxes(points, label_points(labeled, i_r, i_theta), n_clusters, n, d)
