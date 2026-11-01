#!/usr/bin/env bash
# Scarica le immagini della coppia stereo KITTI (camera 2 e 3) ed estrae solo training/ in $KITTI_ROOT.
#
# Uso:  bash scripts/download_images.sh
# Ri-eseguibile: curl riprende i download interrotti, gli zip gia' completi vengono saltati.

set -euo pipefail

BASE="https://s3.eu-central-1.amazonaws.com/avg-kitti"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$(cd -P "${KITTI_ROOT:-$ROOT/data/kitti}" && pwd)"    # -P: segue il link simbolico fino al dataset vero
DL="$(dirname "$DEST")/_archives"

mkdir -p "$DL"

FILES=(
  "data_object_image_2.zip"   # 12.6 GB - camera sinistra (quella delle label)
  "data_object_image_3.zip"   # 12.0 GB - camera destra
)

gb() { awk -v n="$1" 'BEGIN{printf "%.1f GB", n / 1e9}'; }

for f in "${FILES[@]}"; do
  dest="$DL/$f"
  expected=$(curl -sIL "$BASE/$f" | awk 'tolower($1)=="content-length:"{n=$2} END{print n+0}')
  if [[ -f "$dest" && $(wc -c < "$dest" | tr -d ' ') -eq $expected ]]; then
    echo "[skip] $f  (gia' completo, $(gb "$expected"))"
  else
    echo "[dl]   $f  ->  $(gb "$expected")"
    curl -L -C - --retry 5 --retry-delay 5 -sS -o "$dest" "$BASE/$f"
  fi
done

echo "[unzip] solo training/ in $DEST"
for f in "${FILES[@]}"; do
  echo "  - $f"
  unzip -n -q "$DL/$f" 'training/*' -d "$DEST"
done

echo
echo "=== verifica ==="
for cam in image_2 image_3; do
  n=$(find "$DEST/training/$cam" -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
  echo "$cam: $n   (attesi 7481)"
  [[ "$n" -eq 7481 ]] || echo "ATTENZIONE: $cam incompleto"
done
