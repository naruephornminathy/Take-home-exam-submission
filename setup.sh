#!/bin/bash
# setup.sh
# Downloads and extracts the 38-Cloud dataset used by this project.
#
# Usage:
#   bash setup.sh [destination_dir]
#
# destination_dir defaults to ./38-cloud

set -e

DEST_DIR="${1:-./38-cloud}"
ZIP_PATH="${DEST_DIR}.zip"

echo "==> Downloading 38-Cloud dataset (~12.4 GB)..."
wget --no-check-certificate -c -O "$ZIP_PATH" \
  "https://vault.sfu.ca/index.php/s/pymNqYF09JkM8Bp/download/38-Cloud_cloud_detection_dataset.zip"

echo "==> Extracting outer zip..."
mkdir -p "$DEST_DIR"
python3 -c "
import zipfile
with zipfile.ZipFile('$ZIP_PATH', 'r') as zf:
    zf.extractall('$DEST_DIR')
"

echo "==> Extracting inner .rar archives (training/test patches)..."
if ! command -v unrar &> /dev/null; then
    echo "unrar not found -- installing (requires apt, e.g. on Colab/Debian/Ubuntu)..."
    apt-get install -y unrar
fi
unrar x -o+ "$DEST_DIR/38-Cloud_training.rar" "$DEST_DIR/"
unrar x -o+ "$DEST_DIR/38-Cloud_test.rar" "$DEST_DIR/"

echo "==> Done. Dataset extracted to: $DEST_DIR"
echo "    Expected subfolders: train_red, train_green, train_blue, train_nir, train_gt,"
echo "    test_red, test_green, test_blue, test_nir, Entire_scene_gts, plus CSV files."
