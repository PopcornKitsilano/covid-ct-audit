#!/usr/bin/env bash
# Linux / macOS / WSL. Run from the repo root.  Needs: pip install kaggle   and   export KAGGLE_API_TOKEN=...
set -euo pipefail
mkdir -p data/main data/extA data/extB_raw
kaggle datasets download -d azaemon/preprocessed-ct-scans-for-covid19 -p data/main --unzip
kaggle datasets download -d plameneduardo/sarscov2-ctscan-dataset   -p data/extA --unzip

# COVID-CT-MD (figshare, CC0, ~10.5 GiB). curl -C - resumes if interrupted: just rerun.
curl -L -C - -o data/COVID-CT-MD.zip https://ndownloader.figshare.com/files/26069987
echo "expected size 11273767727 bytes, md5 7cd2a4fdc7b1348c093b9a384d4b2240"
stat -c %s data/COVID-CT-MD.zip 2>/dev/null || stat -f %z data/COVID-CT-MD.zip
(md5sum data/COVID-CT-MD.zip 2>/dev/null || md5 data/COVID-CT-MD.zip)
unzip -q data/COVID-CT-MD.zip -d data/extB_raw
