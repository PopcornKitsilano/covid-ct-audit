# Windows PowerShell. Run from the repo root.  Needs: pip install kaggle   and   $env:KAGGLE_API_TOKEN set.
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force data\main, data\extA, data\extB_raw | Out-Null

kaggle datasets download -d azaemon/preprocessed-ct-scans-for-covid19 -p data/main --unzip
kaggle datasets download -d plameneduardo/sarscov2-ctscan-dataset   -p data/extA --unzip

# COVID-CT-MD (figshare, CC0, ~10.5 GiB). curl resumes if interrupted: just rerun this line.
curl.exe -L -C - -o data\COVID-CT-MD.zip https://ndownloader.figshare.com/files/26069987

# integrity check (expected size 11273767727 bytes, MD5 7cd2a4fdc7b1348c093b9a384d4b2240)
(Get-Item data\COVID-CT-MD.zip).Length
certutil -hashfile data\COVID-CT-MD.zip MD5

tar -xf data\COVID-CT-MD.zip -C data\extB_raw
