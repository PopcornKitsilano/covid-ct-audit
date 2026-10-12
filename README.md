# covid-ct-audit

**Beyond 99%: Auditing Leakage, Shortcuts and Calibration in COVID-19 CT Classification**
Sayantan Bhattacharya · preprint, not peer-reviewed · [paper (PDF)](paper/beyond99percenttest1.pdf)

<!-- After the Zenodo upload, add the DOI here, for example: Preprint DOI: https://doi.org/10.5281/zenodo.XXXXXXX -->

A public COVID-19 CT slice classifier reports 99.76% test accuracy using a random split of individual slices. Slices from one scan are near-duplicates, so that split can leak. This project re-evaluates a MobileNetV3-Large classifier (3 classes: nCT / pCT / NiCT, 19,685 slices) with progressively more honest protocols, and packages the audit as one script.

## Headline results (3 seeds, details in the paper)

| Question | Result |
|---|---|
| Accuracy, slice split vs recovered-group splits | 99.42% vs 97.18% (feature groups) and 95.39% (file-order groups); error rate 4.9x and 8.0x higher |
| Scan metadata alone (15 scalar features), balanced accuracy | 0.996 under a slice split, 0.630 under a group split |
| Model that never sees the lungs / sees only the lungs | 0.930 / 0.929 balanced accuracy |
| External AUROC (nCT vs pCT, Sao Paulo; Tehran COVID vs normal) | 0.83; 0.86 |
| COVID vs community-acquired pneumonia (Tehran) | AUROC 0.43 to 0.49: every model scores pneumonia as more COVID-like |
| Calibration (binary ECE), in-domain vs external | 0.02 vs 0.19 to 0.53; temperature scaling gains at most 0.034 |
| Recovered groups vs true patient IDs (Tehran) | ARI at most 0.29; they keep 37% of the nearest-neighbour leakage advantage |

![Protocol comparison](paper/figures/fig04_protocols.png)

## What the code does
1. Recovers pseudo-patient groups (no IDs exist in the training data) and compares slice-level vs group-level splits.
2. Validates the group recovery against real patient IDs on COVID-CT-MD.
3. Tests for shortcuts: acquisition-metadata classifier, lung-masked ablations, test-time ablation, Grad-CAM lung enrichment.
4. Evaluates on two external sites (Sao Paulo, Tehran) and reports calibration (ECE, temperature scaling).
5. Reports uncertainty: cluster bootstrap, effective sample size, seed spread.

## Reproduce
```bash
pip install -r requirements.txt            # install the CUDA build of PyTorch first
bash scripts/get_data.sh                   # or scripts\get_data.ps1 on Windows (needs a Kaggle API token in the environment)
python covid_ct_audit.py --smoke --ext-b data/extB_raw/COVID-CT-MD   # quick check of the pipeline
python covid_ct_audit.py --ext-b data/extB_raw/COVID-CT-MD   # full run, about 45 minutes on an RTX 3060 Ti
```
`covid_ct_audit.ipynb` is the same pipeline as a notebook; its last cell (`scripts/cell_14_export.py`) exports every output. `results/all_outputs.txt` is the raw log of the full run that the paper's numbers come from. Datasets and sources are in [docs/DATA.md](docs/DATA.md).

## Limitations (see paper, Section 10)
Groups on the main dataset are recovered, not true patient IDs. One backbone and three seeds. Lung masks are heuristic. External sets differ in label definition. This is research code and makes no clinical claim.

## Cite
See `CITATION.cff`. Licence: MIT (code only; each dataset keeps its own licence).
