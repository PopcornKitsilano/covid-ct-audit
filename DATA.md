# Data card

None of the images are in this repository. Download each dataset from its own source and respect its licence.

| Role | Dataset | Source | Licence | Notes |
|---|---|---|---|---|
| Training and in-domain test | Preprocessed CT scans for COVID-19 (nCT / pCT / NiCT), original-slice copy | Kaggle `azaemon/preprocessed-ct-scans-for-covid19` | check the Kaggle page before redistributing | 19,685 2D slices (nCT 9,979; pCT 4,001; NiCT 5,705). No patient IDs in the file names. The counts match the iCTCF resource (Ning et al., 2020: 61 COVID-19 and 43 control patients), which we take to be the source. |
| External A | SARS-CoV-2 CT-scan dataset | Kaggle `plameneduardo/sarscov2-ctscan-dataset` | CC-BY-NC-SA-4.0 | Sao Paulo. 2,481 slices, binary COVID vs non-COVID (other lung disease). No patient IDs. Images often cropped. |
| External B and group-recovery check | COVID-CT-MD | figshare collection 5129081 | CC0 | Tehran. 305 patients (169 COVID-19, 76 normal, 60 community-acquired pneumonia). Full DICOM volumes with patient IDs. Slice-level lesion labels are used for the B2 (COVID vs normal) and B3 (COVID vs CAP) sets. |

COVIDx-CT, CNCB and other iCTCF derivatives are not used as external data: they come from the same source as the training data, so they would not be independent.

Research use only. This is not a medical device and makes no clinical claim.
