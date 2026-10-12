"""
COVID CT audit - runs from a terminal, no Jupyter needed.

  python covid_ct_audit.py --smoke                    # 5-10 minute end-to-end check, run this FIRST
  python covid_ct_audit.py                            # full run
  python covid_ct_audit.py --main D:/data/main --ext-a D:/data/extA --ext-b D:/data/extB_raw/COVID-CT-MD --work D:/covid_work

Figures are saved as PNG files in <work>/figs_<smoke|full>/, tables as CSV and a markdown summary in <work>/.
Cells are separated by '# %%' so the same file also opens as a notebook in VS Code / Spyder.
"""
import matplotlib
matplotlib.use("Agg")      # no windows; figures are written to disk


# %% [markdown]
# # How much of "99.76%" is real?  Leakage, shortcuts and external validity in slice-level COVID-19 CT classification
# 
# Starting point: [`chrispathway/covid-ct-classifier`](https://github.com/chrispathway/covid-ct-classifier), a MobileNetV3-Large fine-tuned on the Kaggle *CT Scans for COVID-19 Classification* data (nCT / pCT / NiCT), reporting 99.76 % test accuracy with a **per-slice random split**.
# 
# This notebook audits that number. Every section produces one result you can put in a report:
# 
# | # | Question | Section |
# |---|----------|---------|
# | 1 | Are the test slices near-duplicates of training slices? | 6 |
# | 2 | How much accuracy is lost when slices of the same (recovered) patient never straddle train/test? | 7-8 |
# | 3 | Can a model score well **without looking at lungs** (image size, FOV, intensity, background)? | 9 |
# | 4 | Does the model look at the lungs? (Grad-CAM, lung/no-lung ablations) | 9 |
# | 5 | Does it generalise to a different hospital / country? | 10 |
# | 6 | Do simple robustness changes (intensity/FOV augmentation, lung-only input) help? | 8-10 |
# | 7 | Are the probabilities calibrated? | 11 |
# | 8 | (optional) Does the *patient-group recovery* method itself work, checked against real patient IDs? | 12 |
# 
# **Not a medical device. Research / education only.**
# 
# ## Before you run
# 
# 1. **Kaggle:** `+ Add Input` -> attach
#    * `azaemon/preprocessed-ct-scans-for-covid19`  (main data; Wuhan, iCTCF; CC BY 4.0)
#    * `plameneduardo/sarscov2-ctscan-dataset`  (external A; Sao Paulo, HSPM; **no patient IDs**; "non-COVID" = *other lung diseases*, not healthy)
# 2. Settings: **Accelerator = GPU**, **Internet = On** (needed once to download the ImageNet weights).
# 3. Set `SMOKE = True` in the next cell and run everything once (about 5 min). If it finishes, set `SMOKE = False` and run again.
# 4. Optional external B: **COVID-CT-MD** (Babak Imaging Center, Tehran; 169 COVID / 76 normal / 60 CAP; has **patient-level IDs**; DICOM). It is on figshare (DOI 10.6084/m9.figshare.c.5129081.v1), so you must download it yourself, arrange it as `<root>/<class>/<patient>/<files>`, upload it as a private Kaggle dataset (or point `EXT_B_ROOT` at it locally), and check its licence. It is the only way to *validate* the patient-group recovery against ground truth (Section 12).
# 
# **Datasets deliberately not used as "external":** COVIDx CT and any CNCB / iCTCF / CC-CCII derivative share their source (Chinese hospitals) with the training data, so they would silently contaminate an external test. MosMedData (Moscow, NIfTI, CC BY-NC-ND) is a good additional external set but needs a volume-to-slice conversion and its no-derivatives licence needs reading first.
# 
# **Honest limitations built into the design** (keep them in your write-up): the "patients" here are *recovered clusters*, not true patients; the lung masks are classical image processing, not a trained segmenter; external A has no patient IDs, so its confidence intervals are optimistic.

# %%
# =============================== CONFIG ===============================
import os, sys, json, math, time, re, random, warnings, itertools
from pathlib import Path
warnings.filterwarnings("ignore")

import argparse
_ap = argparse.ArgumentParser(description="COVID CT audit")
_ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch: checks the whole pipeline")
_ap.add_argument("--main", default="data/main", help="folder containing the nCT / pCT / NiCT data (or its parent)")
_ap.add_argument("--ext-a", default="data/extA", help="Sao Paulo SARS-CoV-2 CT-scan dataset (COVID / non-COVID folders)")
_ap.add_argument("--ext-b", default="data/extB_raw/COVID-CT-MD", help="COVID-CT-MD folder (class/patient/*.dcm)")
_ap.add_argument("--work", default="work", help="output folder (caches, runs, figures, tables)")
_ap.add_argument("--b-max", type=int, default=40, help="max slices per patient from external B")
ARGS = _ap.parse_args()

SMOKE = ARGS.smoke
SMOKE_PER_CLASS = 150

SEEDS      = [0, 1, 2]   # seeds for the headline comparisons
SEEDS_ABL  = [0, 1, 2]   # seeds for the shortcut ablations
EPOCHS, BATCH, LR, WD, IMG = 8, 32, 1e-4, 1e-4, 224
GROUP_K       = 300                          # pseudo-patient clusters used for the group split
GROUP_K_LIST  = [50, 100, 300, 1000, 3000]   # K values for the cheap leakage-vs-K curve
GROUP_FEATURES = "deep+thumb"                # "deep" | "deep+thumb"
N_BOOT = 1000
ALLOW_RANDOM_INIT = False   # True ONLY for offline smoke tests (results are meaningless then)

# Data locations. None = auto-discover under /kaggle/input (or the current folder when running locally).
MAIN_ROOT  = ARGS.main
EXT_A_ROOT = ARGS.ext_a if Path(ARGS.ext_a).exists() else None
EXT_B_ROOT = ARGS.ext_b if Path(ARGS.ext_b).exists() else None
B_WINDOW = (-600, 1500)   # (centre, width) in HU when converting DICOM -> 8 bit
B_CENTER_FRAC = 0.6       # keep the central fraction of each patient's slices (lungs are there)
B_MAX_PER_PATIENT = 3 if ARGS.smoke else ARGS.b_max

ON_KAGGLE = False
TAG  = "smoke" if SMOKE else "full"
WORK = Path(ARGS.work)
RUNS = WORK / f"runs_{TAG}"; FIGS = WORK / f"figs_{TAG}"
for d in (WORK, RUNS, FIGS): d.mkdir(parents=True, exist_ok=True)

if SMOKE:
    EPOCHS, SEEDS, SEEDS_ABL = 1, [0], [0]
    GROUP_K, GROUP_K_LIST, N_BOOT = 20, [10, 20, 50], 200

CLASSES = ["nCT", "pCT", "NiCT"]          # label ids 0, 1, 2
CLASS_NAMES = {"nCT": "Normal", "pCT": "COVID-19 positive", "NiCT": "Non-informative"}
print(f"SMOKE={SMOKE}  ON_KAGGLE={ON_KAGGLE}  outputs -> {WORK.resolve()}")

# %%
# =============================== IMPORTS + PLOT STYLE (no torch yet) ===============================
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage as ndi, stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score,
                             adjusted_rand_score, roc_curve, average_precision_score)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

# Chart colours: validated categorical slots 1-3 (blue, orange, aqua) + neutral inks. Text never wears a series colour.
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "font.size": 10, "figure.dpi": 110,
    "lines.linewidth": 2.0, "lines.markersize": 6,
})
def savefig(name):
    plt.savefig(FIGS / f"{name}.png", bbox_inches="tight"); plt.close("all")

# %%
# =============================== TORCH ===============================
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
print("torch", torch.__version__, "| device:", DEVICE, "|", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def amp_ctx():
    return torch.autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu", dtype=torch.float16, enabled=(DEVICE.type == "cuda"))

def get_backbone():
    """MobileNetV3-Large with ImageNet weights (needs Internet=On the first time)."""
    try:
        return models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    except Exception as e:
        if not ALLOW_RANDOM_INIT:
            raise RuntimeError("Could not download ImageNet weights. Turn Internet ON in the Kaggle settings "
                               "(or attach the weights as a dataset). For an offline smoke test only, set ALLOW_RANDOM_INIT=True.") from e
        print("WARNING: using RANDOM weights (smoke test only)"); return models.mobilenet_v3_large(weights=None)

# %%
# =============================== 1. FIND THE DATA ===============================
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
def list_images(d): return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXT)

def _walk(base, max_depth=6):
    base = Path(base)
    for dp, dn, _ in os.walk(base):
        if len(Path(dp).relative_to(base).parts) >= max_depth: dn[:] = []
        yield Path(dp), list(dn)

def find_main_dirs(base):
    cands = []
    for dp, dn in _walk(base):
        low = {d.lower(): d for d in dn}
        if {"nct", "pct", "nict"} <= set(low):
            cands.append({c: dp / low[c.lower()] for c in CLASSES})
    return cands

def find_ext_a_dirs(base):
    for dp, dn in _walk(base):
        low = {d.lower().replace("_", "-"): d for d in dn}
        pos = [k for k in low if k in ("covid", "covid-19", "covid19")]
        neg = [k for k in low if k in ("non-covid", "noncovid", "non-covid19", "non-covid-19")]
        if pos and neg: return {1: dp / low[pos[0]], 0: dp / low[neg[0]]}
    return None

BASE = Path("/kaggle/input") if ON_KAGGLE else Path(".")
cands = find_main_dirs(MAIN_ROOT or BASE)
if not cands:
    raise FileNotFoundError("Could not find folders nCT / pCT / NiCT. Attach the Kaggle dataset or set MAIN_ROOT.")
# Prefer the ORIGINAL slices over the 'preprocessed' copy (the original repo uses the originals)
cands.sort(key=lambda d: sum(any(t in str(p).lower() for t in ("prepro", "process", "resized", "norm")) for p in d.values()))
for i, c in enumerate(cands): print(i, {k: str(v) for k, v in c.items()})
MAIN_DIRS = cands[0]
print("\nUSING:", {k: str(v) for k, v in MAIN_DIRS.items()}, "\n(if this is the wrong copy, set MAIN_ROOT to the right folder and re-run)")

# %%
# =============================== 2. INDEX + CACHE (224x224 uint8 grayscale) ===============================
rows = [(str(p), c, ci) for ci, c in enumerate(CLASSES) for p in list_images(MAIN_DIRS[c])]
df = pd.DataFrame(rows, columns=["path", "cls", "y"]); df["stem"] = df.path.map(lambda s: Path(s).stem)
if SMOKE:
    df = pd.concat([df[df.y == k].sample(min((df.y == k).sum(), SMOKE_PER_CLASS), random_state=0) for k in range(3)]).reset_index(drop=True)
print(df.cls.value_counts().to_string()); assert len(df) > 0

def load_gray(path, size=IMG):
    with Image.open(path) as im:
        w, h = im.size
        if im.mode.startswith("I") or im.mode == "F":              # 16/32-bit images
            a = np.asarray(im, dtype=np.float32); a = (a - a.min()) / max(a.max() - a.min(), 1e-6) * 255
            g = Image.fromarray(a.astype(np.uint8))
        else:
            g = im.convert("L")
        arr = np.asarray(g.resize((size, size), Image.Resampling.BICUBIC), dtype=np.uint8)
    return arr, w, h, os.path.getsize(path)

def load_many(paths, size=IMG, desc="loading"):
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as ex:
        res = list(tqdm(ex.map(lambda p: load_gray(p, size), paths), total=len(paths), desc=desc))
    return np.stack([r[0] for r in res]), np.array([r[1] for r in res]), np.array([r[2] for r in res]), np.array([r[3] for r in res])

def scalar_feats(X, w, h, nbytes):
    """Cheap acquisition descriptors that contain NO lung information on purpose."""
    f = X.astype(np.float32) / 255; s = X.shape[1]; k = s // 8; c0 = s // 2 - s // 8
    corners = np.concatenate([f[:, :k, :k].reshape(len(f), -1), f[:, :k, -k:].reshape(len(f), -1),
                              f[:, -k:, :k].reshape(len(f), -1), f[:, -k:, -k:].reshape(len(f), -1)], 1)
    return pd.DataFrame(dict(
        w=w, h=h, aspect=w / h, nbytes=nbytes, bytes_per_px=nbytes / (w * h),
        mean=f.mean((1, 2)), std=f.std((1, 2)), frac_black=(X <= 8).mean((1, 2)), frac_white=(X >= 247).mean((1, 2)),
        corner_mean=corners.mean(1), center_mean=f[:, c0:c0 + s // 4, c0:c0 + s // 4].mean((1, 2)),
        top_row=f[:, :2].mean((1, 2)), bottom_row=f[:, -2:].mean((1, 2)),
        left_col=f[:, :, :2].mean((1, 2)), right_col=f[:, :, -2:].mean((1, 2))))

fx, fm = WORK / f"main_X_{TAG}.npy", WORK / f"main_meta_{TAG}.csv"
if fx.exists() and fm.exists() and list(pd.read_csv(fm).path) == list(df.path):
    X = np.load(fx); meta = pd.read_csv(fm).drop(columns=["path"]); print("loaded cache", X.shape)
else:
    X, w_, h_, nb_ = load_many(df.path.tolist(), desc="main images")
    meta = scalar_feats(X, w_, h_, nb_)
    np.save(fx, X); pd.concat([df[["path"]], meta], axis=1).to_csv(fm, index=False)
y = df.y.values.astype(int); N = len(y)
print("X:", X.shape, X.dtype, f"{X.nbytes / 1e9:.2f} GB")

# %%
# =============================== 3. AUDIT: what could a model cheat with? ===============================
print("Image size (w x h) per class - top 5:")
for c in CLASSES:
    m = meta[df.cls == c]; print(f"  {c:5s}", m.assign(s=m.w.astype(str) + "x" + m.h.astype(str)).s.value_counts().head(5).to_dict())
print("\nFile-name examples:"); [print(f"  {c:5s}", df[df.cls == c].stem.head(4).tolist()) for c in CLASSES]

def parse_fname_groups(stems):
    """If names look like <scan-id><slice-number>, the prefix is a candidate scan/patient ID."""
    m = [re.match(r"^(.*?)(\d+)$", s) for s in stems]
    if not all(m): return None
    prefix = np.array([x.group(1) for x in m]); n = len(set(prefix))
    return prefix if 20 <= n <= len(stems) // 3 else None
fg = parse_fname_groups(df.stem.tolist())
if fg is not None:
    df["fname_group"] = fg; print(f"\nFile names look like <scan><slice>: {df.fname_group.nunique()} candidate scan IDs. VERIFY by eye before trusting them.")
else:
    df["fname_group"] = None; print("\nNo usable scan/patient ID in the file names -> groups must be recovered from image content (Section 6).")

# How different are the classes in 'non-lung' properties?
feats = [("frac_black", "fraction of black pixels"), ("corner_mean", "corner brightness"), ("mean", "mean intensity"), ("aspect", "width / height")]
fig, axs = plt.subplots(1, 4, figsize=(15, 3.1))
for ax, (f, t) in zip(axs, feats):
    lo, hi = np.percentile(meta[f], [0.5, 99.5]); bins = np.linspace(lo, hi, 40)
    for c, col in zip(range(3), (C1, C2, C3)):
        ax.hist(meta[f][y == c].clip(lo, hi), bins=bins, histtype="step", color=col, lw=1.8, density=True, label=CLASSES[c])
    ax.set_title(t); ax.set_yticks([])
axs[0].legend(); fig.suptitle("Properties that say nothing about lungs but differ by class = shortcut candidates", x=0.01, ha="left", color=INK2, fontsize=10)
savefig("01_nonlung_feature_hist")

rng = np.random.default_rng(0)
fig, axs = plt.subplots(3, 6, figsize=(13, 6.6))
for r, c in enumerate(range(3)):
    for j, i in enumerate(rng.choice(np.where(y == c)[0], 6, replace=False)):
        axs[r, j].imshow(X[i], cmap="gray", vmin=0, vmax=255); axs[r, j].axis("off")
    axs[r, 0].set_title(CLASSES[c], loc="left", fontsize=10)
savefig("02_sample_grid")

# %%
# =============================== 4. CLASSICAL LUNG SEGMENTATION (heuristic!) ===============================
from skimage import measure, morphology, filters

def lung_mask(a, min_frac=0.01, max_thresh=0.65):
    """Heuristic lung mask for a 2-D uint8 slice. Returns a (dilated) boolean mask, empty if no lungs are found."""
    H, W = a.shape; z = np.zeros((H, W), bool)
    g = filters.gaussian(a.astype(np.float32), sigma=1.0, preserve_range=True)
    # 1) pure-black surroundings that touch the border (outside the circular field of view) -> ignore for thresholding
    lab = measure.label(g < 0.08 * 255, connectivity=1)
    border_ids = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])); border_ids = border_ids[border_ids > 0]
    bg = np.isin(lab, border_ids)
    vals = g[~bg]
    if vals.size < 200 or vals.max() - vals.min() < 10: return z
    try: t = filters.threshold_otsu(vals)
    except Exception: return z
    if t > max_thresh * 255: return z      # no genuinely dark (air-like) class -> e.g. body vs white background, no lungs in this slice
    # 2) dark (air-like) regions that do NOT touch the border = lung candidates
    dark = ndi.binary_opening((g < t) & ~bg, structure=morphology.disk(2))   # drops thin dark rims (skin edge) and tiny airway spots
    lab2 = measure.label(dark, connectivity=1)
    bids = np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]]))
    keep = (lab2 > 0) & ~np.isin(lab2, bids)
    lab3 = measure.label(keep, connectivity=1)
    if lab3.max() == 0: return z
    # body = largest THICK bright structure (thin table frames are removed by the opening); lungs must lie inside its convex hull.
    # This drops the dark cushion region of the CT table, which is enclosed by bright lines and otherwise looks like a lung.
    bl = measure.label(ndi.binary_opening((g >= t) & ~bg, structure=morphology.disk(4)), connectivity=1)
    if bl.max() > 0:
        hull = morphology.convex_hull_image(bl == (np.argmax(np.bincount(bl.ravel())[1:]) + 1))
        inside = np.array([hull[lab3 == k].mean() >= 0.8 for k in range(1, lab3.max() + 1)])
    else:
        inside = np.ones(lab3.max(), bool)
    areas = np.bincount(lab3.ravel())[1:] * inside
    if areas.max() == 0: return z
    ok = np.where((areas >= min_frac * H * W) & (areas >= 0.2 * areas.max()))[0] + 1
    if ok.size == 0: return z
    ok = ok[np.argsort(-areas[ok - 1])][:3]
    m = np.isin(lab3, ok)
    m = ndi.binary_closing(m, structure=morphology.disk(5))       # re-include small bright vessels
    m = ndi.binary_fill_holes(m)                                    # re-include consolidations / ground-glass inside the lung
    # lung FIELD: convex hull of every lung component. Dense consolidations at the lung edge are not 'dark', so the plain threshold
    # leaves them outside the mask; the hull puts them back, so 'lungs removed' really removes the pathology.
    ml = measure.label(m, connectivity=1); mh = np.zeros_like(m)
    for k in range(1, ml.max() + 1):
        comp = ml == k
        mh |= morphology.convex_hull_image(comp) if comp.sum() >= 30 else comp
    m = mh
    return ndi.binary_dilation(m, structure=morphology.disk(3))     # small safety margin

def compute_masks(Xs):
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as ex:
        ms = list(tqdm(ex.map(lung_mask, Xs), total=len(Xs), desc="lung masks"))
    return np.stack(ms).astype(np.uint8)

MASK_VERSION = "v3"   # v3 = lung field (convex hull per lung); bump this to invalidate cached masks and mask-based runs
fmk = WORK / f"main_M_{TAG}_{MASK_VERSION}.npy"
if fmk.exists() and np.load(fmk, mmap_mode="r").shape[0] == N: M = np.load(fmk)
else: M = compute_masks(X); np.save(fmk, M)
area = M.reshape(N, -1).mean(1)
qc = pd.DataFrame({"cls": df.cls, "lung_area": area, "empty": area < 0.01}).groupby("cls").agg(median_area=("lung_area", "median"), frac_empty=("empty", "mean")).loc[CLASSES]
print("Mask QC (expect: nCT/pCT mostly non-empty, NiCT mostly empty or small):"); print(qc.round(3).to_string())

fig, axs = plt.subplots(3, 8, figsize=(15, 6.2)); rng = np.random.default_rng(1)
for r in range(3):
    for j, i in enumerate(rng.choice(np.where(y == r)[0], 8, replace=False)):
        axs[r, j].imshow(X[i], cmap="gray", vmin=0, vmax=255)
        if M[i].any(): axs[r, j].contour(M[i], levels=[0.5], colors=C2, linewidths=1.0)
        axs[r, j].axis("off")
    axs[r, 0].set_title(CLASSES[r], loc="left", fontsize=10)
fig.suptitle("QC: LOOK AT THIS before trusting any lung-mask result. Orange = heuristic lung region", x=0.01, ha="left", color=INK2, fontsize=10)
savefig("03_lung_mask_qc")

# %%
# =============================== 5. EXTERNAL DATASETS ===============================
EXT = {}   # name -> dict(X, y, M, groups, desc)

def add_ext(name, Xe, ye, groups=None, desc=""):
    EXT[name] = dict(X=Xe, y=np.asarray(ye).astype(int), M=compute_masks(Xe), groups=groups, desc=desc)
    print(f"[{name}] {len(Xe)} slices | positives={int(np.sum(ye))} | {desc}")

# ---- External A: SARS-CoV-2 CT-scan dataset (Sao Paulo, HSPM). Binary: COVID vs non-COVID (other lung diseases!)
dA = find_ext_a_dirs(EXT_A_ROOT or BASE)
if dA is None:
    print("External A not found (attach plameneduardo/sarscov2-ctscan-dataset or set EXT_A_ROOT). Skipping.")
else:
    pa = [(str(p), 1) for p in list_images(dA[1])] + [(str(p), 0) for p in list_images(dA[0])]
    if SMOKE:
        pa = [t for k in (0, 1) for t in [u for u in pa if u[1] == k][:SMOKE_PER_CLASS]]
    Xa, *_ = load_many([p for p, _ in pa], desc="external A")
    add_ext("A_saopaulo", Xa, [l for _, l in pa], None, "Sao Paulo; non-COVID = other lung disease; images are often cropped; NO patient IDs")

# ---- External B (optional): COVID-CT-MD arranged as <root>/<class>/<patient>/<files>
def _nat(s): return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]
def _read_dicom_u8(path, center, width, size=IMG):
    import pydicom
    ds = pydicom.dcmread(str(path))
    try: a = ds.pixel_array.astype(np.float32)
    except RuntimeError as e:
        raise RuntimeError("Cannot decode this DICOM (JPEG-Lossless?). Run `pip install pylibjpeg pylibjpeg-libjpeg`, then "
                           "Runtime -> Restart session (pydicom caches decoder availability) and re-run from CONFIG.") from e
    a = a * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
    lo = center - width / 2; a = np.clip((a - lo) / width, 0, 1) * 255
    return np.asarray(Image.fromarray(a.astype(np.uint8)).resize((size, size), Image.Resampling.BICUBIC))

def load_ext_b(root):
    root = Path(root); Xs, ys, pid = [], [], []
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        n = cdir.name.lower()
        lab = 1 if n.startswith("covid") else (0 if n.startswith(("normal", "healthy", "non")) else None)
        if lab is None: print("  skipping class folder (not COVID/normal):", cdir.name); continue
        for pdir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            files = sorted([f for f in pdir.rglob("*") if f.suffix.lower() in IMG_EXT | {".dcm", ".dicom"}], key=lambda f: _nat(f.name))
            if not files: continue
            a, b = int(len(files) * (1 - B_CENTER_FRAC) / 2), int(len(files) * (1 + B_CENTER_FRAC) / 2)
            files = files[a:max(b, a + 1)]
            if len(files) > B_MAX_PER_PATIENT: files = [files[i] for i in np.linspace(0, len(files) - 1, B_MAX_PER_PATIENT).astype(int)]
            for f in files:
                try: img = _read_dicom_u8(f, *B_WINDOW) if f.suffix.lower() in (".dcm", ".dicom") else load_gray(f)[0]
                except Exception as e: print("  unreadable:", f.name, e); continue
                Xs.append(img); ys.append(lab); pid.append(f"{cdir.name}/{pdir.name}")
    return np.stack(Xs), np.array(ys), np.array(pid)

if EXT_B_ROOT:
    Xb, yb, pb = load_ext_b(EXT_B_ROOT)
    add_ext("B_covidctmd", Xb, yb, pb, f"Tehran; {len(set(pb))} patients; PRELIMINARY patient-level labels on central slices; true patient IDs available")
else:
    print("External B not configured (optional).")
print("\nExternal sets loaded:", list(EXT))

# %%
# =============================== 5b. EXTERNAL B (CORRECTED): real slice-level lesion labels ===============================
# COVID-CT-MD ships Slice-level-labels.npy (79 patients x 200 slices; 1 = radiologist marked a lesion on that slice)
# and Index.csv (Label Index -> folder).  Rows 0..53 are COVID, rows 54..78 are CAP.  The label of a slice is looked up
# by ONE of three conventions; "auto" tests which one the data support and prints the evidence.
#   filenum1: slice IM0042.dcm  -> label[41]      filenum0: IM0042.dcm -> label[42]      position: k-th slice by InstanceNumber -> label[k]
# Sets created (old B_covidctmd stays, as the preliminary patient-level-label version; it is also used for group-recovery in Sec 12):
#   B2_tehran_lesion : lesion slices of COVID patients (positive)  vs  central slices of NORMAL patients (negative)
#   B3_tehran_vs_cap : the same positives  vs  lesion slices of CAP patients (hard negatives: another pneumonia)
B_LABEL_CONVENTION = "auto"        # "auto" | "filenum1" | "filenum0" | "position"
B2_MAX_PER_PATIENT = 40

def _fnum(p):
    m = re.search(r"(\d+)", Path(p).stem); return int(m.group(1)) if m else -1
def _inst_no(p):
    import pydicom
    try: return int(pydicom.dcmread(str(p), stop_before_pixels=True, specific_tags=["InstanceNumber"]).InstanceNumber)
    except Exception: return -1

def load_ext_b_lesion(root):
    root = Path(root); lab_f, idx_f = root / "Slice-level-labels.npy", root / "Index.csv"
    if not (lab_f.exists() and idx_f.exists()): print("  Slice-level-labels.npy / Index.csv not found -> skipping corrected B"); return None
    lab = np.load(lab_f); idx = pd.read_csv(idx_f).sort_values("Label Index").reset_index(drop=True)
    dirs = {"COVID": root / "COVID-19 Cases", "CAP": root / "Cap Cases"}
    # Index.csv's "Label Index" is NOT a safe row number (in COVID-CT-MD it skips a value: COVID 0..53, CAP 55..79, but the label
    # array has only 79 rows).  When the index has exactly as many entries as the array has rows, the only consistent mapping is
    # row = position in Index order.  We print where the two disagree so you can see it.
    by_rank = len(idx) == lab.shape[0]
    mism = [(r["Folder/ID"], int(r["Label Index"]), k) for k, r in idx.iterrows() if int(r["Label Index"]) != k]
    print(f"  Index.csv: {len(idx)} entries, label array: {lab.shape[0]} rows -> using {'position in Index order' if by_rank else 'Label Index'} as the row; "
          f"{len(mism)} entries where Label Index != position, first few (folder, Label Index, position): {mism[:4]}")
    pats = []
    for k, r in idx.iterrows():
        kind = "COVID" if str(r["Diagnosis"]).upper().startswith("COVID") else "CAP"
        row = k if by_rank else int(r["Label Index"])
        if row >= lab.shape[0]: print("  label row out of range, skipping", r["Folder/ID"]); continue
        d = dirs[kind] / str(r["Folder/ID"])
        files = sorted(d.rglob("*.dcm"), key=lambda f: _nat(f.name)) if d.exists() else []
        if files: pats.append(dict(kind=kind, pid=str(r["Folder/ID"]), row=row, files=files))
    if SMOKE: pats = [p for k in ("COVID", "CAP") for p in [q for q in pats if q["kind"] == k][:8]]
    print(f"  patients with labels and DICOMs found: COVID={sum(p['kind']=='COVID' for p in pats)}, CAP={sum(p['kind']=='CAP' for p in pats)}")

    # ---- which label convention do the data support?  (fraction of labelled-positive slice numbers that exist as files)
    f1, f0, c1, c0, kd = [], [], [], [], []
    for p in pats:
        nums = {_fnum(f) for f in p["files"]}; pos = np.nonzero(lab[p["row"]])[0]
        if len(pos) == 0: continue
        f1.append(np.mean([(k + 1) in nums for k in pos])); f0.append(np.mean([k in nums for k in pos]))
        c1.append(np.mean([(k + 1) in nums for k in range(pos.max() + 1)])); c0.append(np.mean([k in nums for k in range(pos.max() + 1)])); kd.append(p["kind"])
    ev = pd.DataFrame({"kind": kd, "filenum1": f1, "chance1": c1, "filenum0": f0, "chance0": c0}).groupby("kind").mean().round(3)
    ev.loc["ALL"] = [np.mean(f1), np.mean(c1), np.mean(f0), np.mean(c0)]; ev = ev.round(3)
    print("  label-convention evidence = fraction of labelled-positive slice numbers that exist as files (a convention is supported when filenumX ~1.0 and clearly above chanceX;"
          "\n  if COVID and CAP rows disagree, the row mapping is wrong for one of them):"); print(ev.to_string())
    conv = B_LABEL_CONVENTION
    if conv == "auto":
        if np.mean(f1) >= 0.98 and np.mean(c1) <= 0.9: conv = "filenum1"
        elif np.mean(f0) >= 0.98 and np.mean(c0) <= 0.9: conv = "filenum0"
        else: conv = "position"
    print("  -> using label convention:", conv, "(if this is 'position' the file numbers carried no information; slices are ordered by DICOM InstanceNumber)")

    def slice_labels(p):
        """-> list of (file, label or None) for one labelled patient"""
        row = lab[p["row"]]; fs = p["files"]; out = []
        if conv == "position":
            order = sorted(fs, key=lambda f: (_inst_no(f), _nat(f.name)))
            out = [(f, int(row[i]) if i < len(row) else None) for i, f in enumerate(order)]
        else:
            off = 1 if conv == "filenum1" else 0
            for f in sorted(fs, key=lambda f: _fnum(f)):
                k = _fnum(f) - off; out.append((f, int(row[k]) if 0 <= k < len(row) else None))
        return out

    def take(items):   # <= B2_MAX_PER_PATIENT evenly spaced
        if len(items) > B2_MAX_PER_PATIENT: items = [items[i] for i in np.linspace(0, len(items) - 1, B2_MAX_PER_PATIENT).astype(int)]
        return items
    def read(fs):
        out = []
        for f in fs:
            try: out.append(_read_dicom_u8(f, *B_WINDOW))
            except Exception as e: print("  unreadable:", f.name, e)
        return out

    Xp, gp, Xc, gc, n_les, n_tot = [], [], [], [], {"COVID": 0, "CAP": 0}, {"COVID": 0, "CAP": 0}
    for p in pats:
        sl = [(f, l) for f, l in slice_labels(p) if l is not None]
        les = [f for f, l in sl if l == 1]; n_les[p["kind"]] += len(les); n_tot[p["kind"]] += len(sl)
        imgs = read(take(les))
        (Xp if p["kind"] == "COVID" else Xc).extend(imgs); (gp if p["kind"] == "COVID" else gc).extend([f"{p['kind']}/{p['pid']}"] * len(imgs))
    print(f"  lesion slices: COVID {n_les['COVID']} of {n_tot['COVID']} labelled slices, CAP {n_les['CAP']} of {n_tot['CAP']} (subsampled to <= {B2_MAX_PER_PATIENT} per patient)")

    Xn, gn = [], []
    ndir = root / "Normal Cases"
    nps = sorted(p for p in ndir.iterdir() if p.is_dir()) if ndir.exists() else []
    if SMOKE: nps = nps[:8]
    for pdir in nps:
        fs = sorted(pdir.rglob("*.dcm"), key=lambda f: (_inst_no(f), _nat(f.name)))
        if not fs: continue
        a, b = int(len(fs) * (1 - B_CENTER_FRAC) / 2), int(len(fs) * (1 + B_CENTER_FRAC) / 2)
        imgs = read(take(fs[a:max(b, a + 1)])); Xn.extend(imgs); gn.extend([f"Normal/{pdir.name}"] * len(imgs))
    return dict(Xp=np.stack(Xp), gp=np.array(gp), Xc=np.stack(Xc) if Xc else None, gc=np.array(gc), Xn=np.stack(Xn) if Xn else None, gn=np.array(gn), conv=conv)

if EXT_B_ROOT:
    _b = load_ext_b_lesion(EXT_B_ROOT)
    if _b is not None and len(_b["Xp"]) and _b["Xn"] is not None:
        add_ext("B2_tehran_lesion", np.concatenate([_b["Xp"], _b["Xn"]]), np.r_[np.ones(len(_b["Xp"])), np.zeros(len(_b["Xn"]))],
                np.r_[_b["gp"], _b["gn"]], f"Tehran, lesion-labelled COVID slices vs normal patients; labels by {_b['conv']}; true patient IDs")
    if _b is not None and len(_b["Xp"]) and _b["Xc"] is not None:
        add_ext("B3_tehran_vs_cap", np.concatenate([_b["Xp"], _b["Xc"]]), np.r_[np.ones(len(_b["Xp"])), np.zeros(len(_b["Xc"]))],
                np.r_[_b["gp"], _b["gc"]], f"Tehran, lesion-labelled COVID slices vs CAP lesion slices (hard negatives); labels by {_b['conv']}; true patient IDs")
print("\nExternal sets now:", list(EXT))

# %%
# =============================== 6a. EMBEDDINGS (frozen ImageNet MobileNetV3) + PCA ===============================
def l2n(a): return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)

def embed_deep(Xs, bs=256):
    m = get_backbone().to(DEVICE).eval(); fe = nn.Sequential(m.features, m.avgpool, nn.Flatten()); out = []
    with torch.no_grad(), amp_ctx():
        for i in range(0, len(Xs), bs):
            xb = torch.from_numpy(Xs[i:i + bs]).to(DEVICE).float().div(255).unsqueeze(1).expand(-1, 3, -1, -1)
            out.append(fe((xb - MEAN) / STD).float().cpu())
    return torch.cat(out).numpy()

def embed_thumb(Xs, bs=2048):
    # chunked: converting all 19.7k slices to float32 at once needs ~8 GB of RAM and crashes a free Colab session
    t = np.concatenate([F.avg_pool2d(torch.from_numpy(Xs[i:i + bs]).float().unsqueeze(1) / 255, 7).flatten(1).numpy()
                        for i in range(0, len(Xs), bs)])                                              # 32x32 thumbnails
    return t - t.mean(1, keepdims=True)

def build_Z(Xs, mode=GROUP_FEATURES, seed=0):
    """Returns (Z for clustering/neighbours, deep PCA features for probes)."""
    d = l2n(embed_deep(Xs)); dp = PCA(min(64, len(Xs) - 1, d.shape[1]), random_state=seed).fit_transform(d); dpn = l2n(dp)
    if mode == "deep": Z = dpn
    else:
        t = l2n(embed_thumb(Xs)); tp = l2n(PCA(min(32, len(Xs) - 1, t.shape[1]), random_state=seed).fit_transform(t))
        Z = np.concatenate([dpn, tp], 1)
    Z = l2n(Z + 1e-6 * np.random.default_rng(seed).standard_normal(Z.shape))   # avoids zero vectors (cosine NaN)
    return Z.astype(np.float32), dp.astype(np.float32)

fz = WORK / f"main_Z_{TAG}_{GROUP_FEATURES}.npz"
if fz.exists() and np.load(fz)["Z"].shape[0] == N: z_ = np.load(fz); Z, Zdeep = z_["Z"], z_["Zdeep"]
else: Z, Zdeep = build_Z(X); np.savez(fz, Z=Z, Zdeep=Zdeep)
print("Z", Z.shape, "Zdeep", Zdeep.shape)

# %%
# =============================== 6a-bis. DO FILE NUMBERS ENCODE SCAN ORDER? ===============================
# If consecutive file numbers (nCT7214, nCT7215, ...) are much more similar than random pairs of the same class,
# the numbering follows acquisition order and neighbouring numbers come from the same scan.
num = df.stem.str.extract(r"(\d+)$")[0]
if num.notna().all():
    num = num.astype(int).values; rng = np.random.default_rng(0)
    for c in range(3):
        idx = np.where(y == c)[0]
        if len(idx) < 3: continue
        idx = idx[np.argsort(num[idx])]
        adj = (Z[idx[:-1]] * Z[idx[1:]]).sum(1)
        p1, p2 = rng.permutation(idx), rng.permutation(idx)
        rnd = (Z[p1] * Z[p2]).sum(1)
        print(f"{CLASSES[c]:5s}: consecutive file numbers median cosine similarity {np.median(adj):.3f} | random same-class pairs {np.median(rnd):.3f}")
    print("If the first number is far above the second, file order carries scan identity -> a filename-order grouping would beat feature clustering.")
else:
    print("File names have no trailing number; skipping.")

# %%
# =============================== 6b. RECOVER PSEUDO-PATIENT GROUPS ===============================
# Average-linkage agglomerative clustering on cosine distance. One linkage, many cuts (K = number of clusters).
if N > 30000: print("WARNING: N is large; linkage needs O(N^2) memory (~N^2*4 bytes).")
t0 = time.time(); LK = linkage(pdist(Z, metric="cosine"), method="average"); print(f"linkage done in {time.time() - t0:.0f}s")
def cut(K): return fcluster(LK, t=min(K, N), criterion="maxclust") - 1

def group_stats(g, yy):
    sizes = np.bincount(g); majority = np.array([np.bincount(yy[g == k], minlength=3).max() for k in range(len(sizes))])
    return dict(groups=int((sizes > 0).sum()), largest_frac=sizes.max() / len(g), singleton_frac=(sizes == 1).sum() / max((sizes > 0).sum(), 1),
                mean_size=len(g) / max((sizes > 0).sum(), 1), class_purity=majority.sum() / len(g))
tab = pd.DataFrame({K: group_stats(cut(K), y) for K in GROUP_K_LIST}).T; tab.index.name = "K"
print(tab.round(3).to_string())
print("\nRead this table: largest_frac must stay small (otherwise one giant cluster ruins the split) and class_purity tells you whether "
      "clusters are class-pure (expected: high, because slices of one patient share a label).")

GROUPS = cut(GROUP_K)
print(f"\nOperating point: K={GROUP_K} -> {len(np.unique(GROUPS))} groups, largest = {np.bincount(GROUPS).max() / N:.1%} of the data")

# ---- a second, independent grouping from FILE-NUMBER ORDER. Consecutive numbers (nCT7214, nCT7215, ...) are neighbouring slices of one
# scan in this dataset (see the adjacency check above), so cluster along the number line only: contiguity-constrained average linkage.
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import diags
def filename_order_groups(Zs, yy, stems, k_total):
    num = pd.Series(list(stems)).str.extract(r"(\d+)$")[0]
    if num.isna().any(): return None
    num = num.astype(int).values; g = np.zeros(len(yy), int); off = 0
    for c in range(3):
        idx = np.where(yy == c)[0]; n = len(idx)
        if n < 4: g[idx] = off; off += 1; continue
        idx = idx[np.argsort(num[idx], kind="stable")]
        k = int(max(2, min(n - 1, round(k_total * n / len(yy)))))
        conn = diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], format="csr")
        try: ag = AgglomerativeClustering(n_clusters=k, connectivity=conn, linkage="average", metric="cosine")
        except TypeError: ag = AgglomerativeClustering(n_clusters=k, connectivity=conn, linkage="average", affinity="cosine")
        g[idx] = ag.fit_predict(Zs[idx]) + off; off += k
    return g
try: GROUPS_F = filename_order_groups(Z, y, df.stem.values, GROUP_K)
except Exception as e: print("file-order grouping failed:", e); GROUPS_F = None
if GROUPS_F is not None:
    st = group_stats(GROUPS_F, y)
    print(f"File-order groups: {st['groups']:.0f} groups, largest = {st['largest_frac']:.1%}, singletons = {st['singleton_frac']:.1%}, class purity = {st['class_purity']:.3f}")
    print(f"ARI(feature-cluster groups, file-order groups) = {adjusted_rand_score(GROUPS, GROUPS_F):.3f}  (1 = same grouping; two independent methods agreeing is evidence the groups are real scans)")
else: print("No usable trailing numbers in the file names: file-order grouping skipped.")
if df.fname_group.notna().any():
    print("ARI(recovered groups, file-name groups) =", round(adjusted_rand_score(df.fname_group.values, GROUPS), 3), "(only meaningful if file-name groups are real scan IDs)")

# %%
# =============================== 6c. LEAKAGE WITHOUT TRAINING: probes + nearest-neighbour similarity ===============================
def make_splits(yy, groups=None, seed=0, n_folds=7):
    """72 / 14 / 14 split. groups=None -> stratified random SLICE split (the original repo). Otherwise stratified GROUP split."""
    if groups is None: skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed); it = skf.split(np.zeros(len(yy)), yy)
    else: skf = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed); it = skf.split(np.zeros(len(yy)), yy, groups)
    folds = [te for _, te in it]
    sp = dict(test=folds[0], val=folds[1], train=np.concatenate(folds[2:]))
    for k in ("val", "test"):
        if len(np.unique(yy[sp[k]])) < len(np.unique(yy)): print(f"WARNING: split '{k}' is missing a class (too few groups?) - increase GROUP_K")
    if groups is not None:
        g = {k: set(groups[v]) for k, v in sp.items()}
        assert not (g["train"] & g["test"]) and not (g["train"] & g["val"]) and not (g["val"] & g["test"]), "group leak in split!"
    return sp

def probe(Fe, Zn, yy, sp):
    tr, te = sp["train"], sp["test"]; sc = StandardScaler().fit(Fe[tr])
    lr = LogisticRegression(max_iter=300, C=0.5, class_weight="balanced").fit(sc.transform(Fe[tr]), yy[tr])
    ba_lr = balanced_accuracy_score(yy[te], lr.predict(sc.transform(Fe[te])))
    nn_idx = np.concatenate([np.argmax(Zn[te[i:i + 2000]] @ Zn[tr].T, 1) for i in range(0, len(te), 2000)])
    return ba_lr, balanced_accuracy_score(yy[te], yy[tr][nn_idx])

curve = []
for K in GROUP_K_LIST + [N]:
    gK = None if K >= N else cut(K)
    r = np.array([probe(Zdeep, Z, y, make_splits(y, gK, s)) for s in range(3)])
    curve.append(dict(K=K, LR_balacc=r[:, 0].mean(), LR_sd=r[:, 0].std(), NN_balacc=r[:, 1].mean(), NN_sd=r[:, 1].std()))
curve = pd.DataFrame(curve); print(curve.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(6.4, 3.8)); xs = np.arange(len(curve))
for col, sd, c, lab in (("LR_balacc", "LR_sd", C1, "linear probe"), ("NN_balacc", "NN_sd", C2, "1-nearest-neighbour")):
    ax.errorbar(xs, curve[col], yerr=curve[sd], color=c, marker="o", capsize=3)
    ax.text(xs[0] + 0.12, curve[col].iloc[0] + (0.05 if col == "LR_balacc" else -0.05), lab, color=INK2, va="center", fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels([str(k) for k in GROUP_K_LIST] + ["slice\nsplit"]); ax.set_xlim(-0.3, len(xs) - 0.3)
ax.set_xlabel("number of recovered groups K (fewer = stricter split)"); ax.set_ylabel("balanced accuracy"); ax.set_title("Frozen features: accuracy as the split gets stricter")
savefig("04_probe_leakage_curve")

# nearest-neighbour similarity of every test slice to the training set
sp_s, sp_g = make_splits(y, None, 0), make_splits(y, GROUPS, 0)
def nn_sim(sp): te, tr = sp["test"], sp["train"]; return np.concatenate([(Z[te[i:i + 2000]] @ Z[tr].T).max(1) for i in range(0, len(te), 2000)])
s1, s2 = nn_sim(sp_s), nn_sim(sp_g)
fig, ax = plt.subplots(figsize=(6.4, 3.6)); bins = np.linspace(min(s1.min(), s2.min()), 1.0, 50)
ax.hist(s1, bins=bins, histtype="step", color=C1, lw=2, density=True); ax.hist(s2, bins=bins, histtype="step", color=C2, lw=2, density=True)
ax.set_yticks([]); ax.set_xlabel("cosine similarity to the closest TRAINING slice"); ax.set_title("How close is each test slice to something it was trained on?")
ax.text(0.02, 0.92, "random slice split", color=INK2, transform=ax.transAxes); ax.plot([0.0, 0.015], [0.935, 0.935], transform=ax.transAxes, color=C1, lw=2, clip_on=False)
ax.text(0.02, 0.84, f"group split (K={GROUP_K})", color=INK2, transform=ax.transAxes); ax.plot([0.0, 0.015], [0.855, 0.855], transform=ax.transAxes, color=C2, lw=2, clip_on=False)
savefig("05_nn_similarity")
print(f"median nearest-train similarity: slice split {np.median(s1):.3f} | group split {np.median(s2):.3f}")

# %% [markdown]
# ## 7. Training machinery
# 
# Same model as the original repo (MobileNetV3-Large -> 1280 -> ReLU -> Dropout 0.2 -> 3, AdamW 1e-4, weight decay 1e-4), with three deliberate differences: the whole slice is resized (no `CenterCrop`, which can cut off lung edges), the best epoch is chosen on **validation loss**, and augmentation runs on the GPU. Everything is saved per run, so an interrupted notebook resumes where it stopped.

# %%
# =============================== 7. TRAINING MACHINERY ===============================
class Head(nn.Module):
    def __init__(self, n=3):
        super().__init__(); self.net = nn.Sequential(nn.Linear(960, 1280), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1280, n))
    def forward(self, x): return self.net(x)

def build_model():
    m = get_backbone(); m.classifier = Head(); return m.to(DEVICE)

def make_scaler():
    try: return torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    except Exception: return torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

# everything lives on the device as uint8 (the whole dataset is only ~1 GB)
XT, MT = torch.from_numpy(X).to(DEVICE), torch.from_numpy(M).to(DEVICE)
EXT_T = {k: (torch.from_numpy(v["X"]).to(DEVICE), torch.from_numpy(v["M"]).to(DEVICE)) for k, v in EXT.items()}
yT = torch.as_tensor(y.astype(np.int64), device=DEVICE)
_gy, _gx = torch.meshgrid(torch.linspace(-1, 1, IMG), torch.linspace(-1, 1, IMG), indexing="ij"); R2 = (_gx ** 2 + _gy ** 2).to(DEVICE)

def augment(x, aug):
    """x: (B,1,H,W) in [0,1]. 'basic' = random zoom/shift + horizontal flip (as in the original repo).
       'robust' = basic + random intensity window + gamma + random circular field-of-view + noise (acquisition-style shifts)."""
    B = x.shape[0]; dev = x.device
    s = torch.empty(B, device=dev).uniform_(0.8, 1.0); flip = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = s * flip; theta[:, 1, 1] = s
    theta[:, 0, 2] = (torch.rand(B, device=dev) * 2 - 1) * (1 - s); theta[:, 1, 2] = (torch.rand(B, device=dev) * 2 - 1) * (1 - s)
    x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False), mode="bilinear", padding_mode="zeros", align_corners=False)
    if aug == "robust":
        x_basic = x
        lo = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.0, 0.25); hi = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.75, 1.0)
        x = ((x - lo) / (hi - lo)).clamp(0, 1)
        x = x.clamp_min(1e-4) ** torch.exp(torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.4, 0.4))
        r = torch.empty(B, 1, 1, device=dev).uniform_(0.85, 1.1); use = (torch.rand(B, 1, 1, device=dev) < 0.3)
        fov = (R2.unsqueeze(0) <= r ** 2) | ~use
        x = x * fov.unsqueeze(1).float()
        x = (x + 0.02 * torch.randn_like(x)).clamp(0, 1)
        x = torch.where(torch.rand(B, 1, 1, 1, device=dev) < 0.5, x_basic, x)   # half of every batch stays clean (keeps BN statistics sane)
    return x

def prep(xb, mb, mode, aug, train):
    x = xb.float().div(255).unsqueeze(1)
    if mode == "lung_only": x = x * mb.unsqueeze(1).float()
    elif mode == "no_lung": x = x * (1 - mb.unsqueeze(1).float())
    if train: x = augment(x, aug)
    return (x.expand(-1, 3, -1, -1) - MEAN) / STD

@torch.no_grad()
def predict(model, XT_, MT_, idx=None, mode="full", bs=256):
    model.eval(); n = len(XT_) if idx is None else len(idx); out = []
    for i in range(0, n, bs):
        sel = slice(i, i + bs) if idx is None else torch.as_tensor(idx[i:i + bs], device=DEVICE)
        with amp_ctx(): out.append(model(prep(XT_[sel], MT_[sel], mode, "basic", False)).float().cpu())
    return torch.cat(out).numpy()

@torch.no_grad()
def bn_recalibrate(model, idx, mode, n=2048, bs=128):
    """Re-estimate BatchNorm running statistics on CLEAN (un-augmented) training images. After training under strong
    augmentation the running statistics describe augmented images, not the clean images used at test time."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bns: m.reset_running_stats(); m.momentum = None          # None = cumulative average
    sel = idx[:n] if len(idx) <= n else np.random.default_rng(0).choice(idx, n, replace=False)
    model.train()
    for i in range(0, len(sel), bs):
        j = torch.as_tensor(sel[i:i + bs], device=DEVICE)
        with amp_ctx(): model(prep(XT[j], MT[j], mode, "basic", False))
    for m in bns: m.momentum = 0.1
    model.eval()

def run_key(name, mode, aug):
    return name + ("_a2" if aug == "robust" else "") + (f"_m{MASK_VERSION}" if mode != "full" else "")   # old caches are not reused when the recipe changed

def train_run(name, sp, mode="full", aug="basic", seed=0, save_weights=False):
    key = run_key(name, mode, aug); f = RUNS / f"{key}_s{seed}.npz"
    if f.exists():
        r_old = dict(np.load(f, allow_pickle=True))
        if all(("ext_" + k) in r_old for k in EXT_T): return r_old
        print(f"  [{key} s{seed}] cached run has no predictions for a newly added external set -> retraining")
    set_seed(seed); model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); scaler = make_scaler()
    tr = torch.as_tensor(sp["train"], device=DEVICE); best_loss, best_state, hist = 1e9, None, []; t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = tr[torch.randperm(len(tr), device=DEVICE)]; tl, nb = 0.0, 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2: continue
            xb = prep(XT[idx], MT[idx], mode, aug, True)
            with amp_ctx(): loss = F.cross_entropy(model(xb), yT[idx])
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item(); nb += 1
        if aug == "robust": bn_recalibrate(model, sp["train"], mode)
        zv = predict(model, XT, MT, sp["val"], mode)
        vl = float(F.cross_entropy(torch.from_numpy(zv), torch.from_numpy(y[sp["val"]].astype(np.int64))))
        vba = balanced_accuracy_score(y[sp["val"]], zv.argmax(1)); hist.append([tl / max(nb, 1), vl, vba, time.time() - t0])
        print(f"  [{key} s{seed}] ep {ep + 1}/{EPOCHS}  train {tl / max(nb, 1):.4f}  val loss {vl:.4f}  val bal-acc {vba:.4f}  ({time.time() - t0:.0f}s)")
        if vl < best_loss: best_loss, best_state = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    res = dict(z_val=predict(model, XT, MT, sp["val"], mode), z_test=predict(model, XT, MT, sp["test"], mode),
               idx_train=sp["train"], idx_val=sp["val"], idx_test=sp["test"], hist=np.array(hist))
    for k, (xe, me) in EXT_T.items(): res["ext_" + k] = predict(model, xe, me, None, mode)
    np.savez(f, **res)
    if save_weights: torch.save({"state_dict": best_state, "mode": mode}, RUNS / f"{key}_s{seed}.pt")
    del model, opt; torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
    return res

def load_model(name, seed):
    m = build_model(); m.load_state_dict(torch.load(RUNS / f"{name}_s{seed}.pt", map_location=DEVICE)["state_dict"]); return m.eval()
print("ready")

# %% [markdown]
# ## 8. The experiments
# 
# | id | split | input | augmentation | question |
# |----|-------|-------|--------------|----------|
# | **E1** | random **slice** split (as in the repo) | full slice | basic | reproduces the headline number |
# | **E2** | **group** split (recovered pseudo-patients) | full slice | basic | how much is lost when near-duplicates cannot cross the split? |
# | **E3** | group | **lungs removed** | basic | can it score well *without lungs*? (shortcut test) |
# | **E4** | group | **lungs only** (rest blacked out) | basic | is the lung region sufficient? |
# | **E5** | group | full slice | **robust** | does intensity / FOV augmentation help under the stricter protocol? |
# | **E6** | group | lungs only | robust | combined |
# 
# E1/E2/E5 use `SEEDS`; the ablations use `SEEDS_ABL`. About 2-3 minutes per run on a T4.

# %%
# =============================== 8. RUN EXPERIMENTS ===============================
all_seeds = sorted(set(SEEDS) | set(SEEDS_ABL))
SPLITS = {("slice", s): make_splits(y, None, s) for s in all_seeds}
SPLITS.update({("group", s): make_splits(y, GROUPS, s) for s in all_seeds})
if GROUPS_F is not None: SPLITS.update({("groupF", s): make_splits(y, GROUPS_F, s) for s in all_seeds})
EXPERIMENTS = [  # name, split, input mode, augmentation, seeds
    ("E1_slice_full",            "slice", "full",      "basic",  SEEDS),
    ("E2_group_full",            "group", "full",      "basic",  SEEDS),
    ("E3_group_nolung",          "group", "no_lung",   "basic",  SEEDS_ABL),
    ("E4_group_lungonly",        "group", "lung_only", "basic",  SEEDS_ABL),
    ("E5_group_full_robust",     "group", "full",      "robust", SEEDS),
    ("E6_group_lungonly_robust", "group", "lung_only", "robust", SEEDS_ABL),
]
if GROUPS_F is not None: EXPERIMENTS.insert(2, ("E2f_fileorder_group_full", "groupF", "full", "basic", SEEDS))   # same as E2, groups from file order
EXP_KEY = {n_: run_key(n_, m_, a_) for n_, _, m_, a_, _ in EXPERIMENTS}
GROUPS_OF = {"slice": GROUPS, "group": GROUPS, "groupF": GROUPS_F}
t0 = time.time()
for name, split, mode, aug, seeds in EXPERIMENTS:
    for s in seeds:
        r = train_run(name, SPLITS[(split, s)], mode, aug, s, save_weights=(s == seeds[0]))
        te = r["idx_test"]; print(f"{name} seed {s}: test acc {accuracy_score(y[te], r['z_test'].argmax(1)):.4f}  (n={len(te)})")
print(f"all runs done in {(time.time() - t0) / 60:.1f} min")

# %% [markdown]
# ## 8b. Statistics that do not lie
# 
# * **Wilson interval**: assumes test slices are independent. They are not.
# * **Cluster bootstrap**: resamples whole *groups*; this is the honest interval.
# * **Design effect** `deff = (SE_cluster / SE_binomial)^2` and `n_eff = n / deff`: how many *independent* test samples the test set is really worth.
# * **McNemar (exact)** for paired model comparisons on the same test slices.

# %%
# =============================== 8b. METRICS + STATISTICS ===============================
def softmax_np(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def ece_multi(p, yy, bins=15):
    conf, pred = p.max(1), p.argmax(1); edges = np.linspace(0, 1, bins + 1); e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any(): e += m.mean() * abs((pred[m] == yy[m]).mean() - conf[m].mean())
    return e

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d

def cluster_boot(pred, yy, groups, B=None, seed=0):
    B = B or N_BOOT; ug, inv = np.unique(groups, return_inverse=True); G = len(ug)
    C = np.zeros((G, 3, 3)); np.add.at(C, (inv, yy, pred), 1)
    w = np.random.default_rng(seed).multinomial(G, np.ones(G) / G, size=B).astype(float)
    tot = np.einsum("bg,gij->bij", w, C); acc = np.trace(tot, axis1=1, axis2=2) / np.maximum(tot.sum((1, 2)), 1)
    rs = tot.sum(2); rec = np.divide(np.einsum("bii->bi", tot), rs, out=np.full(rs.shape, np.nan), where=rs > 0); ba = np.nanmean(rec, 1)
    n = len(yy); p = float(np.clip((pred == yy).mean(), 0.5 / n, 1 - 0.5 / n)); se_b = math.sqrt(p * (1 - p) / n)
    deff = max((acc.std(ddof=1) / se_b) ** 2, 1.0)
    return dict(acc_ci_lo=np.percentile(acc, 2.5), acc_ci_hi=np.percentile(acc, 97.5), ba_ci_lo=np.percentile(ba, 2.5),
                ba_ci_hi=np.percentile(ba, 97.5), deff=deff, n_eff=n / deff, n_groups=G)

def metrics(z, yy, groups=None):
    p = softmax_np(z); pred = p.argmax(1); n = len(yy)
    rec = [(pred[yy == k] == k).mean() if (yy == k).any() else np.nan for k in range(3)]
    out = dict(n=n, acc=(pred == yy).mean(), bal_acc=np.nanmean(rec), recall_nCT=rec[0], recall_pCT=rec[1], recall_NiCT=rec[2],
               macro_f1=f1_score(yy, pred, average="macro"), nll=float(-np.log(p[np.arange(n), yy] + 1e-12).mean()), ece=ece_multi(p, yy))
    m = yy < 2
    if m.sum() and len(set(yy[m])) == 2: out["auroc_n_vs_p"] = roc_auc_score(yy[m], p[m, 1] / (p[m, 0] + p[m, 1] + 1e-12))
    out["acc_wilson_lo"], out["acc_wilson_hi"] = wilson(int((pred == yy).sum()), n)
    if groups is not None: out.update(cluster_boot(pred, yy, groups))
    return out

def mcnemar_exact(c1, c2):
    b, c = int(((c1) & (~c2)).sum()), int(((~c1) & (c2)).sum())
    return b, c, (1.0 if b + c == 0 else stats.binomtest(min(b, c), b + c, 0.5).pvalue)

def load_run(name, s): return dict(np.load(RUNS / f"{EXP_KEY.get(name, name)}_s{s}.npz", allow_pickle=True))

rows = []
for name, split, mode, aug, seeds in EXPERIMENTS:
    for s in seeds:
        r = load_run(name, s); te = r["idx_test"]; m = metrics(r["z_test"], y[te], GROUPS_OF[split][te]); m.update(exp=name, seed=s, split=split, mode=mode, aug=aug); rows.append(m)
RES = pd.DataFrame(rows); RES.to_csv(WORK / f"results_per_run_{TAG}.csv", index=False)
cols = ["acc", "bal_acc", "recall_nCT", "recall_pCT", "recall_NiCT", "auroc_n_vs_p", "ece", "nll"]
agg = RES.groupby("exp", sort=False)[cols].agg(["mean", "std"])
summary = pd.DataFrame({c: [f"{agg[c]['mean'][e]:.4f} +/- {np.nan_to_num(agg[c]['std'][e]):.4f}" for e in agg.index] for c in cols}, index=agg.index)
print(summary.to_string())

e1, e2 = RES[RES.exp == "E1_slice_full"].set_index("seed"), RES[RES.exp == "E2_group_full"].set_index("seed")
common = sorted(set(e1.index) & set(e2.index)); gap = (e1.acc[common] - e2.acc[common])
print(f"\n=== LEAKAGE GAP  (slice split - group split) ===\n  accuracy: {e1.acc.mean():.4f} -> {e2.acc.mean():.4f}   gap = {gap.mean() * 100:.2f} points  (per seed: {np.round(gap.values * 100, 2).tolist()})")
print(f"  balanced accuracy: {e1.bal_acc.mean():.4f} -> {e2.bal_acc.mean():.4f}")
if "E2f_fileorder_group_full" in RES.exp.values:
    e2f = RES[RES.exp == "E2f_fileorder_group_full"].set_index("seed"); cf = sorted(set(e1.index) & set(e2f.index))
    print(f"  same comparison with FILE-ORDER groups: {e1.acc[cf].mean():.4f} -> {e2f.acc[cf].mean():.4f}   gap = {(e1.acc[cf] - e2f.acc[cf]).mean() * 100:.2f} points  (two independent grouping methods)")
print(f"  COVID-positive recall: {e1.recall_pCT.mean():.4f} -> {e2.recall_pCT.mean():.4f}")
print(f"\n=== EFFECTIVE SAMPLE SIZE ===\n  E1 test set: n={int(e1.n.mean())} slices from ~{int(e1.n_groups.mean())} groups -> design effect {e1.deff.mean():.1f}, n_eff ~ {e1.n_eff.mean():.0f}")
print(f"  E1 accuracy 95% CI: Wilson [{e1.acc_wilson_lo.mean():.4f}, {e1.acc_wilson_hi.mean():.4f}]   vs cluster bootstrap [{e1.acc_ci_lo.mean():.4f}, {e1.acc_ci_hi.mean():.4f}]")
for s in sorted(set(SEEDS) & set(RES[RES.exp == "E5_group_full_robust"].seed)):
    ra, rb = load_run("E2_group_full", s), load_run("E5_group_full_robust", s)
    if (ra["idx_test"] == rb["idx_test"]).all():
        b, c, p = mcnemar_exact(ra["z_test"].argmax(1) == y[ra["idx_test"]], rb["z_test"].argmax(1) == y[rb["idx_test"]])
        print(f"  McNemar E2 vs E5 seed {s}: E2-only-correct={b}, E5-only-correct={c}, p={p:.3f}")

# --- figure: headline comparison (balanced accuracy, individual seeds as dots, mean as bar tick)
order = [e for e in RES.exp.unique()]; fig, ax = plt.subplots(figsize=(7.4, 0.55 * len(order) + 1.4))
for i, e in enumerate(order[::-1]):
    v = RES[RES.exp == e].bal_acc.values; ax.scatter(v, [i] * len(v), color=C1, s=36, zorder=3, edgecolor=SURF, linewidth=1.2)
    ax.plot([v.mean()] * 2, [i - 0.28, i + 0.28], color=INK, lw=2, zorder=4); ax.text(v.max() + 0.004, i, f"{v.mean():.3f}", va="center", fontsize=9, color=INK2)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order[::-1]); ax.axvline(1 / 3, color=GRID, lw=1.5); ax.set_xlabel("balanced accuracy on the test split (dots = seeds, bar = mean)")
ax.set_title("Same model, different protocols"); ax.grid(axis="y", visible=False); savefig("06_headline_comparison")

# %% [markdown]
# ## 9. Shortcut analysis
# 
# Four independent lines of evidence. A model is only trustworthy if *all four* say the lungs matter.
# 
# 1. **Metadata-only classifier**: gradient boosting on image size, file size, brightness, corner/edge statistics, black-pixel fraction. No pixel of lung is used.
# 2. **Training with lungs removed (E3)** vs full (E2). *Shortcut Reliance Index* `SRI = (BA_nolung - 1/3) / (BA_full - 1/3)`. Near 1: the lungs are not needed. Near 0: they are.
# 3. **Test-time ablation** of the E2 model.
# 4. **Grad-CAM mass inside the lung mask**, relative to the lung's share of the image (enrichment 1.0 = no better than uniform).

# %%
# =============================== 9a. METADATA-ONLY BASELINE + SRI ===============================
from sklearn.ensemble import HistGradientBoostingClassifier
M_ALL = meta.values.astype(np.float32); cols_all = list(meta.columns)
def meta_ba(cols, split, seeds=(0, 1, 2)):
    ci = [cols_all.index(c) for c in cols]; out = []
    for s in seeds:
        sp = SPLITS.get((split, s)) or make_splits(y, None if split == "slice" else GROUPS, s)
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0, class_weight="balanced").fit(M_ALL[sp["train"]][:, ci], y[sp["train"]])
        out.append(balanced_accuracy_score(y[sp["test"]], clf.predict(M_ALL[sp["test"]][:, ci])))
    return np.mean(out), np.std(out)
rows = []
for nm, cs in {"image + file size only": ["w", "h", "aspect", "nbytes", "bytes_per_px"], "all scalar acquisition features": cols_all}.items():
    for sp_ in ("slice", "group"):
        m_, s_ = meta_ba(cs, sp_); rows.append(dict(features=nm, split=sp_, balanced_acc=f"{m_:.3f} +/- {s_:.3f}"))
print("Metadata-only classifiers (chance = 0.333):"); print(pd.DataFrame(rows).to_string(index=False))

def ba_of(exp): return RES[RES.exp == exp].bal_acc.mean() if (RES.exp == exp).any() else np.nan
def au_of(exp): return RES[RES.exp == exp].auroc_n_vs_p.mean() if (RES.exp == exp).any() else np.nan
ba_full = ba_of("E2_group_full")
t = pd.DataFrame({"balanced_acc": {e: ba_of(e) for e in ["E2_group_full", "E3_group_nolung", "E4_group_lungonly", "E5_group_full_robust", "E6_group_lungonly_robust"]},
                  "AUROC_nCT_vs_pCT": {e: au_of(e) for e in ["E2_group_full", "E3_group_nolung", "E4_group_lungonly", "E5_group_full_robust", "E6_group_lungonly_robust"]}})
t["SRI"] = (t.balanced_acc - 1 / 3) / (ba_full - 1 / 3); print("\nSRI (1 = lungs not needed):"); print(t.round(3).to_string())

# %%
# =============================== 9b. TEST-TIME ABLATION + GRAD-CAM ===============================
model2 = load_model("E2_group_full", SEEDS[0]); sp0 = SPLITS[("group", SEEDS[0])]; te0 = sp0["test"]
rows = []
for md_ in ("full", "lung_only", "no_lung"):
    m_ = metrics(predict(model2, XT, MT, te0, md_), y[te0]); m_["test_time_input"] = md_; rows.append(m_)
print("E2 model, test slices shown in different ways:"); print(pd.DataFrame(rows).set_index("test_time_input")[["acc", "bal_acc", "recall_nCT", "recall_pCT", "recall_NiCT", "auroc_n_vs_p"]].round(3).to_string())

def gradcam_enrichment(model, idx, bs=32):
    out = []
    for i in range(0, len(idx), bs):
        j = torch.as_tensor(idx[i:i + bs], device=DEVICE); xb = prep(XT[j], MT[j], "full", "basic", False); acts = {}
        def hook(m, inp, o): o.retain_grad(); acts["a"] = o
        h = model.features.register_forward_hook(hook)
        with torch.enable_grad():
            lg = model(xb); tgt = lg.argmax(1); model.zero_grad(set_to_none=True); lg.gather(1, tgt[:, None]).sum().backward()
        h.remove(); a = acts["a"]; w = a.grad.mean((2, 3), keepdim=True); cam = F.relu((w * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG, IMG), mode="bilinear", align_corners=False)[:, 0]; cam = cam / (cam.sum((1, 2), keepdim=True) + 1e-8)
        mk = MT[j].float(); out.append(torch.stack([(cam * mk).sum((1, 2)), mk.mean((1, 2))], 1).detach().cpu())
    return torch.cat(out).numpy()

area_te = M[te0].reshape(len(te0), -1).mean(1); sel = {c: te0[(y[te0] == c) & (area_te >= 0.05)][:300] for c in (0, 1)}
fig, ax = plt.subplots(figsize=(6.4, 3.6)); bins = np.linspace(0, 3, 40)
for c, col in ((0, C1), (1, C2)):
    if len(sel[c]) == 0: continue
    g_ = gradcam_enrichment(model2, sel[c]); enr = g_[:, 0] / np.maximum(g_[:, 1], 1e-6)
    ax.hist(enr.clip(0, 3), bins=bins, histtype="step", color=col, lw=2, density=True)
    ax.text(0.98, 0.9 - 0.09 * c, f"{CLASSES[c]}: median {np.median(enr):.2f}", transform=ax.transAxes, ha="right", color=INK2, fontsize=9)
    print(f"{CLASSES[c]}: median Grad-CAM enrichment inside lungs = {np.median(enr):.2f} (n={len(enr)})")
ax.axvline(1, color=INK2, lw=1, ls="--"); ax.set_yticks([]); ax.set_xlabel("Grad-CAM mass in lungs / lung share of image  (1 = uniform)"); ax.set_title("Does the model look at the lungs?")
savefig("07_gradcam_enrichment")

# a few CAMs to look at
show = np.concatenate([sel[0][:3], sel[1][:3]]) if len(sel[0]) >= 3 and len(sel[1]) >= 3 else np.array([], int)
if len(show):
    j = torch.as_tensor(show, device=DEVICE); xb = prep(XT[j], MT[j], "full", "basic", False); acts = {}
    def _hook(m, inp, o): o.retain_grad(); acts["a"] = o          # a forward hook must return None
    h = model2.features.register_forward_hook(_hook)
    with torch.enable_grad():
        lg = model2(xb); model2.zero_grad(set_to_none=True); lg.gather(1, lg.argmax(1)[:, None]).sum().backward()
    h.remove(); a = acts["a"]; cam = F.relu(((a.grad.mean((2, 3), keepdim=True)) * a).sum(1, keepdim=True)); cam = F.interpolate(cam, size=(IMG, IMG), mode="bilinear")[:, 0].detach().cpu().numpy()
    fig, axs = plt.subplots(2, len(show), figsize=(2.1 * len(show), 4.4))
    for k, i in enumerate(show):
        axs[0, k].imshow(X[i], cmap="gray"); axs[0, k].set_title(CLASSES[y[i]], fontsize=9, loc="left"); axs[1, k].imshow(X[i], cmap="gray"); axs[1, k].imshow(cam[k] / (cam[k].max() + 1e-8), cmap="inferno", alpha=0.5)
        axs[0, k].axis("off"); axs[1, k].axis("off")
    savefig("08_gradcam_examples")
del model2

# %% [markdown]
# ## 10. External validation
# 
# Binary task: **COVID-19 vs not**. The model's score is `P(pCT)`. Reported: AUROC, average precision, and sensitivity / specificity at the model's own decision (argmax = pCT), plus the share of slices the model calls *non-informative* (a sign of distribution shift).
# 
# Read the numbers with these caveats: external A has **no patient IDs** (its intervals are optimistic), its negatives are *other lung diseases* (a harder, differently-defined negative class than "normal"), and many of its images are tightly cropped.

# %%
# =============================== 10. EXTERNAL EVALUATION ===============================
def ext_metrics(z, ye, B=None):
    p = softmax_np(z); s = p[:, 1]; pos = p.argmax(1) == 1
    d = dict(n=len(ye), auroc=roc_auc_score(ye, s), ap=average_precision_score(ye, s), sens=pos[ye == 1].mean(), spec=(~pos[ye == 0]).mean(), frac_pred_NiCT=(p.argmax(1) == 2).mean())
    rng = np.random.default_rng(0); bs = []
    for _ in range(B or max(N_BOOT // 5, 50)):
        i = rng.integers(0, len(ye), len(ye))
        if len(set(ye[i])) == 2: bs.append(roc_auc_score(ye[i], s[i]))
    d["auroc_lo"], d["auroc_hi"] = np.percentile(bs, [2.5, 97.5]); return d

ext_rows = []
for en, ed in EXT.items():
    for name, split, mode, aug, seeds in EXPERIMENTS:
        for s in seeds:
            r = load_run(name, s); m_ = ext_metrics(r["ext_" + en], ed["y"]); m_.update(ext=en, exp=name, seed=s); ext_rows.append(m_)
EXTRES = pd.DataFrame(ext_rows)
if len(EXTRES):
    EXTRES.to_csv(WORK / f"results_external_{TAG}.csv", index=False)
    for en in EXT:
        a = EXTRES[EXTRES.ext == en].groupby("exp", sort=False)[["auroc", "ap", "sens", "spec", "frac_pred_NiCT"]].agg(["mean", "std"])
        t = pd.DataFrame({c: [f"{a[c]['mean'][e]:.3f} +/- {np.nan_to_num(a[c]['std'][e]):.3f}" for e in a.index] for c in ["auroc", "ap", "sens", "spec", "frac_pred_NiCT"]}, index=a.index)
        t.insert(0, "in-domain AUROC nCT/pCT", [f"{au_of(e):.3f}" for e in a.index]); print(f"\n=== EXTERNAL: {en}  ({EXT[en]['desc']}) ==="); print(t.to_string())

    # ROC curves (seed 0) for the three headline protocols on every external set
    fig, axs = plt.subplots(1, len(EXT), figsize=(5.2 * len(EXT), 4.2), squeeze=False)
    for ax, (en, ed) in zip(axs[0], EXT.items()):
        for (name, col, lab) in (("E1_slice_full", C1, "E1 slice split"), ("E2_group_full", C2, "E2 group split"), ("E5_group_full_robust", C3, "E5 group + robust aug")):
            r = load_run(name, SEEDS[0]); fpr, tpr, _ = roc_curve(ed["y"], softmax_np(r["ext_" + en])[:, 1]); ax.plot(fpr, tpr, color=col)
            ax.text(0.40, 0.30 - 0.08 * ("E1", "E2", "E5").index(name[:2]), f"{lab}  AUC {roc_auc_score(ed['y'], softmax_np(r['ext_' + en])[:, 1]):.3f}", color=INK2, fontsize=9)
            ax.plot([0.34, 0.38], [0.31 - 0.08 * ("E1", "E2", "E5").index(name[:2])] * 2, color=col, lw=2)
        ax.plot([0, 1], [0, 1], color=GRID, lw=1.2); ax.set_title(f"External: {en}"); ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    savefig("09_external_roc")
else:
    print("No external dataset available - attach at least one (Section 5). This is the single most important experiment.")

# %% [markdown]
# ## 11. Calibration
# 
# The original demo shows probabilities of exactly 100 %. Temperature scaling (one parameter `T` fitted on the **validation** set, `softmax(z / T)`) cannot change accuracy; it only changes confidence. ECE is measured before and after, in-domain and externally.

# %%
# =============================== 11. CALIBRATION ===============================
from scipy.optimize import minimize_scalar
def fit_temperature(z, yy):
    def nll(logT): p = softmax_np(z / np.exp(logT)); return -np.log(p[np.arange(len(yy)), yy] + 1e-12).mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))

def ece_binary(p, yy, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0; pts = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b if b < 1 else p <= b)
        if m.any(): e += m.mean() * abs(yy[m].mean() - p[m].mean()); pts.append((p[m].mean(), yy[m].mean()))
    return e, np.array(pts)

rows = []
for name, split, mode, aug, seeds in EXPERIMENTS:
    for s in seeds:
        r = load_run(name, s); T = fit_temperature(r["z_val"], y[r["idx_val"]]); yt_ = y[r["idx_test"]]
        row = dict(exp=name, seed=s, T=T, ECE_test=ece_multi(softmax_np(r["z_test"]), yt_), ECE_test_cal=ece_multi(softmax_np(r["z_test"] / T), yt_),
                   NLL_test=metrics(r["z_test"], yt_)["nll"], NLL_test_cal=metrics(r["z_test"] / T, yt_)["nll"])
        for en, ed in EXT.items():
            ze = r["ext_" + en]; pe = lambda zz: (lambda q: q[:, 1] / (q[:, 0] + q[:, 1] + 1e-12))(softmax_np(zz))
            row[f"ECE_{en}"] = ece_binary(pe(ze), ed["y"])[0]; row[f"ECE_{en}_cal"] = ece_binary(pe(ze / T), ed["y"])[0]
        rows.append(row)
CAL = pd.DataFrame(rows); CAL.to_csv(WORK / f"results_calibration_{TAG}.csv", index=False)
print(CAL.groupby("exp", sort=False).mean(numeric_only=True).drop(columns="seed").round(4).to_string())

# reliability diagram: E2 seed 0, external vs in-domain (binary P(COVID))
r = load_run("E2_group_full", SEEDS[0]); T = fit_temperature(r["z_val"], y[r["idx_val"]]); yt_ = y[r["idx_test"]]
sets = [("in-domain test (nCT vs pCT)", r["z_test"], (yt_ == 1).astype(int), yt_ < 2)] + [(f"external {en}", r["ext_" + en], ed["y"], np.ones(len(ed["y"]), bool)) for en, ed in EXT.items()]
fig, axs = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 4.2), squeeze=False)
for ax, (ttl, zz, yy, msk) in zip(axs[0], sets):
    f_ = lambda z_: (lambda q: q[:, 1] / (q[:, 0] + q[:, 1] + 1e-12))(softmax_np(z_[msk]))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5)
    for zz_, col, lab in ((zz, C1, "raw"), (zz / T, C2, f"temperature-scaled (T={T:.2f})")):
        e_, pts = ece_binary(f_(zz_), yy[msk])
        if len(pts): ax.plot(pts[:, 0], pts[:, 1], color=col, marker="o"); ax.text(0.03, 0.93 - 0.07 * (col == C2), f"{lab}: ECE {e_:.3f}", color=INK2, fontsize=9, transform=ax.transAxes)
    ax.set_xlabel("predicted P(COVID-19)"); ax.set_ylabel("observed fraction positive"); ax.set_title(ttl, fontsize=10)
savefig("10_reliability")

# %% [markdown]
# ## 12. (Optional) Validate the patient-group recovery against real patient IDs
# 
# This is what turns "a clustering heuristic" into "a method with a measured error". It needs **external B** (COVID-CT-MD or any dataset with real patient IDs). We run the *same* embedding + clustering recipe on B, ignore the IDs, and ask: (a) do the clusters match the true patients (ARI, pairwise precision/recall)? (b) does a split on **recovered** clusters give the same leakage gap as a split on **true** patients? If the recovered split is *stricter* than the true one, your Section 8 gap is an upper bound; if it is more lenient, it is a lower bound.

# %%
# =============================== 12. GROUP-RECOVERY VALIDATION (needs external B) ===============================
def pair_prf(true, pred):
    ct = pd.crosstab(true, pred).values.astype(np.int64); c2 = lambda a: a * (a - 1) // 2
    tp = c2(ct).sum(); return tp / max(c2(ct.sum(0)).sum(), 1), tp / max(c2(ct.sum(1)).sum(), 1)

if "B_covidctmd" in EXT:
    eb = EXT["B_covidctmd"]; ZB, ZdB = build_Z(eb["X"]); true = pd.factorize(eb["groups"])[0]; yb_ = eb["y"]; P = len(set(true))
    LKB = linkage(pdist(ZB, metric="cosine"), method="average"); rows = []
    for K in sorted({max(2, P // 4), max(2, P // 2), P, 2 * P, 4 * P}):
        g = fcluster(LKB, t=K, criterion="maxclust") - 1; pr, rc = pair_prf(true, g)
        rows.append(dict(K=K, true_patients=P, ARI=adjusted_rand_score(true, g), pair_precision=pr, pair_recall=rc, largest_cluster_frac=np.bincount(g).max() / len(g)))
    print("Recovery quality on slices with KNOWN patients:"); print(pd.DataFrame(rows).round(3).to_string(index=False))

    gP = fcluster(LKB, t=P, criterion="maxclust") - 1; out = []
    for nm, grp in (("random slice split", None), ("TRUE patient split", true), (f"recovered clusters (K={P})", gP)):
        r_ = np.array([probe(ZdB, ZB, yb_, make_splits(yb_, grp, s)) for s in range(3)]); out.append(dict(split=nm, LR_balanced_acc=f"{r_[:, 0].mean():.3f}", NN_balanced_acc=f"{r_[:, 1].mean():.3f}"))
    print("\nProbe accuracy on B under three protocols:"); print(pd.DataFrame(out).to_string(index=False))
    print("\nIf 'recovered' is close to 'TRUE', the recovery method is a faithful stand-in. If it is LOWER, your main-dataset gap is an upper bound.")
else:
    print("External B not provided -> skipping. (Without it, say clearly in your report that the groups are recovered and unvalidated.)")

# %% [markdown]
# ## 13. Write-up helper
# 
# Prints the numbers you need, with the right caveats attached, and saves everything under the output folder.

# %%
# =============================== 13. SUMMARY ===============================
try: SUMMARY_MD = summary.to_markdown()
except Exception: SUMMARY_MD = summary.to_string()
lines = ["# Results summary (auto-generated)", f"- Data: {N} slices, class counts {dict(zip(CLASSES, np.bincount(y).tolist()))}; recovered groups K={GROUP_K} ({len(np.unique(GROUPS))} clusters).",
         f"- Seeds: {SEEDS} (ablations: {SEEDS_ABL}); epochs={EPOCHS}; batch={BATCH}; lr={LR}.", "",
         "## Headline", f"- Slice split accuracy {e1.acc.mean():.4f}; group split accuracy {e2.acc.mean():.4f}; gap {gap.mean() * 100:.2f} points.",
         f"- COVID-positive recall: slice split {e1.recall_pCT.mean():.4f}; group split {e2.recall_pCT.mean():.4f}.",
         f"- Test set of {int(e1.n.mean())} slices has design effect ~{e1.deff.mean():.1f} (n_eff ~ {e1.n_eff.mean():.0f}).", "",
         "## Per-experiment (mean +/- sd over seeds)", SUMMARY_MD, "",
         "## Caveats to keep in the paper", "- Groups are recovered by clustering, not true patient IDs" + (" (checked against true patient IDs on Tehran in Section 12: recovery is poor, ARI <= 0.29, so the slice-vs-group gap here is a LOWER BOUND on true patient-level leakage)." if "B_covidctmd" in EXT else " (NOT validated)."),
         "- Lung masks are classical image processing; inspect fig 03 before citing any lung/no-lung number.",
         "- External A has no patient IDs; its negatives are other lung diseases; many images are cropped.", "- Research use only; not a medical device."]
try: (WORK / f"RESULTS_{TAG}.md").write_text("\n".join(lines)); print("\n".join(lines))
except Exception as e: print("could not write summary:", e)
print("\nFiles in", WORK.resolve(), ":", sorted(p.name for p in WORK.iterdir())[:30])

# %% [markdown]
# ## What to do with the results
# 
# * **If the group-split gap is large** (several points, or COVID-recall drops): the original number mostly measured near-duplicate memorisation. That is the finding.
# * **If the lung-removed model (E3) is far above chance**, or the metadata-only classifier is strong: the dataset contains acquisition shortcuts. Report the SRI.
# * **If external AUROC is far below in-domain**: state the drop plainly; do not tune on the external set to close it.
# * **If E5 helps externally but not in-domain** (or vice versa), report both. Negative results are results.
# * Do **not** claim clinical usefulness. State: recovered (not true) groups, heuristic masks, one external set without patient IDs, three seeds.
