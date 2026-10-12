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