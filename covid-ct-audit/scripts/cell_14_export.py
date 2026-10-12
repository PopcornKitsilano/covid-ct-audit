# =============================== 14. EXPORT ALL OUTPUTS ===============================
# Run this LAST, after pressing Ctrl+S (the saved .ipynb is what gets read).
# It asks the running Jupyter server for THIS notebook's saved JSON, so it does not matter
# which drive the .ipynb lives on. Writes:
#   export/all_outputs.txt   every code cell's text output, in order, with cell headers
#   export/images/*.png      every figure
#   export/figures.zip       the same figures, zipped
import json, os, re, base64, zipfile, time, glob, urllib.request, urllib.parse
from pathlib import Path

EXPORT_DIR = Path(os.getcwd()) / "export"      # = E:\covid_audit\export after the CONFIG os.chdir
MARK = "# =============================== 14. EXPORT ALL OUTPUTS"


def _from_server():
    """Return (notebook_dict, description) via the Jupyter contents API, or None."""
    try:
        from ipykernel import get_connection_file
        m = re.search(r"kernel-(.+)\.json", Path(get_connection_file()).name)
        kid = m.group(1) if m else None
    except Exception:
        kid = None
    servers = []
    for mod in ("jupyter_server.serverapp", "notebook.notebookapp"):
        try:
            servers += list(__import__(mod, fromlist=["x"]).list_running_servers())
        except Exception:
            pass
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never route localhost via a proxy
    for s in servers:
        hdr = {"Authorization": f"token {s['token']}"} if s.get("token") else {}
        try:
            sess = json.load(opener.open(urllib.request.Request(s["url"] + "api/sessions", headers=hdr), timeout=10))
            mine = [x for x in sess if x["kernel"]["id"] == kid] or \
                   [x for x in sess if x.get("type", "notebook") == "notebook" and "covid_ct_audit" in x["path"]]
            if not mine:
                continue
            path = mine[0]["path"]
            url = s["url"] + "api/contents/" + urllib.parse.quote(path) + "?content=1&type=notebook"
            r = json.load(opener.open(urllib.request.Request(url, headers=hdr), timeout=30))
            where = f"{s.get('root_dir') or s.get('notebook_dir') or '?'} / {path}"
            return r["content"], f"{where} (server copy, last saved {r.get('last_modified')})"
        except Exception as e:
            print("  server lookup failed:", type(e).__name__, str(e)[:100])
    return None


def _from_disk():
    pats = [os.path.join(os.getcwd(), "covid_ct_audit*.ipynb"), os.path.join(os.getcwd(), "..", "covid_ct_audit*.ipynb"),
            os.path.join(os.path.expanduser("~"), "**", "covid_ct_audit*.ipynb"), "E:/covid_ct_audit*.ipynb", "E:/covid_audit/covid_ct_audit*.ipynb"]
    hits = {os.path.abspath(p) for pat in pats for p in glob.glob(pat, recursive=True) if "checkpoint" not in p}
    if not hits:
        return None
    p = max(hits, key=os.path.getmtime)
    return json.load(open(p, encoding="utf-8")), f"{p} (disk copy, saved {time.ctime(os.path.getmtime(p))})"


def export_outputs(nb, out=EXPORT_DIR):
    img_dir = out / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    lines, nimg, empty = [], 0, []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.startswith(MARK):
            continue
        lines.append(f"\n########## CELL {i}: {src.split(chr(10))[0][:90]} ##########")
        if not c.get("outputs"):
            lines.append("(no output)"); empty.append(i)
        for k, o in enumerate(c.get("outputs", [])):
            t = o.get("output_type")
            if t == "stream":
                lines.append("".join(o["text"]))
            elif t in ("execute_result", "display_data"):
                d = o.get("data", {})
                if "text/plain" in d:
                    lines.append("".join(d["text/plain"]))
                if "image/png" in d:
                    nimg += 1
                    fp = img_dir / f"cell{i}_fig{k}.png"
                    fp.write_bytes(base64.b64decode(d["image/png"]))
                    lines.append(f"[figure saved: {fp.name}]")
            elif t == "error":
                lines.append("ERROR: " + o.get("ename", "") + ": " + o.get("evalue", ""))
    txt = ansi.sub("", "\n".join(lines))
    (out / "all_outputs.txt").write_text(txt, encoding="utf-8")
    with zipfile.ZipFile(out / "figures.zip", "w") as z:
        for f in sorted(img_dir.glob("*.png")):
            z.write(f, f.name)
    return txt, nimg, empty


got = _from_server() or _from_disk()
if got is None:
    print("Could not find the notebook. Use File > Download > .ipynb and send that file instead.")
else:
    nb_, desc = got
    txt_, nimg_, empty_ = export_outputs(nb_)
    print("Read:", desc)
    print(f"Wrote {len(txt_):,} characters and {nimg_} figures to {EXPORT_DIR.resolve()}")
    print("  all_outputs.txt  and  figures.zip   <- attach these two")
    if empty_:
        print("Code cells with NO saved output (not run yet, or you forgot Ctrl+S):", empty_)
