"""Download the Siena Scalp EEG Database from PhysioNet into data/siena/PNxx/.

Siena is our EXTERNAL test set: the seizure CNN is trained on CHB-MIT and
evaluated here, on different patients recorded with a different montage — an
honest test of cross-dataset generalization.

Each subject has a `Seizures-list-PNxx.txt` (annotations, clock times) plus its
EDF recordings. By default we fetch only the EDFs named in that seizure list
(every Siena recording contains a seizure), which is enough to evaluate and keeps
the download compact.

Stdlib only (urllib), matching scripts/download_chbmit.py.

Examples:
  python -m scripts.download_siena                 # all subjects, seizure EDFs
  python -m scripts.download_siena --subjects PN00,PN01
"""
import argparse
import os
import re
import sys
import urllib.request

BASE_URL = "https://physionet.org/files/siena-scalp-eeg/1.0.0"
DATA_ROOT = "data/siena"


def _get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def all_subjects():
    """Subject dir names (PN00, PN01, ...) from the dataset RECORDS file."""
    records = _get(f"{BASE_URL}/RECORDS").decode("utf-8", "replace")
    subs = []
    for line in records.splitlines():
        line = line.strip()
        if "/" in line:
            s = line.split("/")[0]
            if s not in subs:
                subs.append(s)
    return subs


def _download(url, dest):
    """Download url -> dest, skipping if a non-empty file already exists."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  = have {os.path.basename(dest)}")
        return True
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            total = 0
            while True:
                chunk = r.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
        os.replace(tmp, dest)
        print(f"  + {os.path.basename(dest)} ({total/1e6:.1f} MB)")
        return True
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"  !! failed {url}: {type(e).__name__}: {e}")
        return False


def seizure_edfs(seizure_list_text):
    """EDF filenames referenced by 'File name: PNxx-N.edf' lines (deduped, in order)."""
    names = re.findall(r"File name:\s*(\S+\.edf)", seizure_list_text, flags=re.IGNORECASE)
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); out.append(n)
    return out


def download_subject(subj):
    subj_dir = os.path.join(DATA_ROOT, subj)
    os.makedirs(subj_dir, exist_ok=True)

    listname = f"Seizures-list-{subj}.txt"
    list_path = os.path.join(subj_dir, listname)
    if not _download(f"{BASE_URL}/{subj}/{listname}", list_path):
        print(f"  -- no seizure list for {subj}, skipping subject")
        return
    with open(list_path, "r", encoding="utf-8", errors="replace") as f:
        edfs = seizure_edfs(f.read())
    if not edfs:
        print(f"  -- {subj}: no EDFs listed in seizure file")
        return
    for fname in edfs:
        _download(f"{BASE_URL}/{subj}/{fname}", os.path.join(subj_dir, fname))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default="", help="comma list e.g. 'PN00,PN01' (default: all)")
    args = ap.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] or all_subjects()
    print(f"Siena subjects to fetch ({len(subjects)}): {subjects}\n")
    for subj in subjects:
        print(f"[{subj}]")
        download_subject(subj)
    print("\nDone. Files in", DATA_ROOT)


if __name__ == "__main__":
    main()
