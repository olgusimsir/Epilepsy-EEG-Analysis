"""Download CHB-MIT subjects from PhysioNet into data/raw/chbNN/.

The raw EEG isn't committed (it's large), so this fetches it on demand. It reads
each subject's summary first, then downloads the EDFs it lists — so you can pull
ONLY the seizure-containing files (small, enough to train/evaluate) instead of the
full multi-GB recording set.

Stdlib only (urllib), matching serve_ui.py — no extra dependencies.

Examples:
  # seizure-containing files for patients 9-16 (compact, recommended)
  python -m scripts.download_chbmit --subjects chb09-chb16 --seizure-only

  # everything for two specific patients
  python -m scripts.download_chbmit --subjects chb17,chb18

  # all 24 patients, seizure files only
  python -m scripts.download_chbmit --subjects chb01-chb24 --seizure-only
"""
import argparse
import os
import sys
import urllib.request

# import parse_summary without requiring the package to be importable as `src`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.parse_summary import parse_summary  # noqa: E402

BASE_URL = "https://physionet.org/files/chbmit/1.0.0"
DATA_ROOT = "data/raw"


def expand_subjects(spec):
    """Turn 'chb09-chb16,chb20' into ['chb09','chb10',...,'chb16','chb20']."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            lo, hi = int(a.replace("chb", "")), int(b.replace("chb", ""))
            out.extend(f"chb{n:02d}" for n in range(lo, hi + 1))
        else:
            out.append(part)
    return out


def _download(url, dest):
    """Download url -> dest, skipping if a non-empty file already exists."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  = have {os.path.basename(dest)}")
        return True
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
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


def download_subject(subj, seizure_only, max_files):
    subj_dir = os.path.join(DATA_ROOT, subj)
    os.makedirs(subj_dir, exist_ok=True)

    summary_name = f"{subj}-summary.txt"
    summary_path = os.path.join(subj_dir, summary_name)
    if not _download(f"{BASE_URL}/{subj}/{summary_name}", summary_path):
        print(f"  -- no summary for {subj}, skipping subject")
        return

    info = parse_summary(summary_path)  # {filename: [(start,end), ...]}
    files = list(info.keys())
    if seizure_only:
        files = [f for f in files if info[f]]  # only files with >=1 seizure
        if not files:
            print(f"  -- {subj}: no seizure files listed; nothing to fetch")
    if max_files:
        files = files[:max_files]

    for fname in files:
        _download(f"{BASE_URL}/{subj}/{fname}", os.path.join(subj_dir, fname))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", required=True,
                    help="e.g. 'chb09-chb16' or 'chb17,chb18' or 'chb01-chb24'")
    ap.add_argument("--seizure-only", action="store_true",
                    help="download only EDFs that contain a labeled seizure (compact)")
    ap.add_argument("--max-files", type=int, default=0,
                    help="cap EDFs per subject (0 = no cap)")
    args = ap.parse_args()

    subjects = expand_subjects(args.subjects)
    print(f"Subjects to fetch ({len(subjects)}): {subjects}")
    print(f"seizure_only={args.seizure_only}  max_files={args.max_files or 'all'}\n")
    for subj in subjects:
        print(f"[{subj}]")
        download_subject(subj, args.seizure_only, args.max_files)
    print("\nDone. Verify with:  python -m src.subjects")


if __name__ == "__main__":
    main()
