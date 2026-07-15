"""Discover which CHB-MIT subjects are present on disk.

Training scripts used to hardcode patient lists (chb01..chb08) in several places,
so adding a patient meant editing every file. Instead they now ask this module,
so dropping a new `data/raw/chbNN/` folder (summary + EDFs) is enough to include it.
"""
import glob
import os

from src.parse_summary import parse_summary

DATA_ROOT = "data/raw"


def available_subjects(data_root=DATA_ROOT, require_seizure=False):
    """CHB-MIT subject ids that have a summary AND at least one downloaded EDF.

    require_seizure=True keeps only subjects with >=1 labeled seizure in a file we
    actually downloaded. Use that for LOSO/eval *test* folds (a fold with no test
    seizures is meaningless); a seizure-free subject is still valuable as
    training-only negatives, so leave it False when building the training pool.
    Returned sorted, so the subject order is deterministic across runs.
    """
    subjects = []
    for summ in sorted(glob.glob(f"{data_root}/chb*/chb*-summary.txt")):
        subj = os.path.basename(os.path.dirname(summ))
        # `{subj}*.edf` (not `{subj}_*.edf`) so irregular names like chb17a_03.edf
        # / chb17b_63.edf are still matched to subject chb17.
        downloaded = {os.path.basename(e) for e in glob.glob(f"{data_root}/{subj}/{subj}*.edf")}
        if not downloaded:
            continue
        if require_seizure:
            info = parse_summary(summ)
            has_seizure = any(
                seizures and fname in downloaded for fname, seizures in info.items()
            )
            if not has_seizure:
                continue
        subjects.append(subj)
    return subjects


if __name__ == "__main__":
    alls = available_subjects()
    seiz = available_subjects(require_seizure=True)
    print(f"Subjects on disk ({len(alls)}): {alls}")
    print(f"  with >=1 downloaded seizure ({len(seiz)}): {seiz}")
    print(f"  training-only (no seizure file): {sorted(set(alls) - set(seiz))}")
