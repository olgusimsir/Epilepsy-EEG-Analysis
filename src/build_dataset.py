import numpy as np
from src.windowing import load_and_window
from src.parse_summary import parse_summary

def build_dataset(data_dir, summary_path, filenames, window_sec=5):
    seizure_info = parse_summary(summary_path)

    all_windows = []
    all_labels = []

    for filename in filenames:
        seizures = seizure_info.get(filename, [])

        # For now, handle only the first seizure if multiple exist
        seizure_start, seizure_end = (seizures[0] if seizures else (None, None))

        edf_path = f"{data_dir}/{filename}"
        windows, labels = load_and_window(
            edf_path,
            seizure_start=seizure_start,
            seizure_end=seizure_end,
            window_sec=window_sec
        )

        print(f"{filename}: {windows.shape[0]} windows, {labels.sum()} seizure windows")

        all_windows.append(windows)
        all_labels.append(labels)

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)

    return X, y


if __name__ == "__main__":
    X, y = build_dataset(
        data_dir="data/raw/chb01",
        summary_path="data/raw/chb01/chb01-summary.txt",
        filenames=["chb01_01.edf", "chb01_03.edf"]
    )

    print("\nFinal dataset:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Total seizure windows:", y.sum())
    print("Total normal windows:", (y == 0).sum())