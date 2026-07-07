import mne
import numpy as np

def load_and_window(edf_path, seizure_start=None, seizure_end=None, window_sec=5):
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    sfreq = raw.info["sfreq"]
    data = raw.get_data()  # shape: (n_channels, n_samples)

    window_size = int(window_sec * sfreq)
    n_windows = data.shape[1] // window_size

    windows = []
    labels = []

    for i in range(n_windows):
        start_sample = i * window_size
        end_sample = start_sample + window_size
        window = data[:, start_sample:end_sample]

        start_sec = start_sample / sfreq
        end_sec = end_sample / sfreq

        label = 0
        if seizure_start is not None and seizure_end is not None:
            if start_sec < seizure_end and end_sec > seizure_start:
                label = 1

        windows.append(window)
        labels.append(label)

    return np.array(windows), np.array(labels)


if __name__ == "__main__":
    windows, labels = load_and_window(
        "data/raw/chb01/chb01_03.edf",
        seizure_start=2996,
        seizure_end=3036
    )
    print("Windows shape:", windows.shape)
    print("Labels shape:", labels.shape)
    print("Number of seizure windows:", labels.sum())
    print("Number of normal windows:", (labels == 0).sum())