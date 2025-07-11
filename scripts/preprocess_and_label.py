import os
import numpy as np
import mne

# === CONFIG ===
DATA_DIR = "C:/Users/Laksh/CODES/seizure_prediction_project/data"
PATIENTS = ["chb01", "chb02"]
SEIZURE_ANNOTATIONS = {
    "chb01_03.edf": [(2996, 3036)],
    "chb01_04.edf": [(1467, 1494)],
    "chb02_16.edf": [(1120, 1180)]
}
WINDOW_SIZE = 10   # seconds
STEP_SIZE = 5      # seconds
SFREQ = 256        # sampling frequency

# 🧩 NEW: Segment EEG data (for Streamlit and others)
def segment_raw_data(raw, duration=10.0, overlap=0.0):
    sfreq = int(raw.info['sfreq'])
    data = raw.get_data()
    n_channels, n_samples = data.shape

    window_samples = int(duration * sfreq)
    step_samples = int(window_samples * (1 - overlap))

    segments = []
    for start in range(0, n_samples - window_samples + 1, step_samples):
        segment = data[:, start:start + window_samples]
        if segment.shape[1] == window_samples:
            segments.append(segment)

    # Convert list to 3D array: (n_segments, n_channels, n_samples)
    return mne.EpochsArray(np.array(segments), info=raw.info)


def preprocess_eeg(edf_path, window_size, step_size, sfreq):
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    raw.filter(0.5, 40, fir_design='firwin')
    raw.resample(sfreq)

    data = raw.get_data()
    window_samples = int(window_size * sfreq)
    step_samples = int(step_size * sfreq)

    X = []
    start_times = []

    for start in range(0, data.shape[1] - window_samples, step_samples):
        end = start + window_samples
        segment = data[:, start:end]
        if segment.shape[1] == window_samples:
            X.append(segment)
            start_times.append(start / sfreq)  # store window start time in seconds

    return np.array(X), start_times

def label_windows(start_times, seizure_events, preictal_duration=300):
    y = []
    for start_time in start_times:
        label = 0  # default: interictal
        for seizure_start, seizure_end in seizure_events:
            if seizure_start <= start_time <= seizure_end:
                label = 2  # ictal
                break
            elif (seizure_start - preictal_duration) <= start_time < seizure_start:
                label = 1  # preictal
                break
        y.append(label)
    return np.array(y)

def process_all():
    all_X, all_y = [], []
    for patient in PATIENTS:
        patient_path = os.path.join(DATA_DIR, patient)
        for filename in os.listdir(patient_path):
            if filename.endswith(".edf"):
                edf_path = os.path.join(patient_path, filename)
                print(f"Processing {filename}...")
                X, start_times = preprocess_eeg(edf_path, WINDOW_SIZE, STEP_SIZE, SFREQ)

                # Get seizure events
                seizure_events = SEIZURE_ANNOTATIONS.get(filename, [])
                y = label_windows(start_times, seizure_events)

                all_X.append(X)
                all_y.append(y)

    # Concatenate across files
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    print("Final shape:", X_all.shape, y_all.shape)
    return X_all, y_all

if __name__ == "__main__":
    X, y = process_all()
    np.save("data/features_X.npy", X)
    np.save("data/labels_y.npy", y)
