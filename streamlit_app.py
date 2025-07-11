import streamlit as st
import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from tensorflow.keras.models import load_model
from scripts.preprocess_and_label import segment_raw_data

# Set Streamlit page config
st.set_page_config(page_title="NeuroSense", layout="wide")

# Load trained model
model = load_model("models/seizure_model.keras")

st.title("NeuroSense - Real-Time Seizure Prediction Dashboard")
st.markdown("""
Welcome to **NeuroSense**, an AI-powered seizure prediction system for EEG data analysis.
Upload an `.edf` EEG file below to detect preictal or ictal activity in real time.
""")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an EEG .edf file", type="edf")

if uploaded_file:
    st.success("✅ File uploaded successfully! Processing")

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Read EEG using MNE
    raw = mne.io.read_raw_edf(tmp_file_path, preload=True)
    raw.pick_types(eeg=True)
    raw.filter(0.5, 40)

    # Segment data into 10s windows
    epochs = segment_raw_data(raw, duration=10.0, overlap=0.0)
    X = epochs.get_data()  # (segments, channels, samples)
    X = X.transpose(0, 2, 1)  # shape corrected for LSTM model
    probs = model.predict(X).flatten()

    # Model predictions
    probs = model.predict(X).flatten()
    y_pred = (probs > 0.5).astype(int)

    # Display results
    st.subheader("🔍 Prediction Results")
    label_map = {0: "Interictal", 1: "Seizure (Preictal/Ictal)"}
    for i, (label, prob) in enumerate(zip(y_pred, probs)):
        st.markdown(f"**Segment {i + 1}: {label_map[label]}** (Risk Score: `{prob:.2f}`)")

    # Bar chart
    st.subheader("📊 Seizure Risk per Segment")
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    ax1.bar(np.arange(len(probs)) + 1, probs, 
            color=["crimson" if p > 0.5 else "steelblue" for p in probs])
    ax1.axhline(0.5, color='gray', linestyle='--', label='Threshold = 0.5')
    ax1.set_xlabel("Segment Number")
    ax1.set_ylabel("Seizure Risk Probability")
    ax1.set_title("Segment-wise Seizure Risk")
    ax1.legend()
    st.pyplot(fig1)

    # EEG preview (first channel)
    st.subheader("📈 EEG Signal Preview (First Channel)")
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(raw.times[:1000], raw.get_data()[0, :1000])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude (μV)")
    ax2.set_title("EEG Signal - First Channel")
    st.pyplot(fig2)

    # Heatmap
    st.subheader("🌡️ Seizure Risk Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 1.5))
    sns.heatmap([probs], cmap="coolwarm", cbar=True,
                xticklabels=np.arange(1, len(probs)+1), ax=ax3)
    ax3.set_xlabel("Segment Number")
    ax3.set_yticks([])
    st.pyplot(fig3)

    # SHAP placeholder
    st.subheader("🧠 SHAP Explanation (Coming Soon)")
    st.info("Model explainability using SHAP will be added in future updates.")
