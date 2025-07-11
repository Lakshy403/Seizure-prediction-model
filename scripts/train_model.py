import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# === Load Data ===
X = np.load("data/features_X.npy")
y = np.load("data/labels_y.npy")

# Convert to binary: 1 = seizure (preictal + ictal), 0 = interictal
y = np.where(y > 0, 1, 0)

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM expects [samples, time, features]
X_train = X_train.transpose(0, 2, 1)  # (samples, 2560, 23)
X_test = X_test.transpose(0, 2, 1)

# === Build LSTM Model ===
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(2560, 23)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === Train ===
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# ✅ Ensure models directory exists
os.makedirs("models", exist_ok=True)

# ✅ Save using modern format
model.save("models/seizure_model.keras")
print("✅ Model saved to models/seizure_model.keras")

# === Evaluate ===
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Plot Accuracy ===
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Training Accuracy')
plt.show()

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
