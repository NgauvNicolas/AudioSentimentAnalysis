import os
import random
import librosa
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define constants
SAMPLE_RATE = 22050 # Sampling rate for audio files
MFCC_FEATURES = 40 # Number of MFCC features
MAX_LEN = 300 # Maximum number of timesteps
DATASET_PATH = "./data/"
EMOTION_MAP = {
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful"
}

def set_seed(seed: int = 42):
    """Set seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python's hash seed
    random.seed(seed)  # Python's random seed
    np.random.seed(seed)  # NumPy random seed
    tf.random.set_seed(seed)  # TensorFlow random seed

# Function to extract MFCCs
def extract_mfcc(file_path: str, max_len: int = MAX_LEN) -> np.ndarray:
    """Extract MFCC features from an audio file."""
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=MFCC_FEATURES)
    # Pad or truncate the MFCCs to ensure consistent input length
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T

# Function to load the dataset
def load_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the RAVDESS dataset and return features and labels for selected emotions."""
    features, labels = [], []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                # Extract the emotion label from the filename
                emotion_code = file.split("-")[2]
                if emotion_code in EMOTION_MAP:  # Filter only 5 emotions specified in the map
                    mfcc = extract_mfcc(file_path)
                    features.append(mfcc)
                    labels.append(EMOTION_MAP[emotion_code]) 
    return np.array(features), np.array(labels)

# Update preprocess_labels to handle text-based emotion labels
def preprocess_labels(labels: np.ndarray) -> np.ndarray:
    """Encode emotion labels (text) into numerical format."""
    encoder = LabelEncoder()
    return encoder.fit_transform(labels)

# Function to create the LSTM model
def create_lstm_model(input_shape: Tuple[int, int], num_classes: int) -> tf.keras.Model:
    """Create and compile the LSTM model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to plot training/validation loss and accuracy
def plot_training_history(history: tf.keras.callbacks.History):
    """Plot training and validation loss and accuracy."""
    # Extract data from history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('LSTM Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('LSTM Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./plots/LSTM_training_history.png')
    plt.show()

# Function to plot a confusion matrix
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]):
    """Plot confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('LSTM Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('./plots/LSTM_confusion_matrix.png')
    plt.show()

# Update the main function to include plotting
def main():
    """Main function to run the sentiment recognition pipeline."""
    set_seed(42)

    print("Loading dataset...")
    features, labels = load_dataset(DATASET_PATH)

    print("Preprocessing labels...")
    labels = preprocess_labels(labels)

    print("Padding features...")
    features = tf.keras.preprocessing.sequence.pad_sequences(features, maxlen=MAX_LEN, dtype='float32')

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    print("Creating model...")
    input_shape = (MAX_LEN, MFCC_FEATURES)
    num_classes = len(np.unique(labels))
    model = create_lstm_model(input_shape, num_classes)

    print("Training model...")
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=70, batch_size=32)

    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"##### Test accuracy: {test_acc:.2f} #####")

    # Plot training history
    plot_training_history(history)

    # Predict labels for the test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Plot confusion matrix
    class_names = LabelEncoder().fit(labels).classes_  # Get class names from label encoder
    plot_confusion_matrix(y_test, y_pred_classes, class_names)

if __name__ == "__main__":
    main()
