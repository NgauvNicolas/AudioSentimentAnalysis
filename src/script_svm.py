import os
import random
import librosa
import numpy as np
from typing import List, Tuple
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


SAMPLE_RATE = 22050  
MFCC_FEATURES = 40  
MAX_LEN = 300 
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
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

# Function to extract MFCCs
def extract_mfcc(file_path: str, max_len: int = MAX_LEN) -> np.ndarray:
    """Extract MFCC features from an audio file."""
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=MFCC_FEATURES)
    
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T.flatten()  

def load_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the RAVDESS dataset and return features and labels for selected emotions."""
    features, labels = [], []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                emotion_code = file.split("-")[2]
                if emotion_code in EMOTION_MAP:
                    mfcc = extract_mfcc(file_path)
                    features.append(mfcc)
                    labels.append(EMOTION_MAP[emotion_code])
    return np.array(features), np.array(labels)

def preprocess_labels(labels: np.ndarray) -> np.ndarray:
    """Encode emotion labels (text) into numerical format."""
    encoder = LabelEncoder()
    return encoder.fit_transform(labels), encoder

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]):
    """Plot confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('SVM Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('./plots/SVM_confusion_matrix.png')
    plt.show()

def main():
    """Main function to run the sentiment recognition pipeline."""
    set_seed(42)

    print("Loading dataset...")
    features, labels = load_dataset(DATASET_PATH)

    print("Preprocessing labels...")
    labels, encoder = preprocess_labels(labels)

    print("Standardizing features...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    print("Training SVM model...")
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = svm_model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    class_names = encoder.classes_
    plot_confusion_matrix(y_test, y_pred, class_names)

if __name__ == "__main__":
    main()