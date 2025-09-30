import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# --- 1. SETUP ---

# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = 'D:/test/CAPSTONE GITAM/pneumonia_dataset_kaggle/chest_xray/chest_xray'
MODEL_PATH = 'best_pneumonia_finetuned.keras'

# --- 2. LOAD THE TEST DATA ---

# Create a data generator for the test set (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load test data from directory
test_generator = test_datagen.flow_from_directory(
    f'{DATA_DIR}/test',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # Important: Do not shuffle test data
)

print("Test data loaded successfully!")

# --- 3. LOAD THE TRAINED MODEL ---

print(f"\nLoading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# --- 4. EVALUATE THE MODEL ---

print("\nEvaluating model on the test set...")
loss, accuracy = model.evaluate(test_generator)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# --- 5. GENERATE A CLASSIFICATION REPORT AND CONFUSION MATRIX ---

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Get predictions for the whole test set
predictions = model.predict(test_generator)
# Convert probabilities to class labels (0 or 1)
predicted_classes = (predictions > 0.5).astype(int)

# Get true labels
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Print Classification Report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Print Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()