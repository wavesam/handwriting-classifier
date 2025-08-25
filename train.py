import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from joblib import dump

# Set random seed for reproducibility
np.random.seed(42)

def load_images(folder_path):
    images = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            # Load image in grayscale
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
            # Resize to consistent dimensions (e.g., 28x28 like MNIST)
            img = cv2.resize(img, (28, 28))
            
            # Normalize pixel values to 0-1
            img = img / 255.0
            
            # Get label from filename (assuming format "name_number.png")
            label = filename.split("_")[0]
            
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Load your training dataset
x_train, y_train = load_images("./data/train")

# Encode labels (sam -> 0, dad -> 1, sis -> 2)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y_train)
y_train = to_categorical(encoded_labels)

# Add channel dimension (needed for CNN)
x_train = x_train.reshape((-1, 28, 28, 1))

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 output classes: sam, dad, sis
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model (using all data for training)
history = model.fit(x_train, y_train, epochs=10)

# Plot training accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Save the model
model.save('handwriting_classifier.keras')

print("Model trained successfully!")

# Save the label encoder (add this to main.py)
dump(label_encoder, 'label_encoder.joblib')

print("Model and label encoder saved successfully!")