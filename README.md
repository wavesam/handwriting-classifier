# Handwriting Classifier

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify handwritten images. The model is trained to distinguish between three categories: 'sam', 'dad', and 'sis'.

## Features

*   **Image Loading and Preprocessing:** Loads `.png` images from a specified directory, converts them to grayscale, resizes them to 28x28 pixels, and normalizes pixel values between 0 and 1.
*   **CNN Model:** A sequential CNN model with convolutional layers, pooling layers, and dense layers for classification.
*   **Label Encoding:** Utilizes `LabelEncoder` and `to_categorical` for efficient handling of categorical labels.
*   **Model Training:** Trains the CNN model using the prepared dataset and monitors accuracy.
*   **Visualization:** Plots the training accuracy over epochs using Matplotlib.
*   **Model and Encoder Saving:** Saves the trained Keras model (`.keras`) and the fitted `LabelEncoder` (`.joblib`) for future use.

## Requirements

*   Python 3.x
*   OpenCV (`cv2`): For image loading and manipulation.
*   NumPy (`numpy`): For numerical operations.
*   TensorFlow/Keras (`tensorflow`): For building and training the neural network.
*   Scikit-learn (`sklearn`): For label encoding.
*   Matplotlib (`matplotlib`): For plotting training history.
*   Joblib (`joblib`): For saving the label encoder.

You can install these dependencies using pip:

```bash
pip install opencv-python numpy tensorflow scikit-learn matplotlib joblib
```

## Project Structure

```
.
├── data/
│   └── train/
│       ├── sam_*.png
│       ├── dad_*.png
│       └── sis_*.png
├── train.py             # The main script for training the model
├── handwriting_classifier.keras # Saved trained model
└── label_encoder.joblib # Saved fitted label encoder
```

**Note:** You will need to create the `data/train` directory and populate it with your `.png` images, ensuring the filenames follow the `label_number.png` format (e.g., `sam_001.png`, `dad_005.png`).

## Usage

1.  **Prepare your dataset:**
    *   Create a directory named `data`.
    *   Inside `data`, create another directory named `train`.
    *   Place your handwritten images (e.g., `sam_1.png`, `dad_2.png`, `sis_3.png`) within the `data/train` directory. Ensure each image filename starts with the corresponding label ('sam', 'dad', or 'sis') followed by an underscore.

2.  **Run the training script:**
    Execute the `main.py` script from your terminal:

    ```bash
    python main.py
    ```

    The script will:
    *   Load and preprocess images from `./data/train`.
    *   Define and compile a CNN model.
    *   Train the model for a specified number of epochs (currently 10).
    *   Display a plot of the training accuracy.
    *   Save the trained model as `handwriting_classifier.keras`.
    *   Save the fitted `LabelEncoder` as `label_encoder.joblib`.

## How to Use the Trained Model (Example)

After training, you can load the model and encoder to make predictions on new images. Here's a conceptual example of how you might do this in a separate script (`predict.py`):

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load

# Load the trained model
model = load_model('handwriting_classifier.keras')

# Load the label encoder
label_encoder = load('label_encoder.joblib')

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape((-1, 28, 28, 1))
    return img

def predict_label(image_path):
    processed_img = preprocess_image(image_path)
    prediction = model.predict(processed_img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    # To get the original label, you'd need to invert the encoding
    # However, since label_encoder.inverse_transform expects a 1D array of encoded labels,
    # and we have a single prediction, we might need to adjust this part based on how
    # label_encoder was fit. A common way is to reconstruct the classes.
    # For simplicity, let's assume we know the classes are ['dad', 'sam', 'sis'] in order
    # after potential reordering by fit_transform and we can map based on that.
    # A robust way is to store the classes along with the encoder or reconstruct them.

    # Example reconstruction if classes were fit in a specific order ['dad', 'sam', 'sis']
    # If label_encoder.classes_ is accessible and contains the correct order:
    if hasattr(label_encoder, 'classes_'):
        predicted_label = label_encoder.classes_[predicted_class_index]
    else:
        # Fallback or an example if manually known order was used (less robust)
        # This part might need adjustment based on your actual label fitting order
        classes = ['dad', 'sam', 'sis'] # Assuming this order for demonstration
        predicted_label = classes[predicted_class_index]


    return predicted_label, prediction[0]

# Example usage:
if __name__ == "__main__":
    # Replace 'path/to/your/new/image.png' with the actual path to an image
    # For example: './data/test/sam_test_1.png'
    new_image_path = './data/test/sample_image.png' # Create this path and file for testing
    predicted_label, probabilities = predict_label(new_image_path)
    print(f"The predicted label for {new_image_path} is: {predicted_label}")
    print(f"Probabilities: {probabilities}")
```

**Note on `test.py`:**

*   You'll need a `./data/test` directory with images to test.
*   The `predict_label` function includes comments about retrieving the original label from the `LabelEncoder`. The most robust way is if `label_encoder.classes_` is available and correctly ordered. If not, you might need to manually define the original class order if you know it.

---

This README provides a comprehensive overview of the project, its setup, usage, and how to extend it for predictions.
