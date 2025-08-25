import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model and label encoder
model = load_model('handwriting_classifier.keras')
label_encoder = joblib.load('label_encoder.joblib')

def preprocess_image(image_path):
    # Preprocess image identical to training
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1) # Add channel dimension
    return img

def predict_image(image_path):
    # Make prediction on single image
    processed_img = preprocess_image(image_path)
    prediction = model.predict(processed_img)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
    confidence = np.max(prediction)
    return predicted_class[0], float(confidence)

if __name__ == "__main__":
    # Example usage
    image_path = "./data/test/test.png"  # Change to your test image
    class_name, confidence = predict_image(image_path)
    print(f"Predicted: {class_name} (Confidence: {confidence:.2%})")