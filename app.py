import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)  # Apply CORS to the entire app

# Load the pre-trained model (ensure the model path is correct)
model_path = 'kneeosteoarthritis_957.28.h5'  # Model path
model = load_model(model_path)

# Define the class labels (Ensure these match your model’s output)
classes = ['Healthy', 'Moderate', 'Severe']

# Image Preprocessing Function
def load_and_preprocess_image(image_bytes, target_size=(224, 224)):
    # Convert byte data to image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # Ensure image is in RGB format

    # Resize the image to the target size
    img = img.resize(target_size)

    # Convert image to array and preprocess for EfficientNet
    img_array = np.array(img).astype(np.float32)
    img_array = preprocess_input(img_array)

    # Expand dimensions to match the model's input shape (batch size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Route for image classification
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_bytes = file.read()
        img_array = load_and_preprocess_image(image_bytes)
        predictions = model.predict(img_array)[0]
        predicted_class_index = np.argmax(predictions)
        predicted_class = classes[predicted_class_index]
        confidence_score = float(predictions[predicted_class_index]) * 100  # Convert to percentage

        # Create a dictionary of class labels and their corresponding confidence scores
        confidence_scores = {classes[i]: f"{float(predictions[i]) * 100:.2f}%" for i in range(len(classes))}

        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence_score:.2f}%",  # Format as percentage
            'all_confidences': confidence_scores
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Simple route for the root URL
@app.route('/')
def index():
    return "Welcome to the Knee Osteoarthritis Prediction API!"

if __name__ == '__main__':
    app.run(port=5005)
