from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import requests
import zipfile

app = Flask(__name__)

# Función para descargar y descomprimir el modelo desde Google Drive
def download_and_unzip_model():
    url = "https://drive.google.com/uc?export=download&id=1r4KZYCs_Iyfz2Tml7XkmGAMEbpXSsF7n"
    local_zip_path = "model.zip"
    model_filename = "best_model_improved.keras"
    
    response = requests.get(url)
    with open(local_zip_path, 'wb') as file:
        file.write(response.content)
    
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")  # Extraer directamente al directorio actual
    
    os.remove(local_zip_path)
    print(f"Model downloaded and extracted to current directory")
    return model_filename

# Ruta para cargar el modelo
model_path = "best_model_improved.keras"
if not os.path.exists(model_path):
    print(f"{model_path} not found. Downloading and extracting model...")
    model_path = download_and_unzip_model()

# Verificar la existencia del archivo y su tamaño antes de cargarlo
if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    file_size = os.path.getsize(model_path)
    print(f"File size: {file_size} bytes")

    try:
        model = load_model(model_path)
    except ValueError as e:
        print(f"Error loading model: {e}")
        raise
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        filepath = os.path.join("uploads", file.filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(150, 150), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = model.predict(img_array)
        pred_class = np.argmax(predictions, axis=1)
        pred_label = "Pneumonia" if pred_class[0] == 1 else "Normal"

        return render_template('result.html', prediction=pred_label)

# Netlify Functions handler
def handler(event, context):
    return app(event, context)


