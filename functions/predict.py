from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Cargar el modelo entrenado
model = load_model('best_model_improved.keras')

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
        filepath = os.path.join("/tmp", file.filename)
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

