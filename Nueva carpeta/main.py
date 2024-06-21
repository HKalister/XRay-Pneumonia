import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from google.cloud import storage
import logging
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Configurar Google Cloud Storage
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'xraypredict-426805-495099b1b327.json'
bucket_name = 'bucket_241905'
model_filename = 'best_model_improved.h5'

try:
    # Crear cliente de Google Cloud Storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(model_filename)

    # Descargar el modelo de Google Cloud Storage
    model_path = os.path.join('/tmp', model_filename)
    blob.download_to_filename(model_path)

    # Cargar el modelo
    model = tf.keras.models.load_model(model_path)
    logging.info("Modelo cargado correctamente.")
except Exception as e:
    logging.error(f"Error al descargar o cargar el modelo: {e}")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            logging.error("No file part in the request")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            logging.error("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return redirect(url_for('uploaded_file', filename=filename))
        logging.error("File not allowed")
        return redirect(request.url)
    except Exception as e:
        logging.error(f"Error en upload_file: {e}")
        return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image = tf.keras.preprocessing.image.load_img(file_path, target_size=(150, 150), color_mode='grayscale')
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.reshape(image, (1, 150, 150, 1))  # Añadir dimensiones batch y canal
        image /= 255.0  # Normalizar la imagen
        
        prediction = model.predict(image)
        categories = ['NORMAL', 'PNEUMONIA']
        confidence = np.max(prediction) * 100
        predicted_class = categories[np.argmax(prediction)]
        logging.info(f"Prediction: {predicted_class}, Confidence: {confidence:.3f}%")
        return render_template('result.html', prediction=predicted_class, confidence=confidence)
    except Exception as e:
        logging.error(f"Error en uploaded_file: {e}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)


