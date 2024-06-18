const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const unzipper = require('unzipper');
const serverless = require('serverless-http');

const app = express();
const upload = multer({ dest: 'uploads/' });

// Ruta para la página de carga de imágenes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'templates', 'upload.html'));
});

// Descargar y cargar el modelo directamente desde Google Drive sin descomprimirlo
const modelURL = "https://drive.google.com/uc?export=download&id=1KDEGqPBobms06gXO7o6VTS1pF---LXG0";
let model;

const loadModel = async () => {
  try {
    console.log(`Loading model from ${modelURL}`);
    const response = await axios({
      url: modelURL,
      method: 'GET',
      responseType: 'arraybuffer',
    });

    const buffer = Buffer.from(response.data, 'binary');
    const zip = await unzipper.Open.buffer(buffer);

    // Cargar el modelo directamente desde el buffer zip
    const modelFile = zip.files.find(d => d.path === 'best_model_basic/model.json');
    const modelDir = path.join(__dirname, 'models');
    await modelFile.buffer()
      .then(data => {
        fs.writeFileSync(path.join(modelDir, 'model.json'), data);
        console.log('Model JSON saved.');
      });

    model = await tf.loadLayersModel(`file://${modelDir}/model.json`);
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Error loading model:', error);
  }
};

app.post('/predict', upload.single('file'), async (req, res) => {
  try {
    const file = req.file;
    if (!file) {
      return res.status(400).json({ error: 'No file part' });
    }

    const filePath = path.join(__dirname, file.path);
    const imageBuffer = fs.readFileSync(filePath);
    const tensor = tf.node.decodeImage(imageBuffer)
      .resizeNearestNeighbor([150, 150])
      .mean(2)
      .expandDims(0)
      .expandDims(-1)
      .div(255.0);

    const predictions = model.predict(tensor);
    const predArray = predictions.arraySync();
    const predClass = predArray[0].indexOf(Math.max(...predArray[0]));
    const predLabel = predClass === 1 ? "Pneumonia" : "Normal";

    res.send(`
      <html>
        <body>
          <h1>Prediction: ${predLabel}</h1>
          <p>Confidence: ${Math.max(...predArray[0])}</p>
        </body>
      </html>
    `);
  } catch (error) {
    console.error('Error during prediction:', error);
    res.status(500).json({ error: 'An error occurred while processing the image.' });
  }
});

app.listen(8080, async () => {
  await loadModel();
  console.log('Server is running on port 8080');
});

// Netlify Functions handler
module.exports.handler = serverless(app);

