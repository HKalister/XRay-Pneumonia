const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const unzipper = require('unzipper');

const app = express();
const upload = multer({ dest: 'uploads/' });

// Ruta para la página de carga de imágenes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'templates', 'upload.html'));
});

// Ruta para descargar y descomprimir el modelo
const downloadAndUnzipModel = async () => {
  const url = "https://drive.google.com/uc?export=download&id=1r4KZYCs_Iyfz2Tml7XkmGAMEbpXSsF7n";
  const localZipPath = path.join(__dirname, 'model.zip');
  const modelDir = path.join(__dirname, 'models');

  const response = await axios({
    url,
    method: 'GET',
    responseType: 'stream',
  });

  const writer = fs.createWriteStream(localZipPath);
  response.data.pipe(writer);

  return new Promise((resolve, reject) => {
    writer.on('finish', async () => {
      fs.createReadStream(localZipPath)
        .pipe(unzipper.Extract({ path: modelDir }))
        .on('close', () => {
          fs.unlinkSync(localZipPath); // Eliminar el archivo zip
          resolve(modelDir);
        });
    });
    writer.on('error', reject);
  });
};

// Ruta para cargar el modelo
const modelPath = path.join(__dirname, 'models', 'best_model_improved');
let model;

const loadModel = async () => {
  if (!fs.existsSync(modelPath)) {
    console.log(`${modelPath} not found. Downloading and extracting model...`);
    await downloadAndUnzipModel();
  }

  if (fs.existsSync(modelPath)) {
    console.log(`Loading model from ${modelPath}`);
    model = await tf.loadLayersModel(`file://${modelPath}/model.json`);
    console.log('Model loaded successfully');
  } else {
    throw new Error(`Model file not found at ${modelPath}`);
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
module.exports.handler = (event, context) => {
  const serverless = require('serverless-http');
  return serverless(app)(event, context);
};
