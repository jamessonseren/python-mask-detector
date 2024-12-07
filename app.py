from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import cv2 as cv
import base64
from io import BytesIO
import joblib
import os

# Inicializar o Flask
app = Flask(__name__)
CORS(app)

# Configuração do modelo TFLite
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Carregar os nomes das classes para o modelo TFLite
with open("./labels.txt", "r") as file:
    class_names = [line.strip() for line in file.readlines()]

# Configuração do modelo Haar/PCA-KNN
MODEL_PATH = "modelos"
PCA_MODEL_FILE = os.path.join(MODEL_PATH, "pca_model.pkl")
KNN_MODEL_FILE = os.path.join(MODEL_PATH, "knn_model.pkl")

# Verificar se os modelos Haar/PCA-KNN existem
if os.path.exists(PCA_MODEL_FILE) and os.path.exists(KNN_MODEL_FILE):
    pca = joblib.load(PCA_MODEL_FILE)
    knn = joblib.load(KNN_MODEL_FILE)
else:
    raise FileNotFoundError("Modelos Haar/PCA-KNN não encontrados. Treine-os antes de executar a aplicação.")

# Processamento de imagem para o modelo TFLite
def process_image_tflite(image):
    image = ImageOps.fit(image, (224, 224), Image.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1
    return np.expand_dims(normalized_image_array, axis=0)

# Processamento de imagem para o modelo Haar/PCA-KNN
def process_image_haar(image):
    gray = cv.cvtColor(np.array(image), cv.COLOR_RGB2GRAY)
    gray_resized = cv.resize(gray, (160, 160)).flatten()  # Redimensionar e achatar
    return pca.transform([gray_resized])

@app.route('/predict-tm', methods=['POST'])
def predict_tm():
    try:
        # Verificar se a imagem foi enviada no formato esperado
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'Campo "image" não encontrado na requisição'}), 400

        # Decodificar a imagem Base64
        image_data = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Realizar a predição com o modelo TFLite
        input_data_tflite = process_image_tflite(image)
        interpreter.set_tensor(input_details[0]['index'], input_data_tflite)
        interpreter.invoke()
        output_data_tflite = interpreter.get_tensor(output_details[0]['index'])
        tflite_index = np.argmax(output_data_tflite)
        tflite_confidence = output_data_tflite[0][tflite_index] * 100
        tflite_class = class_names[tflite_index]

        # Realizar a predição com o modelo Haar/PCA-KNN
        input_data_haar = process_image_haar(image)
        haar_prediction = knn.predict(input_data_haar)[0]
        haar_confidence = 90  # Simulando confiança fixa; ajuste para calcular corretamente
        haar_class = "Com máscara" if haar_prediction == 1 else "Sem máscara"

        # Retornar os resultados
        return jsonify({
            'tm': {
                'class': tflite_class,
                'confidence': f"{tflite_confidence:.2f}%"
            },
            'haar': {
                'class': haar_class,
                'confidence': f"{haar_confidence:.2f}%"
            }
        })
    except Exception as e:
        return jsonify({'error': f"Erro ao processar a imagem: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
