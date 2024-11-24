import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import requests
from PIL import Image

# Caminhos e Configurações
TRAIN_DIR = './database/train'
VALID_DIR = './database/valid'
IMG_SIZE = (150, 150)
MODEL_FOLDER = './models'
MODEL_PATH = os.path.join(MODEL_FOLDER, 'trained_model.h5')
CLASS_INDICES_PATH = os.path.join(MODEL_FOLDER, 'class_indices.json')
API_URL = "https://api.edamam.com/api/recipes/v2"
APP_ID = '03ef7e2d'
APP_KEY = 'dc7be51f1bdd4c32ca45288155a1d928'

# Inicializar o Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Verificar se o modelo já está treinado ou treinar
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo carregado com sucesso.")

    # Carregar classes salvas
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_names = json.load(f)
    else:
        raise FileNotFoundError("O arquivo class_indices.json não foi encontrado.")
else:
    # Treinar o modelo se não estiver salvo
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical'
    )
    valid_generator = datagen.flow_from_directory(
        VALID_DIR,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical'
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, validation_data=valid_generator, epochs=10)
    model.save(MODEL_PATH)
    print("Modelo treinado e salvo com sucesso.")

    # Salvar as classes em um arquivo JSON
    class_names = list(train_generator.class_indices.keys())
    with open(CLASS_INDICES_PATH, 'w') as f:
        json.dump(class_names, f)

# Função para processar e classificar uma imagem
def classify_image(image_path):
    img = Image.open(image_path).resize(IMG_SIZE)  # Usar PIL.Image para carregar a imagem
    img_array = np.array(img) / 255.0  # Normalizar os valores
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar dimensão de batch
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    return predicted_class

# Rota principal
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('file')  # Obter múltiplos arquivos
        results = []

        for file in files:
            if file:
                # Salvar arquivo carregado
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Classificar a imagem
                predicted_class = classify_image(filepath)

                # Fazer a requisição para a API de receitas
                params = {
                    'type': 'public',
                    'q': predicted_class,
                    'app_id': APP_ID,
                    'app_key': APP_KEY,
                    'to': 10
                }
                response = requests.get(API_URL, params=params)
                data = response.json()

                # Preparar resultados para cada imagem
                recipes = []
                if 'hits' in data and data['hits']:
                    for item in data['hits']:
                        recipe = item['recipe']
                        recipes.append({
                            'name': recipe['label'],
                            'url': recipe['url'],
                            'image': recipe['image']
                        })

                results.append({
                    'image': filename,
                    'predicted_class': predicted_class,
                    'recipes': recipes
                })

        return render_template('results.html', results=results)

    return render_template('index.html')

# Rodar o servidor
if __name__ == '__main__':
    app.run(debug=True)
