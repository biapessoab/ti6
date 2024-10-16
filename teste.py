import tensorflow as tf
import os
import numpy as np

# Diretórios das imagens de treino e validação
train_dir = './database/train'
valid_dir = './database/valid'

# Definir tamanho das imagens e batch size
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Função para processar as imagens
def process_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalização
    return img

# Função para mapear rótulos às imagens
def get_label(file_path, class_names):
    parts = tf.strings.split(file_path, os.path.sep)
    return tf.cast(parts[-2] == class_names, tf.float32)

# Função para carregar e rotular as imagens com paralelismo
def load_data(directory, class_names):
    file_paths = tf.data.Dataset.list_files(os.path.join(directory, '*/*'))
    labeled_data = file_paths.map(lambda x: (process_image(x), get_label(x, class_names)),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)  # Paralelização
    return labeled_data

# Obter nomes das classes a partir dos diretórios
class_names = np.array(sorted(os.listdir(train_dir)))

# Criar datasets para treino e validação com paralelismo e prefetch
train_dataset = load_data(train_dir, class_names)
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)  # Prefetching

valid_dataset = load_data(valid_dir, class_names)
valid_dataset = valid_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)  # Prefetching

# Criar o modelo CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(
    train_dataset,
    epochs=10,
    validation_data=valid_dataset
)

# Função para prever a classe de uma imagem
def predict_image(image_path, model, class_names):
    img = process_image(image_path)
    img = tf.expand_dims(img, axis=0)  # Adicionar batch dimension
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0])
    return class_names[predicted_class]

# Lista de imagens de teste
test_images = ['./teste1.jpeg', './teste2.jpeg', './teste3.jpeg']

# Prever a classe para cada imagem de teste e salvar em um arquivo
with open('predicted_classes.txt', 'w') as f:
    classes = []
    for image_path in test_images:
        predicted_class = predict_image(image_path, model, class_names)
        classes.append(predicted_class)
    # Escrever classes separadas por vírgulas
    f.write(', '.join(classes))

print("Classes previstas salvas em 'predicted_classes.txt'")