import tensorflow as tf
import os
import numpy as np
import time
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import psutil  # Importar psutil para monitorar a utilização da CPU
from tensorflow.keras.callbacks import TensorBoard  # Importar TensorBoard

# Definir o número de CPUs disponíveis para paralelismo
num_cpus = tf.config.threading.get_inter_op_parallelism_threads()
num_intra_threads = tf.config.threading.get_intra_op_parallelism_threads()
print(f"Número de threads inter-operacionais: {num_cpus}")
print(f"Número de threads intra-operacionais: {num_intra_threads}")

# Caso esteja utilizando GPU, configure o uso eficiente da memória
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Diretórios das imagens de treino e validação
train_dir = './database/train'
valid_dir = './database/valid'

# Definir tamanho das imagens e batch size
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Função para processar as imagens
@tf.function
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

# Função para carregar e rotular as imagens
def load_data(directory, class_names):
    file_paths = tf.data.Dataset.list_files(os.path.join(directory, '*/*'))
    labeled_data = file_paths.map(lambda x: (process_image(x), get_label(x, class_names)),
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)  # Paralelização com AUTOTUNE
    labeled_data = labeled_data.ignore_errors()  # Ignorar erros no processamento
    return labeled_data

# Obter nomes das classes a partir dos diretórios
class_names = np.array(sorted(os.listdir(train_dir)))

# Criar datasets para treino e validação com paralelismo, cache e prefetch
train_dataset = load_data(train_dir, class_names)
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

valid_dataset = load_data(valid_dir, class_names)
valid_dataset = valid_dataset.batch(BATCH_SIZE).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Criar o modelo CNN avançado (adaptado do segundo exemplo)
model = keras.models.Sequential([
    keras.layers.Conv2D(4, kernel_size=(4, 4), activation='relu', padding='same', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(),
    keras.layers.BatchNormalization(center=True, scale=True),
    keras.layers.Dropout(0.5),

    keras.layers.Conv2D(8, kernel_size=(8, 8), activation='relu', padding='same'),
    keras.layers.MaxPooling2D(),
    keras.layers.BatchNormalization(center=True, scale=True),
    keras.layers.Dropout(0.5),

    keras.layers.Conv2D(16, kernel_size=(8, 8), activation='relu', padding='same'),
    keras.layers.MaxPooling2D(),
    keras.layers.BatchNormalization(center=True, scale=True),
    keras.layers.Dropout(0.5),

    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(center=True, scale=True),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.BatchNormalization(center=True, scale=True),
    keras.layers.Dense(units=len(class_names), activation='softmax')  # Saída ajustada para o número de classes
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Resumo do modelo
model.summary()

# Criar um diretório para os logs do TensorBoard
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)

# Função para exibir a utilização da CPU
def print_cpu_usage():
    print(f"Uso da CPU: {psutil.cpu_percent(interval=1)}%")
    print(f"Número de CPUs lógicas: {psutil.cpu_count(logical=True)}")
    print(f"Número de CPUs físicas: {psutil.cpu_count(logical=False)}")

# Exibir uso da CPU antes de treinar
print_cpu_usage()

# Registrar o tempo de início
start_time = time.time()

# Treinar o modelo com o callback do TensorBoard
epochs = 5
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=valid_dataset,
    callbacks=[tensorboard]  # Adicionando o callback do TensorBoard
)

# Registrar o tempo de término
end_time = time.time()
print(f"Tempo total de treinamento: {end_time - start_time:.2f} segundos")

# Exibir uso da CPU após o treinamento
print_cpu_usage()

# Plotar o gráfico de perda (loss)
plt.plot(range(1, epochs+1), history.history['loss'], 'r', label='Perda de Treinamento')
plt.plot(range(1, epochs+1), history.history['val_loss'], 'b', label='Perda de Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Plotar o gráfico de acurácia
plt.plot(range(1, epochs+1), history.history['accuracy'], 'g', label='Acurácia de Treinamento')
plt.plot(range(1, epochs+1), history.history['val_accuracy'], 'm', label='Acurácia de Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

# Função para prever a classe de uma imagem
def predict_image(image_path, model, class_names):
    img = process_image(image_path)  # Processar a imagem
    img = tf.expand_dims(img, axis=0)  # Adicionar batch dimension
    prediction = model.predict(img)  # Prever a classe
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

# Previsões e matriz de confusão
y_true = []
y_pred = []

for image_path in test_images:
    true_class = os.path.basename(os.path.dirname(image_path))  # Classe verdadeira da pasta
    predicted_class = predict_image(image_path, model, class_names)
    y_true.append(class_names.tolist().index(true_class))  # Transformar em índice
    y_pred.append(class_names.tolist().index(predicted_class))

cf_matrix = confusion_matrix(y_true, y_pred)

# Visualizar a matriz de confusão
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)

sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.show()
