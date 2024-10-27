import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

# Memuat model yang telah disimpan
model = tf.keras.models.load_model('kucing_classifier.h5', compile=False)

# Kompilasi ulang model dengan metrik yang diinginkan
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Pastikan model dipanggil setidaknya sekali sebelum digunakan dalam Model
dummy_input = np.zeros((1, 128, 128, 3))  # Membuat input dummy untuk inisialisasi
model.predict(dummy_input)  # Memanggil model untuk inisialisasi input

# Preprocessing data
train_dir = 'dataset/train'
train_datagen = ImageDataGenerator(rescale=1.0/255)  # Normalisasi nilai pixel (0-1)

# Membaca gambar dari direktori dan melakukan augmentasi data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # Mengubah ukuran gambar menjadi 128x128
    batch_size=32,
    class_mode='binary'  # Klasifikasi biner (dua kelas)
)

# Pilih beberapa gambar dari batch pertama untuk visualisasi
num_images = 5
sample_images, _ = next(train_generator)  # Mengambil batch pertama
sample_indices = random.sample(range(len(sample_images)), num_images)  # Memilih indeks gambar secara acak

# Menampilkan gambar asli
plt.figure(figsize=(15, 6))
for i, idx in enumerate(sample_indices):
    plt.subplot(2, num_images, i + 1)
    plt.imshow(sample_images[idx])
    plt.title("Gambar Asli")  # Menampilkan judul gambar asli
    plt.axis('off')

# Membuat model aktivasi untuk layer konvolusi
layer_outputs = [layer.output for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
activation_model = Model(inputs=model.inputs, outputs=layer_outputs)  # Model untuk mendapatkan output dari lapisan konvolusi

# Visualisasi fitur dari layer konvolusi pertama
plt.figure(figsize=(15, 6))  # Ukuran figure baru untuk fitur
for i, idx in enumerate(sample_indices):
    sample_image = np.expand_dims(sample_images[idx], axis=0)  # Menambahkan dimensi untuk batch
    activations = activation_model.predict(sample_image)  # Mendapatkan aktivasi dari lapisan konvolusi

    first_layer_activation = activations[0]  # Ambil output dari lapisan konvolusi pertama
    num_filters = min(8, first_layer_activation.shape[-1])  # Menampilkan 8 fitur pertama

    # Menampilkan fitur
    for j in range(num_filters):
        plt.subplot(len(sample_indices), num_filters, i * num_filters + j + 1)
        plt.imshow(first_layer_activation[0, :, :, j], cmap='viridis')  # Menampilkan fitur dalam bentuk heatmap
        plt.axis('off')
        if j == 0:
            plt.title(f"Gambar {i + 1}")  # Judul untuk gambar

plt.suptitle("Visualisasi Fitur dari Layer Konvolusi Pertama")  # Judul untuk visualisasi
plt.tight_layout()
plt.show()
