import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

# Memuat dan kompilasi model
model = tf.keras.models.load_model('kucing_classifier.h5', compile=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Inisialisasi model
dummy_input = np.zeros((1, 128, 128, 3))
model.predict(dummy_input)

# Setup data generator
train_dir = 'dataset/train'
train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Definisi fitur yang mungkin terdeteksi di layer pertama
feature_descriptions = [
    "Tepi/Garis",
    "Tekstur Bulu",
    "Pola Warna",
    "Kontras",
    "Gradien Horizontal",
    "Gradien Vertikal"
]

# Mengambil sampel gambar (kurangi menjadi 3 gambar)
num_images = 3
sample_images, labels = next(train_generator)
sample_indices = random.sample(range(len(sample_images)), num_images)

# Menampilkan gambar asli
plt.figure(figsize=(20, 10))

# Plot gambar asli di baris pertama
for i, idx in enumerate(sample_indices):
    plt.subplot(2, num_images, i + 1)
    plt.imshow(sample_images[idx])
    class_label = "Bengal" if labels[idx] > 0.5 else "Kampung"
    plt.title(f"Gambar Asli\n({class_label})", pad=10)
    plt.axis('off')

# Membuat model aktivasi
layer_outputs = [layer.output for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
activation_model = Model(inputs=model.inputs, outputs=layer_outputs)

# Visualisasi fitur dengan keterangan
for i, idx in enumerate(sample_indices):
    sample_image = np.expand_dims(sample_images[idx], axis=0)
    activations = activation_model.predict(sample_image)
    
    first_layer_activation = activations[0]
    num_filters = min(6, first_layer_activation.shape[-1])  # Kurangi jumlah filter yang ditampilkan
    
    # Plot aktivasi di baris kedua
    plt.subplot(2, num_images, i + num_images + 1)
    
    # Gabungkan beberapa filter menjadi satu gambar composite
    composite_activation = np.zeros((first_layer_activation.shape[1], first_layer_activation.shape[2]))
    for j in range(num_filters):
        composite_activation += first_layer_activation[0, :, :, j]
    
    plt.imshow(composite_activation, cmap='viridis')
    plt.title("Gabungan Fitur yang Terdeteksi", pad=10)
    plt.axis('off')

plt.suptitle("Visualisasi Fitur Layer Konvolusi Pertama\nKuning = Aktivasi Tinggi, Biru = Aktivasi Rendah", 
             fontsize=14, y=1.02)

# Menambahkan colorbar
plt.subplots_adjust(right=0.92)
cbar_ax = plt.axes([0.93, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap='viridis')
plt.colorbar(sm, cax=cbar_ax)
cbar_ax.set_ylabel('Intensitas Aktivasi', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()

# Menampilkan penjelasan fitur
print("\nPenjelasan Fitur yang Terdeteksi:")
print("\nArea Kuning (Aktivasi Tinggi):")
print("- Menunjukkan area di mana fitur penting terdeteksi kuat")
print("- Biasanya menandakan area dengan pola atau tekstur khas")
print("- Untuk kucing Bengal, sering mengindikasikan area dengan pola rosette atau marbling")
print("- Untuk kucing Kampung, bisa menunjukkan pola tabby atau area dengan tekstur bulu yang khas")

print("\nArea Biru (Aktivasi Rendah):")
print("- Menunjukkan area yang kurang signifikan untuk klasifikasi")
print("- Biasanya merupakan area dengan tekstur seragam atau background")
print("- Bisa juga menandakan area yang tidak memiliki pola khusus")

print("\nFitur-fitur yang Dianalisis:")
for desc in feature_descriptions:
    print(f"\n{desc}:")
    if desc == "Tepi/Garis":
        print("- Mendeteksi batas-batas tajam dalam gambar")
        print("- Berguna untuk mengenali bentuk tubuh dan pola bulu")
    elif desc == "Tekstur Bulu":
        print("- Menangkap detail tekstur dan pola bulu")
        print("- Membantu membedakan tekstur bulu Bengal yang lebih mengkilap")
    elif desc == "Pola Warna":
        print("- Mengidentifikasi distribusi warna")
        print("- Penting untuk mengenali pola khas Bengal")
    elif desc == "Kontras":
        print("- Mendeteksi perbedaan intensitas warna")
        print("- Membantu identifikasi pola rosette Bengal")
    elif desc == "Gradien Horizontal":
        print("- Mendeteksi perubahan warna horizontal")
        print("- Berguna untuk mengenali pola stripe")
    elif desc == "Gradien Vertikal":
        print("- Mendeteksi perubahan warna vertikal")
        print("- Membantu identifikasi pola tubuh")