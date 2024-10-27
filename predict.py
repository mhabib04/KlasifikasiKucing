import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog

# Memuat model yang telah disimpan
model = tf.keras.models.load_model('kucing_classifier.h5', compile=False)

# Kompilasi model dengan metrik yang diinginkan
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def predict_image():
    # Membuka dialog file untuk memilih gambar
    root = Tk()
    root.withdraw()  # Menyembunyikan jendela utama Tkinter
    img_path = filedialog.askopenfilename(title='Pilih Gambar untuk Prediksi',
                                          filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if img_path:  # Pastikan file dipilih
        # Memuat dan memproses gambar
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi gambar
        prediction = model.predict(img_array)
        if prediction[0][0] > 0.5:
            print("Prediksi: Kucing Kampung")
        else:
            print("Prediksi: Kucing Bengal")
    else:
        print("Tidak ada file yang dipilih.")

# Menjalankan fungsi prediksi
predict_image()
