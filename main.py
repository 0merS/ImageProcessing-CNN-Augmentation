# Gerekli kütüphaneleri import etmek
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Veri Setini Kullanmak

# Resimlerin bulunduğu ana dizini belirlemek
image_dir = "C:\\Users\\omers\\Desktop\\dataset\\JPEGImages"

# 10 sınıfın adlarını girmek
classes = ['collie', 'dolphin', 'elephant', 'fox', 'moose', 'rabbit', 'sheep', 'squirrel', 'giant panda', 'polar bear']

# Resimleri ve etiketlerini yüklemek
def load_images(image_dir, classes, image_size=(128, 128)):
    images = []
    labels = []
    
    # Her sınıf için resimleri almak
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(image_dir, class_name)
        for image_name in os.listdir(class_dir)[:650]:  # Her sınıftan 650 resim
            image_path = os.path.join(class_dir, image_name)
            image = Image.open(image_path)
            image = image.resize(image_size)  # Resimleri aynı boyutta yap
            image = np.array(image) / 255.0  # Resimleri normalize et
            images.append(image)
            labels.append(label)  # Etiketleri ekle
    
    return np.array(images), np.array(labels)

# Veriyi yüklemek
X, y = load_images(image_dir, classes)

# Eğitimi ve test setlerini ayırmak
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Eğitim ve test boyutlarını yazdırmak
print(f"Eğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")

# 2. Veri Artırma (Augmentation)

# Eğitim verisi için veri artırma (augmentation) kullanmak
datagen = ImageDataGenerator(
    rotation_range=30,    # Resimleri döndürme
    width_shift_range=0.2,  # Yatay kaydırma
    height_shift_range=0.2, # Dikey kaydırma
    shear_range=0.2,      # Kesme dönüşümü
    zoom_range=0.2,       # Zoom
    horizontal_flip=True, # Yatay çevirme
    fill_mode='nearest'   # Boş bölgeleri doldurma
)

# Eğitim verisi için augmentation uygulamak
datagen.fit(X_train)

# 3. CNN Modelinin Tasarlanması

# Modeli oluşturmak
model = models.Sequential()

# Konvolüsyonel katmanlar
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Fully connected (dense) katman
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout ile overfitting'i engelle
model.add(layers.Dense(len(classes), activation='softmax'))  # Son katman, 10 sınıf

# Modeli derle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Modeli Eğitme

# Modeli eğit
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

# Modeli test et
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test seti doğruluğu: {test_acc * 100:.2f}%")

# 5. Resimlerin Farklı Işıklar ile Manipüle Edilmesi

# Resimleri farklı ışık koşullarında manipüle etme
def get_manipulated_images(image):
    # Resmi farklı ışık koşulları altında manipüle etme
    image_brightness = cv2.convertScaleAbs(image, alpha=1.2, beta=50)  # Parlaklık artırma
    return image_brightness

# 6. Modeli Manipüle Edilmiş Test Seti ile Denemek

# Manipüle edilmiş test seti ile modeli test et
X_test_manipulated = np.array([get_manipulated_images(img) for img in X_test])
test_loss, test_acc = model.evaluate(X_test_manipulated, y_test)
print(f"Manipüle edilmiş test seti doğruluğu: {test_acc * 100:.2f}%")

# 7. Renk Sabitliği Uygulama (Gray World Algoritması)

# Gray World algoritmasını kullanarak renk sabitliği
def get_wb_images(image):
    avg_r = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_b = np.mean(image[:, :, 2])
    
    avg = (avg_r + avg_g + avg_b) / 3
    image[:, :, 0] = image[:, :, 0] * (avg / avg_r)  # Red channel
    image[:, :, 1] = image[:, :, 1] * (avg / avg_g)  # Green channel
    image[:, :, 2] = image[:, :, 2] * (avg / avg_b)  # Blue channel
    
    return np.clip(image, 0, 255).astype(np.uint8)

# 8. Modeli Renk Sabitliği Uygulanmış Test Seti ile Denemek

# Renk sabitliği uygulanmış test seti ile testi yap
X_test_wb = np.array([get_wb_images(img) for img in X_test])
test_loss, test_acc = model.evaluate(X_test_wb, y_test)
print(f"Renk sabitliği uygulanmış test seti doğruluğu: {test_acc * 100:.2f}%")

# 9. Başarıların Karşılaştırılması

# Sonuçları karşılaştırma
print(f"Eğitim seti doğruluğu: {history.history['accuracy'][-1]*100:.2f}%")
print(f"Test seti doğruluğu: {test_acc*100:.2f}%")
