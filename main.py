import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Parametry
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1
BATCH_SIZE = 16
EPOCHS = 50

# Ścieżki do danych
image_dir = 'path_to_images'  # Ścieżka do folderu z obrazami
mask_dir = 'path_to_masks'    # Ścieżka do folderu z maskami

# Funkcja do wczytywania danych
def load_data(image_dir, mask_dir, img_width, img_height):
    images = []
    masks = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)
        
        # Wczytanie obrazu i maski
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize do pożądanego rozmiaru
        img = cv2.resize(img, (img_width, img_height))
        mask = cv2.resize(mask, (img_width, img_height))
        
        # Normalizacja
        img = img / 255.0
        mask = mask / 255.0
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# Wczytanie danych
images, masks = load_data(image_dir, mask_dir, IMG_WIDTH, IMG_HEIGHT)
images = np.expand_dims(images, axis=-1)  # Dodanie kanału dla obrazów
masks = np.expand_dims(masks, axis=-1)    # Dodanie kanału dla masek

# Podział na zbiór treningowy i walidacyjny
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Definicja modelu U-Net
def unet(input_size=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Middle
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    
    # Decoder
    up4 = UpSampling2D(size=(2, 2))(conv3)
    up4 = Conv2D(128, 2, activation='relu', padding='same')(up4)
    merge4 = concatenate([conv2, up4], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(merge4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)
    
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = Conv2D(64, 2, activation='relu', padding='same')(up5)
    merge5 = concatenate([conv1, up5], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)
    
    # Output
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)
    
    model = Model(inputs, outputs)
    return model

# Tworzenie modelu
model = unet()
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)

# Ewaluacja modelu
loss, accuracy = model.evaluate(X_val, y_val, verbose=1)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# Wizualizacja wyników
def plot_results(images, masks, predictions, num_samples=5):
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        # Oryginalny obraz
        plt.subplot(3, num_samples, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title("Oryginalny obraz")
        plt.axis('off')
        
        # Maska
        plt.subplot(3, num_samples, i + 1 + num_samples)
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.title("Maska")
        plt.axis('off')
        
        # Przewidywana maska
        plt.subplot(3, num_samples, i + 1 + 2 * num_samples)
        plt.imshow(predictions[i].squeeze(), cmap='gray')
        plt.title("Przewidywana maska")
        plt.axis('off')
    plt.show()

# Przewidywanie na zbiorze walidacyjnym
predictions = model.predict(X_val)
plot_results(X_val, y_val, predictions)
