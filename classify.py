import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
from zipfile import ZipFile
import warnings

warnings.filterwarnings('ignore')

# --- UNZIP ONLY IF NOT ALREADY DONE ---
if not os.path.exists('PetImages'):
    with ZipFile('archive.zip', 'r') as zip_ref:
        zip_ref.extractall()
        print('✅ Dataset extracted.')
else:
    print('✅ Dataset already extracted.')

# --- CLEAN BAD IMAGES ---
for folder in ['Cat', 'Dog']:
    folder_path = os.path.join('PetImages', folder)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            img = mpimg.imread(file_path)
            if img is None or img.size == 0:
                raise ValueError("Corrupt image")
        except:
            os.remove(file_path)  # remove broken files

# --- VISUALIZE SAMPLE IMAGES ---
cat_dir = os.path.join('PetImages', 'Cat')
dog_dir = os.path.join('PetImages', 'Dog')
cat_images = [os.path.join(cat_dir, fname) for fname in os.listdir(cat_dir)[:8]]
dog_images = [os.path.join(dog_dir, fname) for fname in os.listdir(dog_dir)[:8]]

fig = plt.gcf()
fig.set_size_inches(16, 16)
for i, img_path in enumerate(cat_images + dog_images):
    sp = plt.subplot(4, 4, i + 1)
    sp.axis('Off')
    img = mpimg.imread(img_path)
    plt.imshow(img)
plt.show()

# --- LOAD DATA ---
batch_size = 32
img_size = (200, 200)
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = image_dataset_from_directory(
    'PetImages',
    validation_split=0.1,
    subset='training',
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_dataset = image_dataset_from_directory(
    'PetImages',
    validation_split=0.1,
    subset='validation',
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# ❗ Automatically skip bad data
def configure_dataset(ds):
    return ds.prefetch(buffer_size=AUTOTUNE).cache().filter(lambda x, y: tf.reduce_all(tf.math.is_finite(x)))

train_dataset = configure_dataset(train_dataset)
val_dataset = configure_dataset(val_dataset)

# --- BUILD MODEL ---
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# --- TRAIN MODEL (3 Epochs only) ---
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3
)

# --- SAVE MODEL ---
os.makedirs('model', exist_ok=True)
model.save('model/cat_dog_model.h5')
print("✅ Model saved in 'model/' folder.")

# --- PLOT TRAINING RESULTS ---
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot(title='Loss')
history_df[['accuracy', 'val_accuracy']].plot(title='Accuracy')
plt.show()

# --- PREDICT FUNCTION ---
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(200, 200))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]
    print(f"\n🧠 Prediction Score: {prediction:.4f}")
    print("🐶 Dog" if prediction >= 0.5 else "🐱 Cat")

# --- SAMPLE TEST ---
sample_cat = os.path.join(cat_dir, '0.jpg')
sample_dog = os.path.join(dog_dir, '0.jpg')
if os.path.exists(sample_cat): predict_image(sample_cat)
if os.path.exists(sample_dog): predict_image(sample_dog)
