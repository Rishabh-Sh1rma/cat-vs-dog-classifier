import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

# --- LOAD MODEL ---
model = load_model('model/cat_dog_model.h5')
print("✅ Model loaded.")

# --- PREDICT FUNCTION ---
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(200, 200))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)

    print(f"\n🧠 Prediction Score: {result[0][0]:.4f}")
    print("🐶 Dog" if result[0][0] >= 0.5 else "🐱 Cat")

# --- EXAMPLE TESTS ---
# --- EXAMPLE TESTS ---
predict_image('PetImages/Cat/10.jpg')
predict_image('PetImages/Dog/10.jpg')

