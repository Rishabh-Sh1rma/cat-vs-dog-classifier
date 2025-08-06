# 🐱🐶 Cat vs Dog Image Classifier  
### Built with 💙 by [Rishabh Sharma](https://github.com/Rishabh-Sh1rma)

An intelligent and fun deep learning project to **classify images as either cats or dogs** using TensorFlow & Keras.  
This project walks you through the entire ML pipeline — from data preprocessing, model training, to live prediction. Perfect for beginners or developers looking to get hands-on with CNNs and image classification. 🧠📸
Get The Database here : https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset?resource=download
---

## ✨ Features
- 📂 Preprocessed dataset of 100 cats and 100 dogs from the PetImages archive.
- 🧠 Custom Convolutional Neural Network using Keras.
- 📊 Training & validation performance visualization.
- 📸 Predict from custom images (drag & drop your own pets!).
- 💾 Model is saved (`cat_dog_model.h5`) for easy reusability.

---

## 📁 Project Structure

```bash
cat-vs-dog-classifier/
├── PetImages/
│   ├── Cat/
│   └── Dog/
├── test_images/
│   └── sample1.jpg
├── model/
│   └── cat_dog_model.h5
├── classify.py         # Trains and saves the CNN model
├── predict.py          # Run predictions on new images
└── README.md


```
🚀 How to Run
1. Clone the repo
```bash
git clone https://github.com/Rishabh-Sh1rma/cat-vs-dog-classifier.git
cd cat-vs-dog-classifier

```
2. Install dependencies
```bash
pip install tensorflow opencv-python matplotlib
```
3. Train the model
```bash
python classify.py
```
✅ This will train the model on the 200 images and save it to model/cat_dog_model.h5.

🔍 Predict New Images
Once trained, you can predict any custom image like this:

```bash
python predict.py test_images/sample1.jpg
```
✔️ Output:
```bash
Predicted Class: Dog 🐶
```
📸 Sample Output
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/478791f6-713b-4ee9-b50f-98b2937ab21a" />
![WhatsApp Image 2025-08-06 at 16 53 08_ff4070ec](https://github.com/user-attachments/assets/25052be4-3490-42a5-a0e4-812623a4665e)
<img width="271" height="251" alt="image" src="https://github.com/user-attachments/assets/b9cc62d4-36e3-4533-be34-f6bc605f0131" />
<img width="257" height="257" alt="image" src="https://github.com/user-attachments/assets/6ced72e5-68fc-4301-8db1-b1c8a8dfc5a8" />


📊 Accuracy & Loss Plot
The script automatically shows you training vs validation accuracy and loss.
Helps you visually analyze if your model is overfitting, underfitting, or learning well!

🤖 Model Summary
A simple yet effective CNN architecture:

Conv2D → MaxPool → Conv2D → MaxPool → Flatten → Dense → Output

Binary classification (sigmoid)

🙋‍♂️ About Me
I'm Rishabh Sharma, a developer passionate about AI, computer vision, and building useful tools.
