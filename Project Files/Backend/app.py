# app.py
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = load_model('healthy_vs_rotten.h5')

# Your index_to_class mapping
index_to_class = {'Apple__Healthy': 0,
 'Apple__Rotten': 1,
 'Banana__Healthy': 2,
 'Banana__Rotten': 3,
 'Bellpepper__Healthy': 4,
 'Bellpepper__Rotten': 5,
 'Carrot__Healthy': 6,
 'Carrot__Rotten': 7,
 'Cucumber__Healthy': 8,
 'Cucumber__Rotten': 9,
 'Grape__Healthy': 10,
 'Grape__Rotten': 11,
 'Guava__Healthy': 12,
 'Guava__Rotten': 13,
 'Jujube__Healthy': 14,
 'Jujube__Rotten': 15,
 'Mango__Healthy': 16,
 'Mango__Rotten': 17,
 'Orange__Healthy': 18,
 'Orange__Rotten': 19,
 'Pomegranate__Healthy': 20,
 'Pomegranate__Rotten': 21,
 'Potato__Healthy': 22,
 'Potato__Rotten': 23,
 'Strawberry__Healthy': 24,
 'Strawberry__Rotten': 25,
 'Tomato__Healthy': 26,
 'Tomato__Rotten': 27}
index_to_class = {v: k for k, v in index_to_class.items()}

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess image
    img = load_img(filepath, target_size=(224, 224))
    x = img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    # Predict
    preds = model.predict(x)
    predicted_index = np.argmax(preds)
    predicted_class = index_to_class[predicted_index]

    # Optionally delete the file after prediction
    os.remove(filepath)

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
