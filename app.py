import os
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from data import class_names
import keras

all_classes = class_names

app = Flask(__name__)

# Load your trained ML model
mon_model = tf.keras.models.load_model("saved_trained_model")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    image = request.files['image']

    if image.filename == '':
        return jsonify({'error': 'No selected file'})

    if image:
        # Save the uploaded image
        image_path = os.path.join('uploads/', image.filename)
        image.save(image_path)

        # Load and preprocess the image for prediction
        img = Image.open(image_path)
        # Preprocess the image (resize, normalize, etc.) as needed for your model
        img = load_and_prep_image(image_path)

        # Make a prediction
        prediction = pred_and_plot(mon_model, img, all_classes)

        return jsonify({'prediction': prediction})
    

def load_and_prep_image(filename, img_shape=300):
    img = tf.io.read_file(filename)

    img = tf.image.decode_image(img, channels = 3)

    img = tf.image.resize(img, size=[img_shape, img_shape])

    img = img/255.

    return img

def pred_and_plot(model, img, class_names):

    pred = model.predict(tf.expand_dims(img,axis = 0))

    if len(pred[0]) > 1:
        pred_class = class_names[pred.argmax()]
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]

    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False); 

    return pred_class

if __name__ == '__main__':
    app.run(debug=True)



