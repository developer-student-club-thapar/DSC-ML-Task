from flask import Flask, render_template, request, jsonify
import os
import keras
from keras import models
import tensorflow as tf
from PIL import Image
from pred import load_and_prep_image, pred_and_plot

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'

model = 'D:/monument-prediction/monument_model/saved_model.pb'

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'] )    
def predict():
    if 'image' not in request.files:
        return 'No Image Part'
    
    image = request.files['image']
    if image.filename == '':
        return 'No Image Selected'
    
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    load_and_prep_image(image)
    image.save(image_path)
    processed_image = load_and_prep_image(image_path)

    prediction = model.predict(processed_image)

    response = {'prediction' : prediction.tolist()}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug = True)
