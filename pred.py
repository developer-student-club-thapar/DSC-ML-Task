import os 
import tensorflow as tf
import matplotlib.pyplot as plt

def load_and_prep_image(filename, img_shape=300):
    img = tf.io.read_file(img)

    img = tf.image.decode_image(img, channels = 3)

    img = tf.image.resize(img, size=[img_shape, img_shape])

    img = img/255.

    return img

def pred_and_plot(model, filename, class_names):

    img = load_and_prep_image(filename)

    pred = model.predict(tf.expand_dims(img,axis = 0))

    if len(pred[0]) > 1:
        pred_class = class_names[pred.argmax()]
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]

    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False);  
