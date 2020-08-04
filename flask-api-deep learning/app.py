import os
import requests
import numpy as np
import tensorflow as tf
from scipy.misc import imsave,imread 
from flask import Flask,request,jsonify

with open("fashion_model_flask.json","r") as f:
    model_json=f.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights("fashion_model_flask.h5")

app=Flask(__name__)
@app.route("/api/v1/<string:img_name>",method=["POST"])
def classify_image(img_name):
    upload_dir="uploads/"
    image=imread(upload_dir+img_name)
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    prediction=model.predict([image.reshape(1,28*28)])
    return jsonify({"object_detected":classes[np.argmax(prediction)]})
app.run(port=5000,debug=Flase)