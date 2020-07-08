import base64
import numpy as np
import io
import os
from PIL import Image
import tensorflow 
import cv2
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify, json
from flask import Flask, render_template, url_for
from flask import send_from_directory     
from flask_cors import CORS, cross_origin
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='template')
cors = CORS(app, resources={r"/foo": {"origins": "http://localhost:port"}})


def get_model():
    global model
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = tensorflow.keras.models.load_model('1_savedModel.h5')
    print(" * Model GEI loaded!")

def get_model2():
    global model2
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model2 = tensorflow.keras.models.load_model('SEI_finalModel.h5')
    print(" * Model SEI loaded!")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, target_size, 3)
    print(np.shape(image))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

print(" * Loading Keras Models... ")
get_model()
get_model2()

@app.route('/template/predict', methods=['POST', 'GET'])
@cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def predict():
    if request.method == 'POST':
        message = request.get_json(force=True)
        print(message)
        encoded = message['image']
        decoded = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image, target_size=(224, 224))
    
        prediction = model.predict(processed_image).tolist()
        print(prediction[0][0])
        print(prediction[0][1])
        print(prediction[0][2])
        print(prediction[0][3])
        print(prediction[0][4])

        response = {
            "prediction": {
              "diplegic": prediction[0][1],
              "hemiplegic": prediction[0][2],
              "neuropathic": prediction[0][4],
              "normal": prediction[0][3],
              "parkinsonian": prediction[0][0]
            }
        }
        print(response)
        return jsonify(response)
    else:
        return render_template('predict.html')


@app.route('/template/predict2', methods=['POST', 'GET'])
@cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def predict2():
    if request.method == 'POST':
        message = request.get_json(force=True)
        print(message)
        encoded = message['image']
        decoded = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image, target_size=(224, 224))
    
        prediction = model2.predict(processed_image).tolist()
        print(prediction[0][0])
        print(prediction[0][1])
        print(prediction[0][2])
        print(prediction[0][3])
        print(prediction[0][4])

        response = {
            "prediction": {
              "diplegic": prediction[0][1],
              "hemiplegic": prediction[0][2],
              "neuropathic": prediction[0][4],
              "normal": prediction[0][3],
              "parkinsonian": prediction[0][0]
            }
        }
        print(response)
        return jsonify(response)
    else:
        return render_template('predict.html')


@app.route('/template/index', methods=['GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

#@app.route('/favicon.ico') 
#def favicon(): 
    #return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')