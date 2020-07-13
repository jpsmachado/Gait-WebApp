import base64
import numpy as np
import io
import os
from PIL import Image
import tensorflow as tf
import cv2
import base64
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from tensorflow.keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify, json
from flask import Flask, render_template, url_for
from flask import send_from_directory     
from flask_cors import CORS, cross_origin
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import numpy as np
import matplotlib.pyplot as plt
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tensorflow.keras.applications.vgg19 import preprocess_input

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='template')
cors = CORS(app, resources={r"/foo": {"origins": "http://localhost:port"}})


def get_model():
    global model
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = tf.keras.models.load_model('1_savedModel.h5')
    print(" * Model GEI loaded!")

def get_model2():
    global model2
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model2 = tf.keras.models.load_model('SEI_finalModel.h5')
    print(" * Model SEI loaded!")

def lossGEI(output):
    return (output[0][geiclass])

def lossSEI(output):
    return (output[0][seiclass])

def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear
    return m

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, target_size, 3)
    print(np.shape(image))
    image = np.expand_dims(image, axis=0)
    print(np.shape(image))
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

        global geiclass
        geiclass = np.argmax(prediction[0])

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
        
        global seiclass
        seiclass = np.argmax(prediction[0])

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


@app.route('/template/feature', methods=['POST', 'GET'])
def featureExtraction():
    if request.method == 'POST':
        message = request.get_json(force=True)
        print(message)
        encoded = message['image']
        decoded = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image, target_size=(224, 224))
        layers = message['lay']
        print(layers)
        channels = message['chan']
        print(channels)
        print("Numero de layers:"+str(len(model.layers)))
        layer_outputs = [layer.output for layer in model.layers[:21]]
        activation_model = Model(inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(processed_image)
        first_layer_activation = activations[int(layers)]
        print(first_layer_activation.shape)
        plt.imsave('imageFeature.png', first_layer_activation[0, :, :, int(channels)], cmap='viridis')
        imageresp = open("imageFeature.png", "rb").read()
        imageresp = base64.b64encode(imageresp)
        print(str(imageresp))
        response = {
           "feature": imageresp.decode('utf-8')
        }

        print(response)
        return jsonify(response)
    else:
        return render_template('predict.html')

@app.route('/template/feature2', methods=['POST', 'GET'])
def featureExtraction2():
    if request.method == 'POST':
        message = request.get_json(force=True)
        print(message)
        encoded = message['image']
        decoded = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image, target_size=(224, 224))
        layers = message['lay']
        print(layers)
        channels = message['chan']
        print(channels)
        print("decodeeeee")
        print("Numero de layers:"+str(len(model.layers)))
        layer_outputs = [layer.output for layer in model.layers[:21]]
        activation_model = Model(inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(processed_image)
        first_layer_activation = activations[int(layers)]
        print(first_layer_activation.shape)
        plt.imsave('imageFeature.png', first_layer_activation[0, :, :, int(channels)], cmap='viridis')
        imageresp = open("imageFeature.png", "rb").read()
        imageresp = base64.b64encode(imageresp)
        print(str(imageresp))
        response = {
           "feature": imageresp.decode('utf-8')
        }

        print(response)
        return jsonify(response)
    else:
        return render_template('predict.html')


@app.route('/template/saliency', methods=['POST', 'GET'])
def saliency():
    if request.method == 'POST':
        message = request.get_json(force=True)
        print(message)
        encoded = message['image']
        decoded = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image, target_size=(224, 224))
        
        saliency = Saliency(model,
                    model_modifier=model_modifier,
                    clone=False)
        saliency_map = saliency(lossGEI, processed_image)
        saliency_map = normalize(saliency_map)

        plt.imsave("saliency.png", saliency_map[0], cmap='jet')

        imageresp = open("saliency.png", "rb").read()
        imageresp = base64.b64encode(imageresp)
        print(str(imageresp))
        response = {
           "saliency": imageresp.decode('utf-8')
        }

        print(response)
        return jsonify(response)
    else:
        return render_template('predict.html')


@app.route('/template/saliency2', methods=['POST', 'GET'])
def saliency2():
    if request.method == 'POST':
        message = request.get_json(force=True)
        print(message)
        encoded = message['image']
        decoded = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image, target_size=(224, 224))
        
        saliency = Saliency(model,
                    model_modifier=model_modifier,
                    clone=False)
        saliency_map = saliency(lossSEI, processed_image)
        saliency_map = normalize(saliency_map)

        plt.imsave("saliency.png", saliency_map[0], cmap='jet')

        imageresp = open("saliency.png", "rb").read()
        imageresp = base64.b64encode(imageresp)
        print(str(imageresp))
        response = {
           "saliency": imageresp.decode('utf-8')
        }

        print(response)
        return jsonify(response)
    else:
        return render_template('predict.html')



@app.route('/template/grad', methods=['POST', 'GET'])
def grad():
    if request.method == 'POST':
        message = request.get_json(force=True)
        print(message)
        encoded = message['image']
        decoded = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image, target_size=(224, 224))
        gradcam = Gradcam(model,
                  model_modifier=model_modifier,
                  clone=False)
        cam = gradcam(lossGEI, processed_image, penultimate_layer=-1)#, model.layers number)
        cam = normalize(cam)
        cmap = cm.get_cmap("jet")
        heatmap = np.uint8(cmap(cam[0])[..., :5] * 255)
        plt.imsave("grad.png", heatmap, cmap='jet')

        imageresp = open("grad.png", "rb").read()
        imageresp = base64.b64encode(imageresp)
        print(str(imageresp))
        response = {
           "grad": imageresp.decode('utf-8')
        }

        print(response)
        return jsonify(response)


    else:
        return render_template('predict.html')


@app.route('/template/grad2', methods=['POST', 'GET'])
def grad2():
    if request.method == 'POST':
        message = request.get_json(force=True)
        print(message)
        encoded = message['image']
        decoded = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image, target_size=(224, 224))
        gradcam = Gradcam(model,
                  model_modifier=model_modifier,
                  clone=False)
        cam = gradcam(lossSEI, processed_image, penultimate_layer=-1)#, model.layers number)
        cam = normalize(cam)
        cmap = cm.get_cmap("jet")
        heatmap = np.uint8(cmap(cam[0])[..., :5] * 255)
        plt.imsave("grad.png", heatmap, cmap='jet')

        imageresp = open("grad.png", "rb").read()
        imageresp = base64.b64encode(imageresp)
        print(str(imageresp))
        response = {
           "grad": imageresp.decode('utf-8')
        }

        print(response)
        return jsonify(response)


    else:
        return render_template('predict.html')


@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')