import base64
import numpy as np
import io
import os
import re
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
from flask import Flask
from flask_mail import Mail, Message
from skimage import img_as_ubyte

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='template')
cors = CORS(app, resources={r"/foo": {"origins": "http://localhost:port"}})

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
#app.config['MAIL_DEBUG'] = False
app.config['MAIL_USERNAME'] = 'gaitwebapp@gmail.com'
app.config['MAIL_PASSWORD'] = 'vgg-19gait69'
app.config['MAIL_DEFAULT_SENDER'] = 'gaitwebapp@gmail.com'
app.config['MAIL_MAX_EMAILS'] = None
#app.config['MAIL_SUPPRESS_SEND'] = False
app.config['MAIL_ASCII_ATTACHMENTS'] = True

mail = Mail(app)

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

def sort_nicely( l ):
    # Sort the given list in the way that humans expect.
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )

def mass_center(img,is_round=True):
    Y = img.mean(axis=1)
    X = img.mean(axis=0)
    Y_ = np.sum(np.arange(Y.shape[0]) * Y)/np.sum(Y)
    X_ = np.sum(np.arange(X.shape[0]) * X)/np.sum(X)
    if is_round:
        return int(round(X_)),int(round(Y_))
    return X_,Y_

def resize(images, pathOut):

    count = 0
    for image in images:

        if image.dtype != 'uint8':
            # try:
            print(image.dtype)
            image = img_as_ubyte(image)
            # except ValueError:
            #     continue

        x,y,w,h = cv2.boundingRect(image)

        if h == 0 or w == 0:
            count += 1
            continue

        x_c, y_c = mass_center(image)

        padded = np.zeros((h,h),dtype=np.uint8)
        x_l = h//2-x_c+x
        x_r = h//2+w-x_c+x

        if h < x_r or x_l < 0:
            count += 1
            continue

        padded[:,x_l:x_r] = image[y:y+h,x:x+w]
        resized_image = np.array(Image.fromarray(padded).resize((224,224)))

        # plt.imshow(resized_image)
        # plt.show()
        cv2.imwrite( pathOut + "/frame%d.png" % count, resized_image)
        count += 1


def subtract(pathFrom, path):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    list_dir = os.listdir(pathFrom)
    sort_nicely(list_dir)
    count = 0

    for frame in list_dir:
        newDir = pathFrom+frame
        try:
            image = Image.open(newDir)
            image = np.array(image)
            fgmask = fgbg.apply(image)
            fgmask = (fgmask != 127) * fgmask
            morph = morphological(fgmask, 5)
            saving_Images(morph, count, path)
            count = count + 1
        except (IOError, SyntaxError) as e:
            print('Bad file:', frame) 
            print(e) 
    
def saving_Images(image, count, path):
    cv2.imwrite(path + "frame%d.png" % count, image)
    print(path + "frame%d.png" % count)
    

def morphological(image, k):
    kernel = np.ones((k,k), np.uint8)
    bin_img = image.astype(np.uint8)
    imageSe = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    return imageSe

def create_gei(lista):
   #check_bad_files(directory)
   #files = os.listdir(directory)
   #sort_nicely(lista)
   #current = os.getcwd
   #path = current +'/static/imgs/generatedGEI.png'
   #img_array = [imread(directory + f) for f in files]
   gei = np.mean(lista,axis=0)
   plt.imsave('generatedGEI.png', gei)

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

@app.route('/template/email', methods=['POST', 'GET'])
def sendEmail():
    message = request.get_json(force=True)
    dip = message['dip']
    hem = message['hemi']
    norm = message['nor']
    neuro = message['neuro']
    park = message['park']
    name = message['nam']
    email = message['pema']
    current = os.getcwd()
    path1 = current+"/static/imgs/saliency.png"
    path2 = current+"/static/imgs/grad.png"
    path3 = current+"/static/imgs/imageFeature.png"
    msg = Message('Diagnosis Results (Do not reply)', recipients=[email])
    msg.body = "Dear "+name+",\r\n\r\n Your results from the Gait Web App are now available. Please check them below:\r\n\r\n Diplegic:"+str(dip)+";\r\n Hemiplegic:"+str(hem)+";\r\n Normal:"+str(norm)+"\r\n Neuropathic:"+str(neuro)+";\r\n Parkinsonian:"+str(park)+".\r\n\r\n Please find attached the image results from the WA diagnosis.\r\n\r\n Best Regars, \r\n\r\n Gait Web App"
    with app.open_resource(path1) as fp:
        msg.attach("saliencyMap.png", "image/png", fp.read())
    
    with app.open_resource(path2) as fp:
        msg.attach("heatmap.png", "image/png", fp.read())
    
    with app.open_resource(path3) as fp:
        msg.attach("featureMap.png", "image/png", fp.read())

    mail.send(msg)

    return 'SUCCESSSSS!!!'


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

@app.route('/template/predict3', methods=['POST', 'GET'])
def predict3():
    if request.method == 'POST':
        message = request.get_json(force=True)
        print(message)
        encoded = message['image']
        decoded = base64.b64decode(encoded)
        current = os.getcwd()
        print(current)
        path = current+"/static/video/"
        pathFrames = current+"/static/frames/"
        pathSubtract = current+"/static/subtract/"
        pathResized = current+"/static/resized/"
        with open(current+"/static/video/video.mp4", "wb") as v:
            v.write(decoded)

        count = 0
        vidcap = cv2.VideoCapture(path+"/video.mp4")
        success,image = vidcap.read()
        success = True
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*100))    
            success,image = vidcap.read()
            print ('Read a new frame: ', success)
            if success is False:
                continue
            else:
                cv2.imwrite(pathFrames + "frame%d.png" % count, image)  
                count = count + 1

        subtract(pathFrames, pathSubtract)
        listasub = os.listdir(pathSubtract)
        sort_nicely(listasub)
        listImages = []
        for frame in listasub:
            if frame != '.DS_Store':
                image = Image.open(pathSubtract + frame)
                image = np.array(image)
                listImages.append(image)
        resize(listImages, pathResized)
        lista = os.listdir(pathResized)
        sort_nicely(lista)
        listImages2 = []
        for res in lista:
            if res != '.DS_Store':
                image2 = Image.open(pathResized + res)
                image2 = np.array(image2)
                listImages2.append(image2)
        create_gei(listImages2)

        imageresp = open("generatedGEI.png", "rb").read()
        imageresp = base64.b64encode(imageresp)

        response = {
           "GEI": imageresp.decode('utf-8')
        }

        print(response)
        return jsonify(response)
    else:

        return render_template('predict3.html')

            

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
        current = os.getcwd()
        print(current)
        plt.imsave(current+'/static/imgs/imageFeature.png', first_layer_activation[0, :, :, int(channels)], cmap='viridis')
        imageresp = cv2.imread(current+'/static/imgs/imageFeature.png')
        imageresp = np.array(imageresp)
        imageresp = cv2.resize(imageresp, (224,224), 3)
        plt.imsave(current+'/static/imgs/imageFeature.png', imageresp)
        imageresp = open(current+"/static/imgs/imageFeature.png", "rb").read()
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
        current = os.getcwd()
        plt.imsave(current+'/static/imgs/imageFeature.png', first_layer_activation[0, :, :, int(channels)], cmap='viridis')
        imageresp = cv2.imread(current+'/static/imgs/imageFeature.png')
        imageresp = np.array(imageresp)
        imageresp = cv2.resize(imageresp, (224,224), 3)
        plt.imsave(current+'/static/imgs/imageFeature.png', imageresp)
        imageresp = open(current+"/static/imgs/imageFeature.png", "rb").read()
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
        current = os.getcwd()
        plt.imsave(current+"/static/imgs/saliency.png", saliency_map[0], cmap='jet')

        imageresp = open(current+"/static/imgs/saliency.png", "rb").read()
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
        current = os.getcwd()
        plt.imsave(current+"/static/imgs/saliency.png", saliency_map[0], cmap='jet')

        imageresp = open(current+"/static/imgs/saliency.png", "rb").read()
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
        path = os.getcwd()
        print(path)
        path = path+'/static/imgs/grad.png'
        plt.imsave(path, heatmap, cmap='jet')

        imageresp = open(path, "rb").read()
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
        path = os.getcwd()
        print(path)
        path = path + "/static/imgs/grad.png"
        plt.imsave(path, heatmap, cmap='jet')

        imageresp = open(path, "rb").read()
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