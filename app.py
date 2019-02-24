from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras.models import model_from_json
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
global model
json_file=open('models/modelfinal.json','r')
loaded_model_json=json_file.read()
json_file.close()
model=model_from_json(loaded_model_json)

model.load_weights('modelfinal.h5')

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')

print('Latest model loaded.')

target_name=['ketch', 'Motorbikes', 'lotus', 'electric_guitar', 'kangaroo', 'watch', 'bonsai', 'umbrella', 'laptop', 'schooner', 'joshua_tree', 'soccer_ball', 'cougar_face', 'minaret', 'ewer', 'lamp', 'elephant', 'trilobite', 'hawksbill', 'sunflower', 'starfish', 'scorpion', 'chandelier', 'car_side', 'stop_sign', 'grand_piano', 'euphonium', 'crayfish', 'ibis', 'Faces_easy', 'dalmatian', 'ferry', 'airplanes', 'llama', 'dragonfly', 'dolphin', 'butterfly', 'menorah', 'BACKGROUND_Google', 'Faces', 'crab', 'yin_yang', 'chair', 'helicopter', 'brain', 'buddha', 'flamingo', 'revolver', 'Leopards'] 



print('Model loaded')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction

        img = image.load_img(file_path, target_size=(224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)

        img_tensor /= 255.
        
     
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])




        preds = model.predict(images, verbose=1)
        preds=np.argmax(preds,axis=1)
        preds=int(preds)
        res=target_name[preds]


        result = str(res)               # Convert to string
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
