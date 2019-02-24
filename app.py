from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

# Keras
from keras.preprocessing import image
from keras.models import model_from_json
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
# ResNet50
# VGG16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16, decode_predictions as decode_predictions_vgg16
# Xception

# Define a flask app
app = Flask(__name__)

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

global our_model

json_file=open('modelfinal.json','r')
loaded_model_json=json_file.read()
json_file.close()
our_model=model_from_json(loaded_model_json)

our_model.load_weights('modelfinal.h5')

our_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')

print('Latest model loaded.')

target_name=['ketch', 'Motorbikes', 'lotus', 'electric_guitar', 'kangaroo', 'watch', 'bonsai', 'umbrella', 'laptop', 'schooner', 'joshua_tree', 'soccer_ball', 'cougar_face', 'minaret', 'ewer', 'lamp', 'elephant', 'trilobite', 'hawksbill', 'sunflower', 'starfish', 'scorpion', 'chandelier', 'car_side', 'stop_sign', 'grand_piano', 'euphonium', 'crayfish', 'ibis', 'Faces_easy', 'dalmatian', 'ferry', 'airplanes', 'llama', 'dragonfly', 'dolphin', 'butterfly', 'menorah', 'BACKGROUND_Google', 'Faces', 'crab', 'yin_yang', 'chair', 'helicopter', 'brain', 'buddha', 'flamingo', 'revolver', 'Leopards'] 

print('Running on http://localhost:5000')

def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    return file_path



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predictVGG16', methods=['GET', 'POST'])
def predictVGG16():
    if request.method == 'POST':
        file_path = get_file_path_and_save(request)

           
        img = image.load_img(file_path, target_size=(224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)

        img_tensor /= 255.
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])




        preds = our_model.predict(images, verbose=1)
        preds=np.argmax(preds,axis=1)
        preds=int(preds)
        res=target_name[preds]

      #  for i in preds:
      #      if(i==1):
       #         res=target_name[i]
        # decode the results into a list of tuples (class, description, probability)
        #pred_class = decode_predictions_vgg16(preds, top=1)
        result = str(res)#str(pred_class[0][0][1])  # Convert to string
        return result
    return None



if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
