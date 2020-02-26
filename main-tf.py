from flask import Flask, request, render_template, request, redirect 
import pandas as pd
import keras
import numpy as np
from keras.applications import mobilenet
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import tensorflow as tf
import numpy as np
import os
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

app = Flask(__name__)


#Load the MobileNet model
mobilenet_model = mobilenet.MobileNet(weights='imagenet')
global graph,session #due to flask multi-threading
graph = tf.get_default_graph() 
session=init()


APP_ROOT= os.path.dirname(os.path.abspath(__file__))


def prediction(image):
    '''generates prediction on image using mobilenet'''

    # load an image in PIL format
    original_image = load_img(image, target_size=(224, 224))
    
    
    print('1')
    # convert the PIL image (width, height) to a NumPy array (height, width, channel)
    numpy_image = img_to_array(original_image)
    #print(numpy_image.shape)
    print('2')
    # Convert the image into 4D Tensor (samples, height, width, channels) by adding an extra dimension to the axis 0.
    input_image = np.expand_dims(numpy_image, axis=0)
    print('3')
    processed_image_mobilenet = mobilenet.preprocess_input(input_image.copy())
    print('4')
    with session.as_default():
        with graph.as_default():
            predictions_mobilenet = mobilenet_model.predict(processed_image_mobilenet)
            print('5')
            label_mobilenet = decode_predictions(predictions_mobilenet)
    print('6')
    print(label_mobilenet)
    print('7')
    return(label_mobilenet)



@app.route('/')
def index():
    return 'placeholder'

@app.route('/upload-image', methods=['GET', 'POST']) #get/post generates a request object
def upload_image():
    target=os.path.join(APP_ROOT,'images/')
    #print(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            print(image)
            destination= '/'.join([target,image.filename])
            print(destination)
            image.save(destination)
            print(image)
            prediction(destination)
            
            return redirect(request.url)

    return render_template('upload_image.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
