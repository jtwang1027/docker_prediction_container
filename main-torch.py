from flask import Flask, request, render_template, request, redirect, url_for 
import pandas as pd
import numpy as np
import os


app = Flask(__name__)
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

APP_ROOT= os.path.dirname(os.path.abspath(__file__)) #for saving uploaded images


def prediction(filename):
    '''takes filename, predicts using mobilenet from pytorch
    and returns tuple (top category, probability)'''
    import torch
    import json
    model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=True)
    model.eval()

    file_read = open("imagenet_class_index.json").read()
    categ = json.loads(file_read)
    #print(categ['0'])

    # sample execution (requires torchvision)
    from PIL import Image
    from torchvision import transforms
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    with torch.no_grad():
        output = model(input_batch)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probab=torch.nn.functional.softmax(output[0], dim=0)
    idx=sorted(range(len(probab)), key=lambda i: probab[i])[-3:] #top 3 predictions
    
    most_prob=str(probab[idx[-1]].numpy()) #probability of most likely category
    print(categ[str(idx[-1])])
    print(most_prob)
    print('finished')
    return( (categ[str(idx[-1])][1], most_prob)) 

#################
##APP BELOW
###########
'''
@app.route('/')
def index():
    return 'placeholder'
'''
#@app.route('/', defaults={'upload-image'})
#@app.route('/upload-image', methods=['GET', 'POST']) #get/post generates a request object
@app.route('/', methods=['GET', 'POST']) #get/post generates a request object
def upload_image():
    target=os.path.join(APP_ROOT,'static/')
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
            output=prediction(destination)
            return render_template('prediction.html',pred_class=output[0], probability=output[1], filepath=image.filename )

    return render_template('upload_image.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
