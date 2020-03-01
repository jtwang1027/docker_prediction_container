## Objective  
Build a Docker container for image classification algorithm ***(mobilenet)*** accessible via Flask application.


## Background  
Mobilenet (v2.0) is a streamlined architecture used to allow light weight deep neural networks that can be run on mobile devices. In contrast to other object detection architectures, mobilenet replaces convolutional layers with depthwise separable convolutions for faster computation.


## Method  
A Docker container was built to run the application. The pytorch/pytorch image was pulled from DockerHub and this repository was added (including the flask application (main-torch.py)) and additional packages were added through *requirements.txt* and files (*imagenet_class_index.json* which contains the mapping for imagenet classes (number to class name) used in the main-torch.py app). See the Dockerfile for more details or pull the image *jtwang1027/torch-app* from DockerHub.
```console
  docker pull jtwang1027/torch-app
  docker run -it -p 8080:8080 jtwang1027/torch-app
```

## Application
When images are uploaded to /upload-image . The images are saved in the folder *static* and the prediction function is run on the saved image which results in:
- *Mobilenet v2* is loaded
- Image is preprocessed (resizing, center cropping, and normalizing values)
- Evaluation 
- predictions are mapped to actual categories using *imagenet_class_index.json*

The application was tested on Google cloud console using test images from the tiny Imagenet test set [(link)](https://tiny-imagenet.herokuapp.com/)

## Screenshots from demo video  
***Image upload page***
<img width="908" alt="upload_image" src="https://user-images.githubusercontent.com/46359281/75620703-fe024500-5b59-11ea-84f7-ed282c36ebe6.png">  

***Predicted image category and probability***  
<img width="388" alt="predicted_image" src="https://user-images.githubusercontent.com/46359281/75620694-ef1b9280-5b59-11ea-9596-8a8129484d08.png">

