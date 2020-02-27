## Objective  
Deploy an object detection algorithm ***(mobilenet)*** in a docker container using Google Cloud.


## Background  
Mobilenet (v2.0) is a streamlined architecture used to allow light weight deep neural networks that can be run on mobile devices. In contrast to other object detection architectures, mobilenet replaces convolutional layers with depthwise separable convolutions for faster computation.


## Method  
A Docker container was built to run the application. The pytorch/pytorch image was pulled from DockerHub and this repository was added (including the flask application (main-torch.py)) and additional packages were added through *requirements.txt* and files (*imagenet_class_index.json* which contains the mapping for imagenet classes (number to class name) used in the main-torch.py app). See the Dockerfile for more details or pull the image jtwang1027/torch-app from DockerHub.
