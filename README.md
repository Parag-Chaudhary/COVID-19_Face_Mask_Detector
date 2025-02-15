# COVID-19_Face_Mask_Detector

## Project Objective 
Due to the COVID-19 Pandemic situation, wearing of a face mask is become mandantory. So to keep the track of this, i create a Machine learning model which helps to detect whether a person in front of the camera(Real-Time), is wearing a mask or not.

## Pre-Requisites Before running
* First of all, Please change the path of DIRECTORY in `Train_Mask_Detector.py` file to your working directory path.
* Install Tensorflow using `pip install tensorflow`
* Install Keras using `pip install keras`
* Install OpenCV using `pip install opencv-python`
* Install Numpy using `pip install numpy`

## Dataset Contents
The dataset consists of 3833 images belonging to the following two classes:
* with_mask: 1915 images
* without_mask: 1918 images

## Order of files to be run
1. First you have to run the `Train_Mask_Detector.py` file to create and train the model.
2. Secondly you have to run the `Mask_Detector_using_model.py` file to detect.

## Result
![ezgif com-gif-maker](https://user-images.githubusercontent.com/70112406/94625115-9cfc4480-02d5-11eb-9431-97881c0bb3d3.gif)
