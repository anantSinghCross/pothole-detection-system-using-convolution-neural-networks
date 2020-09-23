# Pothole Detection System (Real-time Image Classification)

Detecting potholes on roads using live video feed processed through a CNN model. This is (now) a realtime system. The model was trained on my laptop's GPU (NVIDIA GTX 1650 4GB). Note that the model does not tell the number of potholes in the images. That's something for the future and I'll use YOLO (You Only Look Once architecture) OR Mask-RCNN for that.

## Contents Of This Readme

1. [What's In The Repo](https://github.com/anantSinghCross/pothole-detection-system-using-convolution-neural-networks#whats-in-the-repo)
2. [Check Your Libraries](https://github.com/anantSinghCross/pothole-detection-system-using-convolution-neural-networks#check-your-libraries)
3. [Working of Files in *Real-time Files* Folder](https://github.com/anantSinghCross/pothole-detection-system-using-convolution-neural-networks#working-of-files-in-real-time-files-folder)
4. [Future Work](https://github.com/anantSinghCross/pothole-detection-system-using-convolution-neural-networks#future-work)
5. [Note](https://github.com/anantSinghCross/pothole-detection-system-using-convolution-neural-networks#note)

## What's In The Repo

* *My Dataset* - Contains the images which were used for training the model
* *Real-time Files* - Contains the new updated real-time prediction files along with an improved model (now model takes in images of size 300x300 which were previously 100x100 and increased the number of epochs to 1000 which were 500 previously)
* *Predictor.py* - The code that loads the model (*sample.h5*), loads the testing dataset and uses it for prediction
* *main.py* - The code that creates the model, trains it and saves it as *sample.h5*
* *sample.h5* - The saved model that is loaded for prediction

## Check Your Libraries

* `Numpy`
* `Tensorflow`
* `Keras`
* `Scikit-learn`
* `OpenCV`
* `Imutils`

*Instructions on how to install these libraries can be found extensively on internet.*

## Working of Files in *Real-time Files* Folder

* *main.py* - This module’s main aim is to create, prepare and train the model. Internally, also it prepares the dataset which it loads from a specific location in the machine.
Preparing the dataset includes:
   1. Extracting all the images from a specified location.
   2. Preprocessing of images which includes:
      - Converting images from colored to grayscale (to reduce processing power)
      - Resizing all the images to the same dimensions i.e. 300x300 px
   3. Creating corresponding output values for each image from the dataset which will be used for training.
   
* *Predictor.py* - This module’s main aim is to predict the presence of potholes in a certain number of images. The module loads the model and the images from the machine. The images are again preprocessed in the same manner as in trainer module. The images are fed into the model and predictions and accuracies are printed on the console.

* *realtimePredictor.py* - This module’s main aim is to predict presence of potholes in a live video feed. The module loads the model and captures video from the camera hardware using python’s open source library OpenCV. The video feed which is captured is divided into separate frames as 2d arrays. Then each frame is preprocessed in the same manner as in trainer module, to meet the required dimensions of input. After the preprocessing is done, each frame is fed into the model for prediction and the predictions are then printed on the screen along with the confidence level.

* *full_model.h5* - This is the new and improved model. (89.99% testing accuracy)

***

## Future Work

If, in future, I decide work on this project, I will most likely work on finding out the number of potholes in a particular frame of the video feed and also creating bounding boxes around the potholes so that they are identifiable.

### Note

Since the dataset is web-scrapped from Google Images it is highly inconsistent. Therefore, it is recommended to use a proper dataset for training the model. There are a few good pothole datasets on kaggle but I didn't use them due to their huge size. If you're going to use it for research purposes the web-scrapped dataset won't suffice.
