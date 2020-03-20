# Pothole Detection System (Image Classification)

Detecting potholes on roads using images processed through a CNN model. This is not a realtime system but I'm hoping to make it realtime for the application purposes. As of now the model simply processes a single image and tells whether the image has a pothole or not. Note that the model does not tell the number of potholes in the images. That's something for the future and I'll use YOLO (You Only Look Once architecture) for that.

## What's In The Repo

* *My Dataset* - Contains the images which were used for training the model
* *Predictor.py* - The code that loads the model (*sample.h5*), loads the dataset and uses it for prediction
* *main.py* - The code that creates the model, trains it and saves it as *sample.h5*
* *sample.h5* - The saved model that is loaded for prediction

## Future Work

In future I aim to make it a robust realtime system which will use OpenCV and a bit more advanced NN implementation.
