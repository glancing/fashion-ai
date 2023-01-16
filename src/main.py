import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def predictTemperature(imagePath):
  new_model = load_model('clothestemperature.h5')
  img = cv2.imread(imagePath)
  resize = tf.image.resize(img, (256,256))
  prediction = new_model.predict(np.expand_dims(resize/255, 0), verbose=0)
  if prediction > 0.5: 
    print('Predicted for Warm temperature')
  else:
      print('Predicted for Cold temperature')

predictTemperature('./images/shorts.jpeg')

