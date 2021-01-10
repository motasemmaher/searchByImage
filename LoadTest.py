import tensorflow as tf
import numpy as np
import base64
import pickle
import cv2
import os
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from PIL import Image
from numpy import asarray


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
CATEGORIES = []
basePath = os.path.dirname(os.path.abspath(__file__))
DATADIR = (f'{os.path.dirname(os.path.abspath(__file__))}/images')


def getNameIamge(img, name):
  image = base64.b64decode(img.img)
  for folder in os.listdir(DATADIR):
    CATEGORIES.append(folder)
  cnn = tf.keras.models.load_model(f'{basePath}/model')
  path = f'{basePath}/imagesForGettingName'
  if not os.path.exists(path):
    os.mkdir(path)
  path += f'/{name}.png'
  with open(path, 'wb') as f:
    f.write(image)
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (70, 70))
    new_image = new_array.reshape(-1, 70, 70, 1)
    predictions = cnn.predict(new_image)
    num = np.argmax(predictions)
    os.remove(path)
    if len(CATEGORIES) > num:
      return CATEGORIES[num]
    return "Image not found"
