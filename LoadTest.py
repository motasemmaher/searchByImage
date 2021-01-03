import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
# import time
import os
from PIL import Image

from numpy import asarray


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
CATEGORIES = []
basePath = os.path.dirname(os.path.abspath(__file__))
DATADIR = (f'{os.path.dirname(os.path.abspath(__file__))}/images')


def getNameIamge(image, name):
  for folder in os.listdir(DATADIR):
      CATEGORIES.append(folder)
  cnn = tf.keras.models.load_model(f'{basePath}/model')
  # print(image)
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
    return CATEGORIES[num]

  # CATEGORIES = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
  # pickle_in = open("XS.pickle", "rb")
  # X = pickle.load(pickle_in)

  # pickle_in = open("ys.pickle", "rb")
  # y = pickle.load(pickle_in)
  # filepath = r'C:\Users\Hamza\Desktop\26c2a4dc3aa4143fca5e49b455ef36e5.jpg'
  # img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
  # new_array = cv2.resize(img_array, (70, 70))
  # new_image = new_array.reshape(-1, 70, 70, 1)

  # predictions = cnn.predict(X)

  # for i in range(25):
  #     num = np.argmax(predictions[i])
  #     print(num)
