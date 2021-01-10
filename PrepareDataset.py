import pathlib
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
# import tensorflow_datasets as tfds
import os
import cv2

import pickle
basePath = os.path.dirname(os.path.abspath(__file__))
DATADIR = (f'{basePath}/images')


CATEGORIES = []
training_data = []
IMG_SIZE = 70


def prepareDataset():
  for folder in os.listdir(DATADIR):
    CATEGORIES.append(folder)

  def create_training_data():
    for category in CATEGORIES:  #read the name of the folders and assign them as categories

        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
  create_training_data()
  X = []
  y = []

  for features, label in training_data:
      X.append(features)
      y.append(label)

  X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

  pickle_out = open("XS.pickle", "wb")
  pickle.dump(X, pickle_out)
  pickle_out.close()

  pickle_out = open("ys.pickle", "wb")
  pickle.dump(y, pickle_out)
  pickle_out.close()




