#!/usr/bin/env python
# coding: utf-8

# In[4]:

import warnings
warnings.filterwarnings('ignore')
import argparse
import json
import logging
import logging.config
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
from PIL import Image
import scipy
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import layers
logger=logging.getLogger()

image_size = 224

def parse_args():
    # python -W ignore predict.py -h
    parser = argparse.ArgumentParser(description='A Flower Image Classifier')
    parser.add_argument('image_path',
                     help='image path', default='./test_images/cautleya_spicata.jpg')
    parser.add_argument('saved_model', help='A TensorFlow Model', default='./models/best_model.h5')
    parser.add_argument('--top_k', default=-1,
                    help='Return the top K most likely classes')
    parser.add_argument('--category_names', default=None,
                    help='Map flower name with class')
    return parser.parse_args()

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()

def read_json(map_path):
    with open(map_path, 'r') as f:
        class_names = json.load(f)
    return class_names

def predict(image_path, model, top_k=5):

    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    processed_test_image = np.expand_dims(processed_test_image, 0)
    probs = model.predict(processed_test_image)
    return tf.nn.top_k(probs, k=top_k)

def main():
    args = parse_args()
    reloaded_keras_model = tf.keras.models.load_model(args.saved_model,
                                                      custom_objects={'KerasLayer': hub.KerasLayer})


    top_k = int(args.top_k)
    category = args.category_names
    if category is not None:
        class_names = read_json(category)
    if top_k == -1:
        probs, classes = predict(args.image_path,reloaded_keras_model, 1)
        probs = probs.numpy()[0]
        classes = classes.numpy()[0]
        top_prob = probs[0]
        if (category is not None):
            top_class = class_names[str(classes[0]+1)]
        else:
            top_class = classes[0]
        print('The most possible class is {}'.format(top_class), 'with {} possiblity'.format(top_prob))
    else:
        probs, classes = predict(args.image_path, reloaded_keras_model, top_k)
        probs = probs.numpy()[0]
        classes = classes.numpy()[0]
        top_classes = {}
        if (category is not None):
            for i in range(top_k):
                top_classes[class_names[str(classes[i] + 1)]] = probs[i]
        else:
            for i in range(top_k):
                top_classes[classes[i]] = probs[i]
        print('The top k possible classese are: ', top_classes)

if __name__ == '__main__':
    main()


# In[ ]:




