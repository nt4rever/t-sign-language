import os
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import cv2
import base64

classifier = load_model('model.h5')


def predictor(path):
    img = readb64(path)
    test_image = cv2.resize(img, (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    label = label_model()
    res = result[0]
    index = np.where(res == 1)[0][0]
    if index:
        return label[index]
    else:
        return "null"


def readb64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def label_model():
    folder = './dataset/training_set'
    sub_folders = [name for name in os.listdir(
        folder) if os.path.isdir(os.path.join(folder, name))]
    return sub_folders
