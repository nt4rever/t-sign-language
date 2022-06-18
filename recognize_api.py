import sqlite3
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import cv2
import base64
import json

classifier = load_model('./store/model/model_v3.h5')


def predictor(path, l_h, l_s, l_v, u_h, u_s, u_v):
    img = readb64(path)
    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    img_name = "./store/1.png"
    save_img = cv2.resize(mask, (64, 64))
    cv2.imwrite(img_name, save_img)
    test_image = image.load_img('./store/1.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    label = label_model()
    i = list(result[0]).index(max(result[0]))
    if max(result[0])>0:
        return label[i]
    else:
        return "null"


def readb64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def label_model():
    conn = sqlite3.connect("./store/database/gesture.db")
    cursor = conn.execute("SELECT g_id, g_name from gesture")
    labels = []
    for row in cursor:
        labels.append(row[1])
    return labels


def test_image(path, l_h, l_s, l_v, u_h, u_s, u_v):
    img = readb64(path)
    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    _, buffer = cv2.imencode('.png', mask)
    img_as_text = base64.b64encode(buffer)
    return img_as_text


def report():
    f = open('./store/data.json')
    data = json.load(f)
    return data

def get_cm():
    t = cv2.imread('./store/confusion_matrix.png')
    _, buffer = cv2.imencode('.png', t)
    img_as_text = base64.b64encode(buffer)
    return img_as_text