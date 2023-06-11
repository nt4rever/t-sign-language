import numpy as np
from keras.models import load_model
import cv2
import base64
import json
from helper import load_labels, MODEL_PATH, DB_PATH, IMAGE_SIZE

# load model that already training
model = load_model(MODEL_PATH)
# load label names from database
label_names = load_labels(DB_PATH)


def read_base64(uri):
    encoded_data = uri.split(',')[1]
    np_arr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def predictor(path):
    img = read_base64(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 39, 2), (179, 255, 255))
    result_mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img_resized = cv2.resize(result_mask_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    return label_names[predicted_index]


def test_image(path):
    img = read_base64(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 39, 2), (179, 255, 255))
    _, buffer = cv2.imencode('.png', mask)
    img_as_text = base64.b64encode(buffer)
    return img_as_text


def report():
    f = open('../store/data.json')
    data = json.load(f)
    return data


def get_cm():
    t = cv2.imread('../store/confusion_matrix.png')
    _, buffer = cv2.imencode('.png', t)
    img_as_text = base64.b64encode(buffer)
    return img_as_text
