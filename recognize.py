from keras.preprocessing import image
from keras.models import load_model
import sqlite3
import cv2
import numpy as np


def nothing(x):
    pass


image_x, image_y = 64, 64
classifier = load_model('./store/model/model.h5')


def predictor():
    test_image = image.load_img(
        './store/1.png', target_size=(image_x, image_y))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    label = label_model()
    i = list(result[0]).index(max(result[0]))
    if max(result[0]) > 0:
        return label[i]
    return "null"


def label_model():
    conn = sqlite3.connect("./store/database/gesture.db")
    cursor = conn.execute("SELECT g_id, g_name from gesture")
    labels = []
    for row in cursor:
        labels.append(row[1])
    return labels


cam = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 300, 250)
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 21, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 2, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

cv2.namedWindow("capture")

flag_capture = False
img_name = "./store/1.png"
img_text = ''
temp = ''
counter = 0
sentence = ''

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    img = cv2.rectangle(frame, (425, 100), (625, 300),
                        (0, 255, 0), thickness=2, lineType=8, shift=0)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    imcrop = img[102:298, 427:623]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    full = cv2.imread('./store/full_gesture.jpg')
    cv2.putText(frame, img_text, (30, 300),
                cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.putText(frame, sentence, (30, 400),
                cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255))
    cv2.imshow("capture", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("full", full)

    if flag_capture:
        save_img = cv2.resize(mask, (image_x, image_y))
        cv2.imwrite(img_name, save_img)
        img_text = predictor()
        if temp == '' or img_text == temp:
            counter += 1
        else:
            counter = 0
        temp = img_text

        if counter > 20:
            sentence = sentence+temp
            temp = ''
            counter = 0

    t = cv2.waitKey(1)
    if t == ord('c'):
        flag_capture = True
        print("---------Start capture---------")
    elif t == ord('s'):
        flag_capture = False
        print("---------Stop capture---------")
    elif t == ord('r'):
        sentence = ''
    elif t == 32:
        sentence = sentence+' '
    elif t == 8:
        sentence = sentence[:-1]
    elif t == 27:
        break

cam.release()
cv2.destroyAllWindows()
