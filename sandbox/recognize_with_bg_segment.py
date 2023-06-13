from keras.models import load_model
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import cv2
import numpy as np
from app.helper import load_labels, IMAGE_SIZE, MODEL_PATH, DB_PATH

# load model that already training
model = load_model(MODEL_PATH)
# load label names from database
label_names = load_labels(DB_PATH)
# create segmentor to remove background
segmentor = SelfiSegmentation()
# capture video from webcam
cap = cv2.VideoCapture(0)

is_capture = False
img_text = ''
temp = ''
sentence = ''
counter = 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # draw an area zone to recognize gesture
    img = cv2.rectangle(frame, (425, 100), (625, 300),
                        (0, 255, 0), thickness=2, lineType=8, shift=0)
    imgCrop = img[102:298, 427:623]
    imgNoBg = segmentor.removeBG(imgCrop, (0, 0, 0), threshold=0.50)  # background is black
    gray = cv2.cvtColor(imgNoBg, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # (_, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    (_, thresh) = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    cv2.putText(frame, img_text, (425, 90), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
    cv2.putText(frame, sentence, (30, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255))
    cv2.imshow("capture", frame)
    cv2.imshow("thresh", thresh)

    if is_capture:
        img_pred = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        img_resized = cv2.resize(img_pred, (IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.expand_dims(img_resized, axis=0)
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        img_text = label_names[predicted_index]
        if temp == '' or img_text == temp and img_text != "Nothing" and img_text != "Z":
            counter += 1
        else:
            counter = 0
        temp = img_text

        if counter > 30:
            sentence = sentence + temp
            temp = ''
            counter = 0

    t = cv2.waitKey(1)
    if t == ord('c'):
        is_capture = True
        print("---------Start capture---------")
    elif t == ord('s'):
        is_capture = False
        print("---------Stop capture---------")
    elif t == ord('r'):
        sentence = ''
    elif t == 32:
        sentence = sentence + ' '
    elif t == 8:
        sentence = sentence[:-1]
    elif t == 27:
        break

cap.release()
cv2.destroyAllWindows()
