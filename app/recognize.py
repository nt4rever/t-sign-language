from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
from keras.models import load_model
from helper import load_labels, IMAGE_SIZE, MODEL_PATH, DB_PATH

cap = cv2.VideoCapture("../demo/video/common_character.avi")
detector = HandDetector(maxHands=1)
OFFSET = 20
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# load model that already training
model = load_model(MODEL_PATH)
# load label names from database
label_names = load_labels(DB_PATH)

while True:
    _, img = cap.read()
    imgOrigin = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        try:
            if w / h < 1:
                padding = int((h - w) / 2)
                startPoint = (x - padding - OFFSET, y - OFFSET)
                endPoint = (startPoint[0] + h + 2 * OFFSET, startPoint[1] + h + 2 * OFFSET)
            else:
                padding = int((w - h) / 2)
                startPoint = (x - OFFSET, y - padding - OFFSET)
                endPoint = (startPoint[0] + w + 2 * OFFSET, startPoint[1] + w + 2 * OFFSET)
            if startPoint[0] < 0 or startPoint[1] < 0 or endPoint[0] > frame_width or endPoint[1] > frame_height:
                continue
            handCrop = imgOrigin[startPoint[1]:endPoint[1], startPoint[0]:endPoint[0]]
            hsv = cv2.cvtColor(handCrop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0, 39, 2), (179, 255, 255))
            result_mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            img_resized = cv2.resize(result_mask_bgr, (IMAGE_SIZE, IMAGE_SIZE))
            img_array = np.expand_dims(img_resized, axis=0)
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            confidence_score = prediction[0][predicted_index]
            print(confidence_score)
            img_text = label_names[predicted_index]
            cv2.imshow("hand", result_mask_bgr)
            cv2.putText(imgOrigin, f"{img_text}", (startPoint[0] + 10, startPoint[1] + 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 1,
                        (0, 255, 0))
            imgOrigin = cv2.rectangle(imgOrigin, startPoint, endPoint,
                                      (0, 255, 0), thickness=2, lineType=8, shift=0)
        finally:
            pass

    cv2.imshow("app", imgOrigin)
    cv2.waitKey(1)
