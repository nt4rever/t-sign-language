from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
from keras.models import load_model
from helper import load_labels, IMAGE_SIZE, MODEL_PATH, DB_PATH

# ../demo/video/alphabet.avi
# webcam: 0

cap = cv2.VideoCapture(0)
# media pipe
detector = HandDetector(maxHands=1)

OFFSET = 30
# get width height value of video input
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# load model that already training
model = load_model(MODEL_PATH)
# load label names from database
label_names = load_labels(DB_PATH)


def nothing(x):
    pass


cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 300, 250)
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 60, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 2, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

full = cv2.imread('../store/full_gesture.jpg')

while True:
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    # define range color hsv to threshold
    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])

    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgOrigin = img.copy()

    # use mediapipe to detect hands
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        try:
            # fit image aspect ratio to 1/1 (square)
            if w / h < 1:
                padding = int((h - w) / 2)
                startPoint = (x - padding - OFFSET, y - OFFSET)
                endPoint = (startPoint[0] + h + 2 * OFFSET, startPoint[1] + h + 2 * OFFSET)
            else:
                padding = int((w - h) / 2)
                startPoint = (x - OFFSET, y - padding - OFFSET)
                endPoint = (startPoint[0] + w + 2 * OFFSET, startPoint[1] + w + 2 * OFFSET)
            if startPoint[0] < 0 or startPoint[1] < 0 or endPoint[0] > frame_width or endPoint[1] > frame_height:
                raise Exception()

            # crop image that contains hand gesture
            handCrop = imgOrigin[startPoint[1]:endPoint[1], startPoint[0]:endPoint[0]]
            # convert this image to HSV and threshold
            hsv = cv2.cvtColor(handCrop, cv2.COLOR_BGR2HSV)
            # [0,60,2] -> [179,255,255]
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            result_mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # resize this image to 64x64 pixel
            img_resized = cv2.resize(result_mask_bgr, (IMAGE_SIZE, IMAGE_SIZE))
            # convert this image to np array (matrix)
            img_array = np.expand_dims(img_resized, axis=0)
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            confidence_score = prediction[0][predicted_index]
            print(confidence_score)
            img_text = label_names[predicted_index]
            cv2.imshow("hand", result_mask_bgr)
            # put predict result onto screen (if confidence score > 0.9)
            if confidence_score > 0.9:
                cv2.putText(imgOrigin, f"{img_text}", (startPoint[0] + 10, startPoint[1] + 30),
                            cv2.FONT_HERSHEY_TRIPLEX, 1,
                            (18, 19, 179))
            # draw rectangle that contain hand gesture
            imgOrigin = cv2.rectangle(imgOrigin, startPoint, endPoint,
                                      (108, 144, 234), thickness=2, lineType=8, shift=0)
        except:
            if cv2.getWindowProperty("hand", cv2.WND_PROP_VISIBLE) > 0:
                cv2.destroyWindow("hand")
            pass
    else:
        if cv2.getWindowProperty("hand", cv2.WND_PROP_VISIBLE) > 0:
            cv2.destroyWindow("hand")

    cv2.imshow("full gesture", full)
    cv2.imshow("app", imgOrigin)
    cv2.waitKey(1)

cv2.destroyAllWindows()
cam.release()
