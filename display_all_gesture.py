import cv2, os, random, sqlite3
import numpy as np

def label_model():
    conn = sqlite3.connect("./store/database/gesture.db")
    cursor = conn.execute("SELECT g_id, g_name from gesture")
    labels = []
    for row in cursor:
        labels.append(row[1])
    return labels

gestures = os.listdir('dataset/training_set')
if len(gestures)%5 != 0:
	rows = int(len(gestures)/5)+1
else:
	rows = int(len(gestures)/5)
rows

label = label_model()

begin_index = 0
end_index = 5
full_img = None
for i in range(rows):
    col_img = None
    for j in range(begin_index, end_index):
        if j>len(gestures)-1:
            j = j -len(gestures)
        k = label[j]
        img_path = "dataset/training_set/%s/%d.png" % (j, random.randint(1, 100))
        img = cv2.imread(img_path, 0)
        text = np.zeros((64, 64))
        cv2.putText(text, k, (20, 40),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
        img = np.hstack((text,img))
        if np.any(img == None):
            img = np.zeros((64, 64), dtype = np.uint8)
        if np.any(col_img == None):
            col_img = img
        else:
            col_img = np.hstack((col_img, img))
    begin_index += 5
    end_index += 5
    if np.any(full_img == None):
        full_img = col_img
    else:
        full_img = np.vstack((full_img, col_img))

cv2.imshow("gestures", full_img)
cv2.imwrite('./store/full_gesture.jpg', full_img)
cv2.waitKey(0)