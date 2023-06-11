import cv2
import os
import sqlite3
from cvzone.SelfiSegmentationModule import SelfiSegmentation

segmentor = SelfiSegmentation()


def create_folder(folder_name):
    if not os.path.exists('../dataset/training_set/' + folder_name):
        os.mkdir('../dataset/training_set/' + folder_name)
    if not os.path.exists('./dataset/test_set/' + folder_name):
        os.mkdir('../dataset/test_set/' + folder_name)


def store_in_db(g_id, g_name):
    conn = sqlite3.connect("../store/database/gesture.db")
    cmd = "INSERT INTO gesture (g_id, g_name) VALUES (%s, \'%s\')" % (
        g_id, g_name)
    try:
        conn.execute(cmd)
    except sqlite3.IntegrityError:
        choice = input(
            "g_id already exists. Want to change the record? (y/n): ")
        if choice.lower() == 'y':
            cmd = "UPDATE gesture SET g_name = \'%s\' WHERE g_id = %s" % (
                g_name, g_id)
            conn.execute(cmd)
        else:
            print("Doing nothing...")
            return
    conn.commit()


def capture_images(ges_no, ges_name):
    create_folder(str(ges_no))
    cam = cv2.VideoCapture(0)
    img_counter = 0
    t_counter = 1
    training_set_image_name = 1
    test_set_image_name = 1
    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)
        img_crop = img[102:298, 427:623]
        img_no_bg = segmentor.removeBG(img_crop, (0, 0, 0), threshold=0.50)  # background is black
        gray = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        (_, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.putText(frame, str(img_counter), (30, 400),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("test", frame)
        cv2.imshow("thresh", thresh)
        t = cv2.waitKey(1)
        if t == ord('c'):
            if t_counter <= 350:
                img_name = "../dataset/training_set/" + \
                           str(ges_no) + "/{}.png".format(training_set_image_name)
                save_img = cv2.resize(thresh, (64, 64))
                cv2.imwrite(img_name, save_img)
                print("{} written!".format(img_name))
                training_set_image_name += 1

            if 350 < t_counter <= 400:
                img_name = "../dataset/test_set/" + \
                           str(ges_no) + "/{}.png".format(test_set_image_name)
                save_img = cv2.resize(thresh, (64, 64))
                cv2.imwrite(img_name, save_img)
                print("{} written!".format(img_name))
                test_set_image_name += 1
                if test_set_image_name > 250:
                    break

            t_counter += 1
            if t_counter == 401:
                t_counter = 1
            img_counter += 1

        elif t == 27:
            break

        if test_set_image_name > 250:
            break

    store_in_db(ges_no, ges_name)
    cam.release()
    cv2.destroyAllWindows()


ges_no = input("Enter gesture no: ")
ges_name = input("Enter gesture name: ")
capture_images(ges_no, ges_name)
