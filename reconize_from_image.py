import os
from re import T
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import cv2
import base64
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle

classifier = load_model('model_v2.h5')


def predictor(path, l_h, l_s, l_v, u_h, u_s, u_v):
    img = readb64(path)
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
    folder = './dataset/training_set'
    sub_folders = [name for name in os.listdir(
        folder) if os.path.isdir(os.path.join(folder, name))]
    return sub_folders


def test_image(path, l_h, l_s, l_v, u_h, u_s, u_v):
    img = readb64(path)
    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    _, buffer = cv2.imencode('.png', mask)
    img_as_text = base64.b64encode(buffer)
    return img_as_text


def load_test_image():
    images_labels = []
    images = glob("dataset/test_set/*/*.png")
    images.sort()
    for i in images:
        label = i[i.find(os.sep)+1: i.rfind(os.sep)]
        img = cv2.imread(i)
        test_image = cv2.resize(img, (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        images_labels.append((test_image, label))
    return images_labels

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import itertools
    import matplotlib.pyplot as plt

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('./store/confusion_matrix.png')

def report():
    images_labels = load_test_image()
    images_labels = shuffle(shuffle(shuffle(shuffle(images_labels))))
    images, labels = zip(*images_labels)
    pred_labels = []
    folder = './dataset/training_set'
    sub_folders = [name for name in os.listdir(
        folder) if os.path.isdir(os.path.join(folder, name))]

    pred_probabs = []
    for i in images:
        p = classifier.predict(i)
        pred_probabs.append(p)

    for pred_probab in pred_probabs:
        i = list(pred_probab[0]).index(max(pred_probab[0]))
        pred_labels.append(sub_folders[i])

    cm = confusion_matrix(labels, np.array(pred_labels))
    plot_confusion_matrix(cm, range(25), normalize=False)
    cl = classification_report(labels, np.array(pred_labels))
    return cl

def get_cm():
    t = cv2.imread('./store/confusion_matrix.png')
    _, buffer = cv2.imencode('.png', t)
    img_as_text = base64.b64encode(buffer)
    return img_as_text