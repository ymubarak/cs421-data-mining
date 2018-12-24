import os
from skimage.io import imread
from scipy.io import loadmat

def load_images(path):
    path += "images/"
    train_path = path + "/train/"
    val_path = path + "/val/"
    test_path = path + "/test/"

    x_train = []
    train_list = os.listdir(train_path)
    for img in train_list:
        if img.endswith("jpg"):
            x_train.append(imread(train_path+img))


    x_val = []
    val_list = os.listdir(val_path)
    for img in val_list:
        if img.endswith("jpg"):
            x_val.append(imread(val_path+img))


    x_test = []
    test_list = os.listdir(test_path)
    for img in test_list:
        if img.endswith("jpg"):
            x_test.append(imread(test_path+img))

    return x_train, x_val, x_test


def load_ground_truth(path):
    path += "groundTruth/"
    train_path = path + "/train/"
    val_path = path + "/val/"
    test_path = path + "/test/"

    y_train = []
    train_list = os.listdir(train_path)
    for img in train_list:
        if img.endswith("mat"):
            y_train.append(loadmat(train_path+img))


    y_val = []
    val_list = os.listdir(val_path)
    for img in val_list:
        if img.endswith("mat"):
            y_val.append(loadmat(val_path+img))


    y_test = []
    test_list = os.listdir(test_path)
    for img in test_list:
        if img.endswith("mat"):
            y_test.append(loadmat(test_path+img))

    return y_train, y_val, y_test