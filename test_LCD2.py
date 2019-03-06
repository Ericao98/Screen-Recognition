from keras import backend as K  
K.set_image_dim_ordering('th')

from keras.models import load_model, model_from_json
from keras.utils import to_categorical
import numpy as np
import cv2
import xml.dom.minidom as xmldom
import os
import matplotlib.pyplot as plt

model = load_model("Training model by keras3.h5")

def read_data_from_xml(directory):
    dom = xmldom.parse(directory)
    root = dom.documentElement
    xmin = int(root.getElementsByTagName('xmin')[0].childNodes[0].data)
    xmax = int(root.getElementsByTagName('xmax')[0].childNodes[0].data)
    ymin = int(root.getElementsByTagName('ymin')[0].childNodes[0].data)
    ymax = int(root.getElementsByTagName('ymax')[0].childNodes[0].data)
    valid = int(root.getElementsByTagName('ymax')[0].childNodes[0].data)
    return np.array([xmin, xmax, ymin, ymax, valid])

def read_LCD2(root_dict, sub1_dicts):
    sub2_dicts = [r"\effective", r"\no"]
    # 原始图片集合
    images = np.array([])
    # 实际标签数组
    labels = np.array([])
    # 图像边框等相关数据
    infos = np.array([])
    width = 25; height = 27

    for sub_dict1 in sub1_dicts:
        # 截取图片有效区域
        info = read_data_from_xml(root_dict + sub_dict1 + r"\config\valid.xml")
        for sub_dict2 in sub2_dicts:
            for (root, dirs, files) in os.walk(root_dict + sub_dict1 + sub_dict2):
                for filename in files:
                    file_all_name = os.path.join(root, filename)
                    # 统计所有标签
                    if os.path.isfile(file_all_name) and os.path.splitext(file_all_name)[1] == '.jpg':
                        if sub_dict2 == r"\no":
                            labels = np.append(labels, [0], axis=0)
                        else:
                            labels = np.append(labels, [1], axis=0)

                        [xmin, xmax, ymin, ymax, valid] = info
                        img = cv2.imread(file_all_name)
                        image = img[ymin:ymax, xmin:xmax]
                        # print(image.shape)
                        image = cv2.resize(image, (int(image.shape[0]/(ymax-ymin)*width), int(image.shape[1]/(xmax-xmin)*height)))
                        # print(image.shape)
                        model_image = np.array(image.reshape(-1, 3, width, height))
                        # plt.imshow(image)
                        # plt.show()
                        if not images.any():       # 直接写成 if not images 会报错：数组长度超过2时有二义性
                            images = model_image.copy()
                        else:
                            images = np.append(images, model_image, axis=0)

    return images, labels

def predict_on_LCD2(root_dict, sub_dicts):
    # 图片数组
    images = np.array([])
    # 标签数组
    labels = np.array([])
    # 预测数组
    predicts = np.array([])

    # 从文件中读取
    images, labels = read_LCD2(root_dict, sub_dicts)
    predicts = model.predict_classes(images, batch_size=32, verbose=1)
    accuracy = np.count_nonzero(np.equal(labels, predicts)) / len(labels)

    print(labels)
    print(predicts)

    return accuracy


if __name__ == "__main__":
    root_dict = r"..\Effective Screen"
    # sub_dicts = [r"\LCD2L1XL", r"\LCD2L2XL", r"\LCD2L2CS"]
    sub_dicts = [r"\LCD2L2XL", r"\LCD2L2CS"]

    accuracy = predict_on_LCD2(root_dict, sub_dicts)
    print("accuracy =", accuracy)
