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


# \LCD4L1XL3 的valid.xml中只有一个区域的位置信息，以下几行代码是临时处理
def predict_on_LCD4(root_dict, sub1_dicts):
    # 图片数组
    test_images = np.array([])
    # 实际分类数组
    labels = np.array([])
    sub2_dicts = [r"\effective", r"\no"]
    # 预测分类数组
    predicts = np.array([])
    # 将xml中的每一个valid标签汇总到该数组中，以便之后统计模型准确率
    is_valid = np.array([])
    dom = xmldom.parse(root_dict + sub1_dicts[0] + r"\config\valid.xml")
    root = dom.documentElement
    width = 25; height = 27
    area_roots = root.getElementsByTagName('area')
    area_count = area_roots.length

    for area_root in area_roots:
        valid = int(area_root.getElementsByTagName('valid')[0].childNodes[0].data)
        is_valid = np.append(is_valid, [valid])

    for sub_dict1 in sub1_dicts:
        # dom = xmldom.parse(root_dict + sub_dict1 + r"\config\valid.xml")
        # root = dom.documentElement
        # width = 25; height = 27
        # area_roots = root.getElementsByTagName('area')
        # area_count = area_roots.length
        for sub_dict2 in sub2_dicts:
            for (root, dirs, files) in os.walk(root_dict + sub_dict1 + sub_dict2):
                for filename in files:
                    # 按顺序读取图片文件
                    file_all_name = os.path.join(root, filename)
                    if os.path.isfile(file_all_name) and os.path.splitext(file_all_name)[1] == '.jpg':
                        if sub_dict2 == r"\no":
                            labels = np.append(labels, [0], axis=0)
                        else:
                            labels = np.append(labels, [1], axis=0)
                        
                        # 遍历电表上的四个区域
                        img = cv2.imread(file_all_name)
                        # test_images = np.array([])
                        for area_root in area_roots:
                            xmin = int(area_root.getElementsByTagName('xmin')[0].childNodes[0].data)
                            xmax = int(area_root.getElementsByTagName('xmax')[0].childNodes[0].data)
                            ymin = int(area_root.getElementsByTagName('ymin')[0].childNodes[0].data)
                            ymax = int(area_root.getElementsByTagName('ymax')[0].childNodes[0].data)
                            image = img[ymin:ymax, xmin:xmax]
                            # print(image.shape)
                            image = cv2.resize(image, (int(image.shape[0]/(ymax-ymin)*width), int(image.shape[1]/(xmax-xmin)*height)))
                            model_image = np.array(image.reshape(-1, 3, width, height))
                            if not test_images.any():       # 直接写成 if not test_images 会报错：数组长度超过2时有二义性
                                test_images = model_image.copy()
                            else:
                                test_images = np.append(test_images, model_image, axis=0)
    
    temp_predicts = np.array(model.predict_classes(test_images, batch_size=32, verbose=1))
    i = 0
    print("temp_predicts: ", temp_predicts.shape[0])
    while i < temp_predicts.shape[0]:
        if np.bincount(np.equal(np.array(temp_predicts[i:i+area_count]), np.array(is_valid)))[0] == 0:
            predicts = np.append(predicts, [1], axis=0)
        else:
            predicts = np.append(predicts, [0], axis=0)
        i = i + area_count

    print(predicts)
    print(labels)
    accuracy = np.count_nonzero(np.equal(predicts, labels)) / len(predicts)
    return accuracy

if __name__ == "__main__":
    root_dict = r"..\Effective Screen"
    sub1_dicts = [r"\LCD4L1XL1", r"\LCD4L1XL3"]
    accuracy = predict_on_LCD4(root_dict, sub1_dicts)
    print(accuracy)
