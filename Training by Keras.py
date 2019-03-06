from keras import backend as K  
K.set_image_dim_ordering('th') 

import numpy as np
import datetime
nowTime=datetime.datetime.now().microsecond
np.random.seed(nowTime)

from keras.models import Sequential, load_model
from keras.layers import Dense, MaxPooling2D, Activation, Conv2D, Flatten, Dropout
from keras.optimizers import Adam, SGD, RMSprop
import numpy as np
from keras.utils import to_categorical
import cv2

# 从.npy文件中导入训练数据和测试数据
X_train = np.load(r"..\Effective Screen\eff_small_move\numpy\x_train.npy")
y_train = np.load(r"..\Effective Screen\eff_small_move\numpy\y_train.npy")
X_test = np.load(r"..\Effective Screen\eff_small_move\numpy\x_test.npy")
y_test = np.load(r"..\Effective Screen\eff_small_move\numpy\y_test.npy")

# 打印前100张图像
# for i in range(0, 100):
#     b = np.array(XTrain[i][:, :, 0]).astype(np.uint8)
#     g = np.array(XTrain[i][:, :, 1]).astype(np.uint8)
#     r = np.array(XTrain[i][:, :, 2]).astype(np.uint8)
#     img = cv2.merge([b, g, r])
#     cv2.imshow("test", img)
#     print(yTrain[i, 0])
#     cv2.waitKey(0)

# 数据处理
train_shape = X_train.shape
test_shape = X_test.shape
X_train = X_train.reshape(-1, train_shape[3], train_shape[1], train_shape[2])
X_test = X_test.reshape(-1, test_shape[3], test_shape[1], test_shape[2])
print(X_train.shape)

nb_classes = 2
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

model = Sequential()

# 添加卷积层并训练
model.add(Conv2D(32, (5, 5), input_shape=(train_shape[3], train_shape[1], train_shape[2])))
model.add(Activation('tanh'))
model.add(MaxPooling2D(
    pool_size=(2, 2), 
    strides=(2, 2))
)

model.add(Conv2D(64, (5, 5)))
model.add(Activation('tanh'))
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2,2)
))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('tanh'))
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2,2)
))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('tanh'))
model.add(Dense(2))
model.add(Activation('sigmoid'))

adam = Adam(lr=1e-4)

model.compile(
    loss='binary_crossentropy',
    optimizer=adam,
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=1, shuffle=True, batch_size=64)

loss, accuracy = model.evaluate(X_test, y_test)
print("loss=", loss, "\naccuracy=", accuracy)

# if accuracy == 1.0:
# model1: 1层隐藏层，3次训练数据
# model2: 2层隐藏层，3层训练数据
# model3: 2层隐藏层，2层训练数据，效果最佳，但对于 LCD2L1XL 文件夹中的数据输出全1，其它两个文件夹输出准确率 99.8%
# model4: 2层隐藏层，1层训练数据，效果最差
model.save("Training model by keras4.h5")
print("model saved!")
