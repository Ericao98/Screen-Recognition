from keras import backend as K  
K.set_image_dim_ordering('th')

from keras.models import load_model, model_from_json
from keras.utils import to_categorical
import numpy as np
import cv2

model = load_model("Training model by keras3.h5")
X_train = np.load(r"..\Effective Screen\eff_small_move\numpy\x_train.npy")
y_train = np.load(r"..\Effective Screen\eff_small_move\numpy\y_train.npy")
X_test = np.load(r"..\Effective Screen\eff_small_move\numpy\x_test.npy")
y_test = np.load(r"..\Effective Screen\eff_small_move\numpy\y_test.npy")

nb_classes = 2
train_shape = X_train.shape
test_shape = X_test.shape
X_test = X_test.reshape(-1, test_shape[3], test_shape[1], test_shape[2])
y_test = to_categorical(y_test, nb_classes)
loss, accuracy = model.evaluate(X_test, y_test)
y_predict = model.predict_classes(X_test)
# print(y_predict)
# print(y_test)
print("loss=", loss, "\naccuracy=", accuracy)
