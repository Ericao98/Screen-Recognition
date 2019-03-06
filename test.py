from keras import backend as K  
K.set_image_dim_ordering('th')

import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r'D:\Machine Learning\Effective Screen\eff_small_move\1\LCD2L1XL018-11-23-12-21-010.jpg')
# img = cv2.imread(r'D:\Machine Learning\Effective Screen\eff_small_move\1\LCD4L1XL1018-11-25-15-26-010.jpg', cv2.IMREAD_GRAYSCALE)
    
# _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY )
# edges = cv2.Canny(thresh, 10, 20)
# edges = cv2.Canny(img, 10, 15)
# blurred = cv2.GaussianBlur(img,(3,3),0)
print(img)
# print(blurred)
gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
canny = cv2.Canny(blurred, 10, 15)

# img = cv2.imread(r'D:\Machine Learning\Effective Screen\eff_small_move\1\LCD2L1XL018-11-23-12-21-010.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(canny, cmap='gray')
# img = cv2.imread(r'D:\Machine Learning\Effective Screen\eff_small_move\1\LCD2L1XL018-11-23-12-21-010.jpg')
plt.show()
# plt.imshow(blurred, cmap='gray')
# plt.show()
# plt.imshow(canny, cmap='gray')
# plt.show()
