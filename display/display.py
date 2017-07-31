from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

data = np.load("imgs_mask_test.npy")
print(data.shape)

imgs = []
plt.figure(dpi=150)
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
for i in range(data.shape[0]):
    imgs.append(array_to_img(data[i]))
    p1 = plt.subplot(330 + i*3 + 1)
    p2 = plt.subplot(330 + i*3 + 2)
    p3 = plt.subplot(330 + i*3 + 3)
                     
    p1.imshow(imgs[i])
    p2.imshow(cv2.imread("display/" + str(i) + ".tif"))
    p3.imshow(cv2.imread("display/label/" + str(i) + ".tif"))
                     
    p1.set_xticks([])
    p1.set_yticks([])
    p2.set_xticks([])
    p2.set_yticks([])
    p3.set_xticks([])
    p3.set_yticks([])
plt.savefig("results.jpg")