# -*- coding:utf-8 -*-
import os, sys
import cv2
import numpy as np

# 填补没有标注的非癌症图像
def fillNoncancer(labelPrefix, imgPrefix, shape):
    count = 0
    emptyImg = np.zeros((shape[0], shape[1]), np.uint8)
    labels = [f for f in os.listdir(labelPrefix) if f[-4:] == ".tif"]
    imgs = [f for f in os.listdir(imgPrefix) if f[-4:] == ".tif"]
    for f in imgs:
        if f not in labels:
            cv2.imwrite(labelPrefix+"/"+f, emptyImg)
            count += 1
    print("[finish] %d noncancer labels has been filled ..." % count)
	
def main(argv):
	if len(argv) != 3:
		print("usage: python fill_noncancer.py labelPrefix imgPrefix")
		return 

	labelPrefix = argv[1]
	imgPrefix = argv[2]
	if labelPrefix[-1] == "/":
		labelPrefix = labelPrefix[:-1]
	if imgPrefix[-1] == "/":
		imgPrefix = imgPrefix[:-1]
		
	if not os.path.exists(labelPrefix):
		print(labelPrefix + " is not exist!")
		return
	if not os.path.exists(imgPrefix):
		print(imgPrefix + " is not exist!")
		return
	
	fillNoncancer(labelPrefix, imgPrefix, [2048, 2048])

if __name__ == "__main__":
	main(sys.argv)
