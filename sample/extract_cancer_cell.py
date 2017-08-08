#coding=utf-8
import cv2
import numpy as np
import glob
import sys
import os

def findMarginalPoint(contour):
    return (contour[:,:,1]).min(),(contour[:,:,1]).max(),(contour[:,:,0]).min(),(contour[:,:,0]).max()

def getCancerPartsFromImage(srcPath,labelPath):
    src = cv2.imread(srcPath,0)
    label = cv2.imread(labelPath,0)
    if(len(src) == 0 or len(label) == 0):
        print("Not enough src image or not enough lable image!")
        exit(1)
    if(np.shape(src) != np.shape(label)):
        print("assert \"src.width == label.widht && src.height == label.height \" failed!")
        exit(-1)

    (_, contours, _)=cv2.findContours(label,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    outputImages = []
    for contour in contours:
        left,right,top,bottom = findMarginalPoint(contour)
        if((left < 0 or top < 0) and (right >np.shape(src)[0] or bottom > np.shape(src)[1]) ):
            continue
        if(left == right or top == bottom):
            continue
        tempSrc = src.copy()
        tempImage = np.zeros(np.shape(src))
        cv2.drawContours(tempImage,[contour],0,255,-1)
        for i in range(len(tempImage)):
            for j in range(len(tempImage[i])):
                if(tempImage[i][j] == 0):
                    tempSrc[i][j] = 0
        outputImages.append(tempSrc[left:right,top:bottom])
        cv2.imshow("contourImage",tempSrc[left:right,top:bottom])
        cv2.waitKey(1)
    return outputImages

def getCancerPartsFromBatchImages(folderPath,outputPath="",writeToFile = True):
    imgPathes = glob.glob(folderPath+"//img//*")
    labelPathes = glob.glob(folderPath+"//label//*")
    if(len(imgPathes)==0 or len(labelPathes)==0):
        print(r"The folder structure should like this: floderPath\\img\\*.tif  floderpath\\label\\*.tif")
        return
    allCancerParts = []
    total = len(imgPathes)
    index = 1
    for imagePath,labelPath in zip(imgPathes,labelPathes):
        print("Dealing with the %dth/%d image..."%(index,total))
        index += 1
        allCancerParts.extend(getCancerPartsFromImage(imagePath,labelPath))
    if writeToFile:
        for index,img in enumerate(allCancerParts):
            cv2.imwrite(outputPath+"//"+str(index)+".tif",img)
    return allCancerParts
def main():
    argv = sys.argv
    if (len(argv) < 3):
        print("usage: python ExtractCancerCell.py folderPath outputPath writeToFile,")
        print(r"The folder structure should like this: floderPath\\img\\*.tif  floderpath\\label\\*.tif")
        exit(1)
    folderPath = argv[1]
    outputPath = argv[2]
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    getCancerPartsFromBatchImages(folderPath, outputPath, writeToFile=True)

if __name__ == "__main__":
    main()
    exit(0)