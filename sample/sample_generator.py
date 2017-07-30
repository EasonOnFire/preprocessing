import os, sys
import cv2
import random
import numpy as np

def loadData(cancerPrefix, bgPrefix):
    cancerFiles = [f for f in os.listdir(cancerPrefix) if f[-4:] == '.tif']
    bgFiles = [f for f in os.listdir(bgPrefix) if f[-4:] == '.tif']
    print('[loadData] load cancerFile: {}, bgFile: {}'.format(len(cancerFiles), len(bgFiles)))
    
    cancerImgs = []
    for f in cancerFiles:
        s = cancerPrefix + "/" + f
        if not os.path.isfile(s):
            print("img not existed ...")
            break
        temp = cv2.imread(s, 0)
        cancerImgs.append(temp)
    
    bgImgs = []
    for f in bgFiles:
        s = bgPrefix + "/" + f
        if not os.path.isfile(s):
            print("img not existed ...")
            break
        temp = cv2.imread(s, 0)
        bgImgs.append(temp)
        
    print('[loadData] load cancerImg: {}, bgImg: {}'.format(len(cancerImgs), len(bgImgs)))    
    return cancerImgs, bgImgs

def generate(cancerImgs, bgImgs, imgPath, labelPath, maxNbCancer, maxAreaRate):
    nbCancer = random.randint(0, maxNbCancer)
    bgImg = bgImgs[random.randint(0, len(bgImgs)-1)].copy()
    bgRows, bgCols = bgImg.shape
    labelImg = np.zeros((bgRows, bgCols), np.uint8)
    
    areaAvailable = bgImg.size*maxAreaRate
    cancerCount = len(cancerImgs)

    for i in range(nbCancer):
        cancerImg = cancerImgs[random.randint(0, cancerCount-1)]
        areaAvailable -= cancerImg.size
        if areaAvailable < 0:
            break
            
        # binary mask
        _, cancerMask = cv2.threshold(cancerImg, 15, 255, cv2.THRESH_BINARY)
        # randomly location cancer img
        cancerRows, cancerCols = cancerImg.shape[0:2]
        randR = random.randint(0, bgRows-cancerRows-1)
        randC = random.randint(0, bgCols-cancerCols-1)

        # modify sample img
        bgImg[randR:randR+cancerRows, randC:randC+cancerCols] = cv2.bitwise_and(bgImg[randR:randR+cancerRows, randC:randC+cancerCols], 255-cancerMask)
        cancerImg = cv2.bitwise_and(cancerImg, cancerMask)
        bgImg[randR:randR+cancerRows, randC:randC+cancerCols] += cancerImg
        # modify label
        labelImg[randR:randR+cancerRows, randC:randC+cancerCols] = cv2.bitwise_or(labelImg[randR:randR+cancerRows, randC:randC+cancerCols], cancerMask)
        
    cv2.imwrite(imgPath, bgImg)
    cv2.imwrite(labelPath, labelImg)
    
# main args
def main(argv):
    if len(argv) != 7:
        print("usage: python sample_generator.py cancerPrefix bgPrefix imgPrefix labelPrefix maxNbCancer nbGenerate")
        return
    for i in range(1, len(argv)):
        if argv[i][-1] == "/":
            argv[i] = argv[i][:-1]
            
    cancerPrefix = argv[1]
    bgPrefix     = argv[2]
    imgPrefix    = argv[3]
    labelPrefix  = argv[4]
    maxNbCancer  = int(argv[5])
    nbGenerate   = int(argv[6])
    
    if not os.path.exists(cancerPrefix):
        print("[error]", cancerPrefix, " not existed!")
        return
    if not os.path.exists(bgPrefix):
        print("[error]", bgPrefix, " not existed!")
        return
    if not os.path.exists(imgPrefix):
        os.makedirs(imgPrefix)
    if not os.path.exists(labelPrefix):
        os.makedirs(labelPrefix)
        
    cancerImgs, bgImgs = loadData(cancerPrefix, bgPrefix)
    for i in range(nbGenerate):
        imgPath = imgPrefix + '/' + str(i) + '.tif'
        labelPath = labelPrefix + '/' + str(i) + '.tif'
        generate(cancerImgs, bgImgs, imgPath, labelPath, maxNbCancer, 0.6)
    print("[success] generated %d samples" % nbGenerate)
    
    
if __name__ == "__main__":
    main(sys.argv)