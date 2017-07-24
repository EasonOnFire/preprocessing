# -*- coding:utf-8 -*-
import cv2
import numpy as np
import random
import copy
import sys, os

def warper(parts, bias, imgPath, dstPath):    
    def getIdx(x, y):
        return (parts+1)*y + x
    def getCorner(pts, i, j):
        res = [pts[getIdx(i, j)],   pts[getIdx(i, j+1)],
              pts[getIdx(i+1, j)], pts[getIdx(i+1, j+1)]]
    #     print(res)
        return res
    def getImagePart(img, i, j):
        rows,cols,ch = img.shape
        xpart = int(cols/parts)
        ypart = int(rows/parts)
        return img[i*xpart:(i+1)*xpart+1, j*ypart:(j+1)*ypart+1]

    img = cv2.imread(imgPath)
    rows,cols,ch = img.shape

    srcPts = []
    for i in range(parts+1):
        for j in range(parts+1):
            srcPts.append([cols*i/parts, rows*j/parts])
    dstPts = copy.deepcopy(srcPts)
    icount = 0
    for i in range(1,parts):
        for j in range(1,parts):
            idx = getIdx(i, j)
            temp = dstPts[idx]
            dstPts[idx] = [temp[0]+bias[icount][0], temp[1]+bias[icount][1]]
            icount += 1
    # print(dstPts)

    def warperPart(i, j):
        partOringin = getImagePart(img, i, j)
        h, w, c = partOringin.shape
        pts1 = np.float32([[0,0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32(getCorner(dstPts, i, j))
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(partOringin, M, (cols, rows))
        return dst

    result = np.zeros(img.shape, np.uint8)
    for i in range(parts):
        for j in range(parts):
            result = cv2.add(warperPart(i, j), result)

    cv2.imwrite(dstPath, result)

# input: 原始tif文件路径， 输出扭曲tif文件路径
def batchWarper(srcPrefix, srcLabelPrefix, dstPrefix, dstLabelPrefix, nameSuffix):
    tifs = [f for f in os.listdir(srcPrefix) if f[-4:] == ".tif"]
    total = len(tifs)
    print("[start] %d tif files ready to warper ..." % total)
	
    count = 0
    parts = 10
    bias_level = 8
    def getBias():
        return random.randint(-bias_level, bias_level)
    for s in tifs:
        count += 1
        src = srcPrefix + "/" + s
	srcLabel = srcLabelPrefix + "/" + s
        dst = dstPrefix + "/" + s[:-3] + nameSuffix + ".tif"
	dstLabel = dstLabelPrefix + "/" + s[:-3] + nameSuffix + ".tif"
	# print(src)
	# create bias
	bias = []
	for i in range(1,parts):
		for j in range(1,parts):
			bias.append([getBias(), getBias()])
	warper(parts, bias, src, dst)
	warper(parts, bias, srcLabel, dstLabel)

        if count % 100 == 0: print("  {}/{} finished...".format(count, total))
    print("[finish] %d tif files has been warpered..." % count)


def main(argv):
	if len(argv) != 6:
		print("usage: python warper.py srcPrefix srcLabelPrefix dstPrefix dstLabelPrefix nameSuffix")
		return 

	srcPrefix = argv[1]
	srcLabelPrefix = argv[2]
	dstPrefix = argv[3]
	dstLabelPrefix = argv[4]
	if srcPrefix[-1] == "/":
		srcPrefix = srcPrefix[:-1]
	if srcLabelPrefix[-1] == "/":
		srcLabelPrefix = srcLabelPrefix[:-1]
	if dstPrefix[-1] == "/":
		dstPrefix = dstPrefix[:-1]
	if dstLabelPrefix[-1] == "/":
		dstLabelPrefix = dstLabelPrefix[:-1]
		
	if not os.path.exists(srcPrefix):
		print(srcPrefix + " is not exist!")
		return
	if not os.path.exists(srcLabelPrefix):
		print(srcLabelPrefix + " is not exist!")
		return
	if not os.path.exists(dstPrefix):
		os.makedirs(dstPrefix)
	if not os.path.exists(dstLabelPrefix):
		os.makedirs(dstLabelPrefix)

	nameSuffix = argv[5]
	
	batchWarper(srcPrefix, srcLabelPrefix, dstPrefix, dstLabelPrefix, nameSuffix)

if __name__ == "__main__":
	main(sys.argv)
