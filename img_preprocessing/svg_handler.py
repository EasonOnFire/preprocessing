# -*- coding:utf-8 -*-
from xml.dom.minidom import parse
import xml.dom.minidom
import cv2
import numpy as np
import os, sys

# input: svg文件路径
# output: list， 每个元素为m*n*2的三维np.array，m为轮廓数，n为每个轮廓的描点数，2为xy坐标
def decodeSvg(path):
	DOMTree = parse(path)	
	collection = DOMTree.documentElement

	polys = collection.getElementsByTagName("polygon")
	res = []
	for poly in polys:	
		if poly.hasAttribute("points"):
			str = poly.getAttribute("points")
			cords = str.split(" ")
			resCords = []
			for cord in cords:
				xy = cord.split(",")
				resCords.append([int(xy[0]), int(xy[1])])
			if len(resCords) > 0:
				res.append(np.array([resCords]))
		else:
			print(path + " has no polygon!")
	return res


# input: svg文件路径， jpg输出文件路径， 图像大小如[2048， 2048]
# example: drawFromSvg("1.svg", "1.jpg", [2048, 2048])
def drawFromSvg(src, dst, shape):
    contours = decodeSvg(src)
    for c in contours:
        if os.path.isfile(dst):
            res = cv2.imread(dst, 0)
        else:
            res = np.zeros((shape[0], shape[1]), np.uint8)
        cv2.fillPoly(res, contours, 255)
        cv2.imwrite(dst, res)

		
# input: svg文件路径， tif输出文件路径
def batchDrawFromSvg(svgPrefix, tifPrefix):
    svgs = [f for f in os.listdir(svgPrefix) if f[-4:] == ".svg"]
    total = len(svgs)
    print("[start] %d svg files ready to transform..." % total)

    count = 0
    for s in svgs:
        count += 1
        src = svgPrefix + "/" + s
        dst = tifPrefix + "/" + s[:-3] + "tif"
    #     print(src)
        drawFromSvg(src, dst, [2048, 2048])

        if count % 100 == 0: print("  {}/{} finished...".format(count, total))
    print("[finish] %d tif files has been transfered..." % count)


def main(argv):
	if len(argv) != 3:
		print("usage: python svg_handler.py svgPrefix tifPrefix")
		return 

	svgPrefix = argv[1]
	tifPrefix = argv[2]
	if svgPrefix[-1] == "/":
		svgPrefix = svgPrefix[:-1]
	if tifPrefix[-1] == "/":
		tifPrefix = tifPrefix[:-1]
		
	if not os.path.exists(svgPrefix):
		print(svgPrefix + " is not exist!")
		return
	if not os.path.exists(tifPrefix):
		os.makedirs(tifPrefix)
	
	batchDrawFromSvg(svgPrefix, tifPrefix)

if __name__ == "__main__":
	main(sys.argv)
