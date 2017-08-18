import cv2
import sys, os
import numpy as np

# left eyebrow index : 174~193; right : 154~173
def renderPoints(img, points):
	for i, p in enumerate(points):
		cv2.putText(img, str(i), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
	# show image
	cv2.imshow("shape", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
def renderLabel(img, points):
	label = np.zeros((img.shape[0], img.shape[1]), np.int32)
	pts = [points[154:174], points[174:194]]
	pts = np.array(pts, np.int32)
	# print(pts)
	cv2.fillPoly(label, pts, 255)
	return label

def render(imgPrefix, annoPath, func, tarPrefix=''):
	imgSuffix = ''
	points = []
	with open(annoPath) as a:
		for line in a:
			line = line.replace(' ','').rstrip()
			if '' == imgSuffix:
				imgSuffix = line
			else:
				points.append([float(p) for p in line.split(',')])
	imgPath = imgPrefix + '/' + imgSuffix + '.jpg'
	tarPath = tarPrefix + '/' + imgSuffix + '.jpg'
	
	img = cv2.imread(imgPath)
	ret = func(img, points)
	if '' != tarPrefix:
		cv2.imwrite(tarPath, ret)
	
if __name__ == '__main__':
	imgPrefix = '../data/img_total'
	annoPrefix = '../data/annotation'
	labelPrefix = '../data/label_total'
	renderPath = 'render.jpg'
	
	annoFiles = os.listdir(annoPrefix)
	annoPath = annoPrefix + '/' + annoFiles[0]
	
	# draw points for observation
	#render(imgPrefix, annoPath, renderPoints) 
	
	# generate labels
	total = len(annoFiles)
	print('[start] {} label to be generated...]'.format(total))
	count = 0
	for a in annoFiles:
		count += 1
		annoPath = annoPrefix + '/' + a;
		render(imgPrefix, annoPath, renderLabel, labelPrefix) 
		if count % 100 == 0:
			print('  {}/{}'.format(count, total))
			
	print('[success] render finished!')