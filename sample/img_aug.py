import os, sys
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import numpy as np

def aug_img(img):
	res = [
		iaa.Crop(percent=(0, 0.1)).augment_image(img),
		iaa.Fliplr(0.5).augment_image(img),
		iaa.Flipud(0.5).augment_image(img),
		iaa.Add((-3, 3)).augment_image(img),
		iaa.Multiply((0.5, 1.5)).augment_image(img),
		iaa.GaussianBlur(sigma=(0, 0.5)).augment_image(img),
		iaa.ContrastNormalization((0.5, 1.0)).augment_image(img),
		iaa.Affine(
				scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
				translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
				order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
				rotate=(-45, 45),
				mode="reflect"
		).augment_image(img)
	]
	return res
	
def batch_aug(srcPrefix, dstPrefix):
	files = [f for f in os.listdir(srcPrefix) if f[-4:] == '.tif']
	print("[start] augment from {} imgs...".format(len(files)))
	count = 0
	for f in files:
		s = srcPrefix + '/' + f
		dstPrefix2 = dstPrefix + '/' + f[:-4]
		img = cv2.imread(s)
		if img is None:
			print(f, ' is broken...')
			continue
		if len(img.shape) != 3:
			print(f, ' is not 3 channels...')
			continue
		if img.shape[0] < 5 or img.shape[1] < 5:
			print(f, ' is too small...')
			continue
			
		augs = aug_img(np.transpose(img,(2,0,1)))
		cv2.imwrite(dstPrefix + '/' + f, img)
		for i, aug in enumerate(augs):
			trans_aug = np.transpose(aug,(1,2,0))
			d = dstPrefix2 + '-' + str(i) + '.tif'
			cv2.imwrite(d, trans_aug)
			count += 1
			if count % 500 == 0:
				print("  {} done ...".format(count))
	print("[success] generated %d images" % count)

def main(argv):
    if len(argv) != 3:
        print("usage: python img_aug.py srcPrefix dstPrefix")
        return
    for i in range(1, len(argv)):
        if argv[i][-1] == "/":
            argv[i] = argv[i][:-1]
            
    srcPrefix = argv[1]
    dstPrefix = argv[2]
    
    if not os.path.exists(srcPrefix):
        print("[error]", srcPrefix, " not existed!")
        return
    if not os.path.exists(dstPrefix):
        os.makedirs(dstPrefix)
        
    batch_aug(srcPrefix, dstPrefix)
    
    
if __name__ == "__main__":
    main(sys.argv)
	