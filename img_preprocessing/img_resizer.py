import os, sys
import cv2

def resizer(src, dst, shape):
    imgs = [f for f in os.listdir(src) if f[-4:] == ".tif"]
    print("[start] %d tif files ready to resize ..." % len(imgs))
    count = 0
    total = len(imgs)
    for f in imgs:
	count += 1
        s = src + "/" + f
        d = dst + "/" + f
        if not os.path.isfile(s):
            print("img not existed ...")
            break
        temp = cv2.imread(s, 0)
        temp = cv2.resize(temp, shape)
        cv2.imwrite(d, temp)
	if count % 100 == 0: print("  {}/{} finished...".format(count, total))
    print("[finish] %d tif files has been resized ..." % len(imgs))
	
def main(argv):
	if len(argv) != 4:
		print("usage: python img_resizer.py srcPrefix dstPrefix size(int)")
		return 

	srcPrefix = argv[1]
	dstPrefix = argv[2]
	size = int(argv[3])
	if srcPrefix[-1] == "/":
		srcPrefix = srcPrefix[:-1]
	if dstPrefix[-1] == "/":
		dstPrefix = dstPrefix[:-1]
		
	if not os.path.exists(srcPrefix):
		print(srcPrefix + " is not exist!")
		return
	if not os.path.exists(dstPrefix):
		os.makedirs(dstPrefix)
	
	resizer(srcPrefix, dstPrefix, (size, size))

if __name__ == "__main__":
	main(sys.argv)
