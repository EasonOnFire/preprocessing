import cv2
import os, sys

def tiff2tif(src, dst):
    ffs = [f for f in os.listdir(src) if f[-5:] == ".tiff"]
    total = len(ffs)
    print("[start] %d tiff files ready to transform..." % total)
    
    count = 0
    for ff in ffs:
        count += 1
        s = src + "/" + ff
        d = dst + "/" + ff[:-4] + "tif"
#         print("", s, "\n", d)
        temp =  cv2.imread(s)
        if temp is None:
            continue
        cv2.imwrite(d, temp)
        if count % 100 == 0: print("  {}/{} finished...".format(count, total))
    print("[finish] %d tif files has been transfered..." % count)
	
# in case we need this transformer
def jpeg2tif(src, dst):
    ffs = [f for f in os.listdir(src) if f[-5:] == ".jpeg"]
    total = len(ffs)
    print("[start] %d jpeg files ready to transform..." % total)
    
    count = 0
    for ff in ffs:
        count += 1
        s = src + "/" + ff
        d = dst + "/" + ff[:-4] + "tif"
#         print("", s, "\n", d)
        temp =  cv2.imread(s)
        if temp is None:
            continue
        cv2.imwrite(d, temp)
        if count % 100 == 0: print("  {}/{} finished...".format(count, total))
    print("[finish] %d tif files has been transfered..." % count)
	
def main(argv):
	if len(argv) != 3:
		print("usage: python tiff_handler.py tiffPrefix tifPrefix")
		return 

	tiffPrefix = argv[1]
	tifPrefix = argv[2]
	if tiffPrefix[-1] == "/":
		tiffPrefix = tiffPrefix[:-1]
	if tifPrefix[-1] == "/":
		tifPrefix = tifPrefix[:-1]
		
	if not os.path.exists(tiffPrefix):
		print(tiffPrefix + " is not exist!")
		return
	if not os.path.exists(tifPrefix):
		os.makedirs(tifPrefix)
	
	tiff2tif(tiffPrefix, tifPrefix)

if __name__ == "__main__":
	main(sys.argv)