import cv2
import os, sys
import numpy as np

def countPositive(labelPrefix):
	fs = [f for f in os.listdir(labelPrefix) if f[-4:] == ".tif"]
	print("[start] %d tif files ready to count..." % len(fs))
	
	positive = 0
	total = 0
	for f in fs:
		temp =  cv2.imread(labelPrefix+"/"+f)
		if temp is None:
			continue
		total += temp.size
		arr = np.array(temp)
		positive += (np.sum(arr)/255)
        
	print("[finish] positive sample percentage is {}%...".format(positive*100.0/total))
	
def main(argv):
	if len(argv) != 2:
		print("usage: python count_positive.py labelPrefix")
		return 

	labelPrefix = argv[1]
	if labelPrefix[-1] == "/":
		labelPrefix = labelPrefix[:-1]
		
	if not os.path.exists(labelPrefix):
		print(labelPrefix + " is not exist!")
		return

	countPositive(labelPrefix)

if __name__ == "__main__":
	main(sys.argv)