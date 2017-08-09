import os, sys
import random
import numpy as np
import shutil
from poisson_interface import random_blend_by_path

def loadData(cancerPrefix, bgPrefix):
    cancerFiles = [cancerPrefix+'/'+f for f in os.listdir(cancerPrefix) if f[-4:] == '.tif']
    bgFiles = [bgPrefix+'/'+f for f in os.listdir(bgPrefix) if f[-4:] == '.tif']
    print('[loadData] load cancerFile: {}, bgFile: {}'.format(len(cancerFiles), len(bgFiles)))
    return cancerFiles, bgFiles

def generate(cancerFiles, bgFiles, posPath, negPath, maxNbCancer, maxAreaRate):
	# generate negative sample
	negBgPath = bgFiles[random.randint(0, len(bgFiles)-1)]
	shutil.copyfile(negBgPath, negPath)
	
	# generate positive sample
	nbCancer = random.randint(0, maxNbCancer)
	bgPath = bgFiles[random.randint(0, len(bgFiles)-1)]
	cancerCount = len(cancerFiles)

	for i in range(nbCancer):
		cancerPath = cancerFiles[random.randint(0, cancerCount-1)]
		random_blend_by_path(bgPath, cancerPath, posPath)
		bgPath = posPath # alter on the same position
    
# main args
def main(argv):
    if len(argv) != 7:
        print("usage: python sample_generatorV2.py cancerPrefix bgPrefix posPrefix negPrefix maxNbCancer nbGenerate")
        return
    for i in range(1, len(argv)):
        if argv[i][-1] == "/":
            argv[i] = argv[i][:-1]
            
    cancerPrefix = argv[1]
    bgPrefix     = argv[2]
    posPrefix    = argv[3]
    negPrefix    = argv[4]
    maxNbCancer  = int(argv[5])
    nbGenerate   = int(argv[6])
    
    if not os.path.exists(cancerPrefix):
        print("[error]", cancerPrefix, " not existed!")
        return
    if not os.path.exists(bgPrefix):
        print("[error]", bgPrefix, " not existed!")
        return
    if not os.path.exists(posPrefix):
        os.makedirs(posPrefix)
    if not os.path.exists(negPrefix):
        os.makedirs(negPrefix)
        
    cancerFiles, bgFiles = loadData(cancerPrefix, bgPrefix)
    print("[start] generating samples")
    for i in range(nbGenerate):
        posPath = posPrefix + '/' + str(i) + '.tif'
        negPath = negPrefix + '/' + str(i) + '.tif'
        generate(cancerFiles, bgFiles, posPath, negPath, maxNbCancer, 0.6)
        if i%100 == 0:
            print("  {}/{}".format(i+1, nbGenerate))
    print("[success] generated %d samples" % nbGenerate)
    
    
if __name__ == "__main__":
    main(sys.argv)