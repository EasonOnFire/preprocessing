#!/usr/bin/env python
# -*- coding: utf-8 -*-
# blend 2 imgs with given position

import numpy as np
import scipy.sparse
import PIL.Image
import pyamg
from random import randint

def poisson_blend_core(t, s, A, P, region_size, img_target, img_mask):
	# get subimages
    t = t.flatten()
    s = s.flatten()

    # create b
    b = P * s
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if not img_mask[y,x]:
                index = x+y*region_size[1]
                b[index] = t[index]

    # solve Ax = b
    x = pyamg.solve(A,b,verb=False,tol=1e-10)

    # assign x to target image
    x = np.reshape(x, region_size)
	# 
    x[img_mask != 0] /= (np.max(x)/255.0)
    #x[x>256] = 255
    x[x<0] = 0
    x = np.array(x, img_target.dtype)
    return x

# blend source with target, left top position offset(x, y)
def poisson_blend(img_target, img_source, img_mask, offset=(0, 0)):
    # compute regions to be blended
	offx = offset[0]
	offy = offset[1]
	region_source = (
			max(-offy, 0),
            max(-offx, 0),
            min(img_target.shape[0]-offy, img_source.shape[0]),
            min(img_target.shape[1]-offx, img_source.shape[1]))
	region_target = (
            max(offy, 0),
            max(offx, 0),
            min(img_target.shape[0], img_source.shape[0]+offy),
            min(img_target.shape[1], img_source.shape[1]+offx))
	region_size = (region_source[2]-region_source[0], region_source[3]-region_source[1])

	#print('region_source', region_source)
	#print('region_target', region_target)
	#print('region_size', region_size)
	
    # clip and normalize mask image

	img_mask = None
	if len(img_source.shape) == 3:
		img_mask = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3], 0] > 10
	else:
		img_mask = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3]] > 10	
	img_mask = img_mask.astype(np.uint8)
	print(img_mask)
	PIL.Image.fromarray(np.uint8(img_mask*255)).save("poissonblending-demo/testimages/mask.jpg")
	"""
	img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
	if len(img_mask.shape) > 2:
		img_mask = img_mask[:,:,0]
	img_mask[img_mask==0] = False
	img_mask[img_mask!=False] = True
	print(img_mask.dtype)
	"""
    # create coefficient matrix
	A = scipy.sparse.identity(np.prod(region_size), format='lil')
	for y in range(region_size[0]):
		for x in range(region_size[1]):
			if img_mask[y,x]:
				index = x+y*region_size[1]
				A[index, index] = 4
				if index+1 < np.prod(region_size):
					A[index, index+1] = -1
				if index-1 >= 0:
					A[index, index-1] = -1
				if index+region_size[1] < np.prod(region_size):
					A[index, index+region_size[1]] = -1
				if index-region_size[1] >= 0:
					A[index, index-region_size[1]] = -1
	A = A.tocsr()
    
    # create poisson matrix for b
	P = pyamg.gallery.poisson(img_mask.shape)

    # for each layer (ex. RGB)
	if len(img_target.shape) == 3:
		for num_layer in range(img_target.shape[2]):
			# get subimages
			t = img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer]
			s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3],num_layer]
			x = poisson_blend_core(t, s, A, P, region_size, img_target, img_mask)
			img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer] = x
	elif len(img_target.shape) == 2:
		# get subimages
		t = img_target[region_target[0]:region_target[2],region_target[1]:region_target[3]]
		s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3]]
		x = poisson_blend_core(t, s, A, P, region_size, img_target, img_mask)
		img_target[region_target[0]:region_target[2],region_target[1]:region_target[3]] = x
	return img_target

def random_blend_by_path(path_target, path_source, path_output):
    img_source = np.asarray(PIL.Image.open(path_source))
    img_source.flags.writeable = True
    img_target = np.asarray(PIL.Image.open(path_target))
    img_target.flags.writeable = True
    
	# test only
    img_mask = np.asarray(PIL.Image.open("poissonblending-demo/testimages/mask.jpg"))
    img_mask.flags.writeable = True
    print(len(img_source.shape), "channels")
    if len(img_source.shape) not in (2,3) or len(img_target.shape) not in (2,3):
        return "[err] img broken!..."
	
	# calculate random offset
    xrange = img_target.shape[1]-img_source.shape[1]-1
    yrange = img_target.shape[0]-img_source.shape[0]-1
    if xrange <= 0 or yrange <= 0:
        return "[err] sourceImg bigger than targetImg"
	
    offx = randint(0, xrange)
    offy = randint(0, yrange)
	
    img_ret = poisson_blend(img_target, img_source, img_mask, offset=(offx, offy))
    img_ret = PIL.Image.fromarray(np.uint8(img_ret))
    img_ret.save(path_output)
    return None

if __name__ == '__main__':
    print("This is an interface.")
    path_target = "poissonblending-demo/testimages/1.tif"
    path_source = "poissonblending-demo/testimages/2.tif"
    path_output = "poissonblending-demo/testimages/3.tif"
    ret = random_blend_by_path(path_target, path_source, path_output)
    if ret is not None:
        print(ret)
    else:
        print("[success] ...")
