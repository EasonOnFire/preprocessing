#!/usr/bin/env python
# -*- coding: utf-8 -*-
# blend 2 imgs with given position

import numpy as np
import scipy.sparse
import PIL.Image
import pyamg
		

def poisson_blend(img_target, img_source, offset=(0, 0)):
	pass

def blend(img_target, img_source, img_mask, offset=(0, 0)):
	# offset : (40, -30)
    # compute regions to be blended
	offx = offset[1]
	offy = offset[0]
	region_source = (
			max(offx, 0),
            max(offy, 0),
            min(img_target.shape[0]+offx, img_source.shape[0]),
            min(img_target.shape[1]+offy, img_source.shape[1]))
	region_target = (
            max(offx, 0),
            max(offy, 0),
            min(img_target.shape[0], img_source.shape[0]+offx),
            min(img_target.shape[1], img_source.shape[1]+offy))
	region_size = (region_source[2]-region_source[0], region_source[3]-region_source[1])

	print('region_source', region_source)
	print('region_target', region_target)
	print('region_size', region_size)
	
    # clip and normalize mask image
	
	img_mask = img_source[:,:,0] > 0
	img_mask = img_mask.astype(np.uint8)
	print(img_mask)
	"""
	img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
	if len(img_mask.shape) > 2:
		img_mask = img_mask[:,:,0]
	img_mask[img_mask==0] = False
	img_mask[img_mask!=False] = True
	print(img_mask)
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
	for num_layer in range(img_target.shape[2]):
        # get subimages
		t = img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer]
		s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3],num_layer]
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
		x[x>255] = 255
		x[x<0] = 0
		x = np.array(x, img_target.dtype)
		img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer] = x

	return img_target

def test():
    prefix = 'poissonblending-demo/testimages/'
    img_mask = np.asarray(PIL.Image.open(prefix + 'mask.tif'))
    img_mask.flags.writeable = True
    img_source = np.asarray(PIL.Image.open(prefix + 'src.tif'))
    img_source.flags.writeable = True
    img_target = np.asarray(PIL.Image.open(prefix + 'test1_target.png'))
    img_target.flags.writeable = True
    img_ret = blend(img_target, img_source, img_mask, offset=(0, 0))
    img_ret = PIL.Image.fromarray(np.uint8(img_ret))
    img_ret.save(prefix + 'test1_ret.png')


if __name__ == '__main__':
    test()
