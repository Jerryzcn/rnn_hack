import os
import sys
import PIL 
import numpy as np
from PIL import Image
from numpy import newaxis
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def convt_img(lst):
	data = []
	for img in lst:
		o_img = Image.open(img)
		b_img = o_img.convert('L')
		r_img = b_img.resize((500,500), PIL.Image.ANTIALIAS)
		r_img.save('t_results.png')
		t_img = mpimg.imread('t_results.png')
		if len(t_img.shape) == 2: f_img = t_img[newaxis,:,:]
		else: f_img = t_imgc[:]
		data.append(f_img)
		print '---> image_shape', f_img.shape    
	data = np.array(data)
	print '---> Writing', data.shape
	return data


def get_files():
	lst = []
	for i in os.listdir(os.getcwd()):
		if i.endswith('png'): lst.append(i)
	return lst


# #GetFiles
# file_lst = get_files()
# #Convert To Numpy
# data = convt_img(file_lst)



