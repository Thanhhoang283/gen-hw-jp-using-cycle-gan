import numpy as np
import PIL
import cv2 
import os 
import shutil 
import Image 
from matplotlib import pyplot as plt 
import skimage.util as util 
from grid_distortion import warp_image 

def random_resize(img):
    height, width = np.shape(img)
    # plt.imshow(img)
    # plt.show()
    # scale = np.random.randint(1, 3)
    scale = np.random.uniform(1.0, 1.5)
    rand_height = int(height/float(scale))
    rand_width = int(width/float(scale))
    new_shape = (rand_width, rand_height)
    # print("New shape: {}".format(new_shape))
    new_im = cv2.resize(img, new_shape)

    # h = int((height - rand_height)/np.random.randint(1, 3))
    h = int((height - rand_height)/float(np.random.uniform(1.0, 2.0)))
    # print("H: {}".format(h))
    # w = int((width - rand_width)/np.random.randint(1, 5))

    w_stack = np.vstack((np.array(new_im), np.ones((h, rand_width))*255))
    w_stack = np.vstack((np.ones(((height - rand_height - h), rand_width))*255, np.array(w_stack)))

    # plt.imshow(w_stack)
    # plt.show()
    # h_stack = np.hstack((np.array(w_stack), np.ones((height, w))*255))
    # h_stack = np.hstack((np.ones((height, width - rand_width - w))*255, np.array(h_stack)))

    # resized_img = h_stack
    # plt.imshow(w_stack)
    # plt.show()
    # print(np.shape(w_stack))
    return w_stack

def add_bg_noise(image, method):
	float_img = util.img_as_float(image)
	noise_img = util.random_noise(float_img, mode=method)
	# plt.imshow(noise_img)
	# plt.show()
	return noise_img

def add_bg(background, foreground):
	merge = cv2.addWeighted(foreground, 0.3, background, 0.7, 0)
	plt.imshow(merge)
	plt.show()
	# return merge

def random_touching(img1, img2):
	f1 = 255-img1
	f2 = 255-img2

	val = np.random.randint(int(np.shape(f2)[1]/2), np.shape(f2)[1])
	if (np.shape(f1)[1] > np.shape(f2)[1]):
		f3 = np.zeros((np.shape(f1)[0], np.shape(f1)[1]+val))
	else:
		f3 = np.zeros((np.shape(f1)[0], np.shape(f2)[1]+val))
	f3[:np.shape(f1)[0], :np.shape(f1)[1]] = f1 
	f3[:np.shape(f1)[0], val:np.shape(f2)[1]+val] += f2
	# plt.imshow(f3)
	# plt.show()
	return 255-f3

def combine_images(list_im, name, background, savefolder):
	rect_imgs = []
	for im in list_im:
		x, y, width, height = cv2.boundingRect(255-np.asarray(im))
		rect_imgs.append(im[y:y+height, x:x+width])

	# pick the image which is the smallest, and resize the others to match it
	# min_shape = sorted([(np.sum(np.shape(i)), np.shape(i)) for i in rect_imgs])[0][1]
	min_shape = sorted([(np.shape(i)[0], np.shape(i)) for i in rect_imgs])[0][1]
	print("Min shape: {}".format(min_shape))
	rect_imgs_resized = [cv2.resize(img, (np.shape(img)[1], min_shape[0])) for img in rect_imgs]

	stack_imgs = []
	prev = []
	for img in rect_imgs_resized:
		padding = np.ones((min_shape[0], np.random.randint(20)))*255
		if (len(prev) != 0):
			if (np.random.randint(100) <= 80):
				tmp = np.hstack((np.asarray(img), padding))
			else:
				tmp = random_touching(img, prev)
		else:
			tmp = np.hstack((np.asarray(img), padding))

		stack_imgs.append(random_resize(tmp))
		prev = img

	imgs_comb = np.hstack((np.asarray(i)) for i in stack_imgs)
	imgs_comb = PIL.Image.fromarray((imgs_comb).astype(np.uint8))
	imgs_comb.save(os.path.join(savefolder, 'tmp.png'))

	im = cv2.imread(os.path.join(savefolder, 'tmp.png'))
	# im = cv2.cvtColor(imgs_comb, cv2.COLOR_GRAY2BGR)
	new_im = warp_image(im, draw_grid_lines=False)
	# cv2.imwrite(os.path.join(savefolder, "non_background_{}.png".format(name)), new_im)
	# print(np.shape(new_im))
	# plt.imshow(new_im)
	# plt.show()
	methods = ['gaussian', 'localvar', 'poisson', 'salt', 's&p', 'speckle']
	backgrounds = os.listdir(background)
	# print(backgrounds)
	bg_im = cv2.imread(os.path.join(background, backgrounds[np.random.randint(len(backgrounds))]))
	bg_im = cv2.resize(bg_im, np.shape(new_im)[:2][::-1])
	merge = bg_im + new_im
	noise_img = add_bg_noise(merge, methods[np.random.randint(len(methods))])
	# plt.imshow(noise_img)
	# plt.show()
	cv2.imwrite(os.path.join(savefolder, '{}.png'.format(name)), (noise_img*255).astype(np.uint8))
	os.remove(os.path.join(savefolder, 'tmp.png'))