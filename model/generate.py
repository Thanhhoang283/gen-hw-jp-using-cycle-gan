# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import sys
import numpy as np
import scipy.misc
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import collections
import random
import shutil
from matplotlib import pyplot as plt
import cv2
from combine_images import combine_images
reload(sys)
sys.setdefaultencoding("utf-8")


def draw_single_char(ch, font, canvas_size=128, x_offset=0, y_offset=0):
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img

def resize_image(img):
    # pad to square
    pad_size = int(abs(img.shape[0]-img.shape[1]) / 2)
    if img.shape[0] < img.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size, pad_size))
    img = np.lib.pad(img, pad_dims, mode='constant', constant_values=255)
    # resize
    img = scipy.misc.imresize(img, (128, 128))
    assert img.shape == (128, 128)
    # img = scipy.misc.imresize(img, (256, 256))
    # assert img.shape == (256, 256)
    return img


def main(line_th, text, source_font, model, save_directory):
    path = os.getcwd()
    # datafolder = os.path.join(os.path.normpath(path + os.sep + os.pardir), 'printed_images')
    printed_folder = os.path.join(path, 'printed_images')
    if os.path.exists(printed_folder):
        shutil.rmtree(printed_folder)
        # os.makedirs(savefolder)
    os.makedirs(printed_folder)

    hw_folder = os.path.join(path, 'hw_images')
    if os.path.exists(hw_folder):
        shutil.rmtree(hw_folder)
        # os.makedirs(savefolder)
    os.makedirs(hw_folder)

    source_font = ImageFont.truetype(source_font, size=128)
    list_imgs = []
    for index, ch in enumerate(text.decode('utf-8')):
        source_img = draw_single_char(ch, font = source_font)
        # source_img = resize_image(source_img)
        scipy.misc.imsave(os.path.join(printed_folder, str(index) + '.png'), source_img)
        """ 
        Run generating model here 
        """
    os.system(" DATA_ROOT={} name={} phase=test \
        results_dir={} th test.lua".format(printed_folder, model, hw_folder))
    
    list_imgs = [cv2.imread(os.path.join(hw_folder, 
        str(index) + '.png'), 0) for index in range(len(text.decode('utf-8')))]
    bg = os.path.join(os.getcwd(), 'backgrounds')
    combine_images(list_imgs, line_th, bg, save_directory)



if __name__ == '__main__':
    import os
    import glob
    import argparse

    parser = argparse.ArgumentParser(description="Generate hw images from strings")
    parser.add_argument('--file', dest='file', help="file to read")
    parser.add_argument('--font', dest='font', help="font to process")
    parser.add_argument('--checkpoint', dest='checkpoint', help="checkpoint folder to choose model to gen")
    parser.add_argument('--savefolder', dest='savefolder', help="path to save results")
    args = parser.parse_args()

    if args.savefolder:
        save_directory = args.savefolder
    else:
        folder, name = os.path.split(args.file)
        save_directory = os.path.join(folder, 'results')
    print(save_directory)
    print("Saving to {}...".format(save_directory))
    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)
    os.makedirs(save_directory)

    lines = [line.rstrip('\n') for line in open(args.file)]
    
    dictionary = {}
    # models = ['1096-c-dense8', '1244-c-dense8', '1252-c-dense8']
    models = os.listdir(args.checkpoint)
    for i, line in enumerate(lines):
        model = models[np.random.randint(len(models))]
        try:
            print("Model: {} ### n-th-line: {} text: {}".format(model, i, line))
            dictionary[str(i)+'.png'] = line
            main(i, line, args.font, model, save_directory)
        except:
            pass
    
    with open(os.path.join(save_directory, 'hw_labels.json'), 'w') as fp:
        json.dump(dictionary, fp)

    print("Number of input-lines: {}".format(len(lines)))
    print("Number of output-lines: {}".format(len(dictionary.items())))
