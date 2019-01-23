# -*- coding: UTF-8 -*-

import numpy as np
import cv2
import os
from os.path import join
import urllib.request
from argparse import ArgumentParser
import darknet.python.darknet as dn

cfg_path = "/home/czchen/stevenwork/plate_detection_ML_hw3/darknet/cfg/yolov3_1_cls.cfg"
weights_path = "/data1000G/steven/backup/yolov3_1_cls_final.weights"

meta_path = "/home/czchen/stevenwork/plate_detection_ML_hw3/darknet/cfg/plate.data"

save_test_crop_path = '/data1000G/steven/ML_PLATE/data/test_crop'

if not os.path.exists(save_test_crop_path):
    os.mkdir(save_test_crop_path)
#init yolo
dn.perform_detect(cfg_path, weights_path, meta_path, None, init=True)

def url_to_image(url):

    with urllib.request.urlopen(url) as resp:
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

Mode_choice = {'url_mode':0, 'image_path_mode':1, 'test_mode':2}
mode = Mode_choice['test_mode']

if mode == Mode_choice['image_path_mode'] or mode == Mode_choice['url_mode']:   
    if Mode_choice['image_path_mode']:
        with open("./test.txt" , 'r') as f:
            lines = f.read().splitlines()
        import random
        img_path = random.choice(lines)
        if not os.path.isfile(img_path):
            raise FileNotFoundError("No file in %s" %img_path)
        im = cv2.imread(img_path)


    elif Mode_choice['url_mode']:
        url = input("input img url: ")
        im = url_to_image(url)

    im = cv2.resize(im,(320,240))
    cv2.imwrite('ori.png',im)
    det_im,_ = dn.perform_detect(cfg_path, weights_path, meta_path, im)
    cv2.imwrite('prediction.jpg', det_im)

elif mode == Mode_choice['test_mode']:
    test_path = '/data1000G/steven/ML_PLATE/data/test/data/test'
    #save_test_crop_path = '/data1000G/steven/ML_PLATE/data/test_crop'
    test_im_names = [f for f in os.listdir(test_path) if os.path.isfile(join(test_path, f)) and ('jpg' in f)]
    
    for thresh in [.45, .4, .35, .3, .25, .2, .15, .1]:
        #test_im_names = no_det_list
        no_det_list = []
        for test_im_name in test_im_names:
            im = cv2.imread(join(test_path, test_im_name))
            crop_im_list = dn.perform_crop(cfg_path, weights_path, meta_path, im, thresh=thresh, crop=True, verbose=True)
            
            if crop_im_list is False: # detect nothing
                no_det_list.append(test_im_name)
                #print(test_im_name, 'No det') 
                continue

            if not crop_im_list == []:
                for i, crop_im in enumerate(crop_im_list):
                    crop_path = join(save_test_crop_path, test_im_name.split('.')[0]+'_'+str(i)+'.png')
                    
                    #print(test_im_name)
                    cv2.imwrite(crop_path, crop_im)

        print('thresh: '+str(thresh), no_det_list)

