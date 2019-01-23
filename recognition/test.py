import para
import model

import os
from os.path import join
import cv2
import numpy as np
from argparse import ArgumentParser
import keras.backend as K
from keras.models import load_model

BATCHSIZE = 64
parser = ArgumentParser(description='parser for testing the crnn model')

parser.add_argument(
    '--model', type=str, default='resnet18',
    help='select pretrained model: '
         'resnet18'
    )
parser.add_argument(
    '--weights', type=str, default='./experiment/best-model.h5',
    help='weight file')

parser.add_argument(
    '--test_img', type=str, default='/data1000G/steven/ML_PLATE/data/test_crop/9974_0.png',
    help='test img file')

parser.add_argument(
    '--bgr2rgb', action='store_true', default=True,
    help='if cv image then need bgr2rgb.')

args = parser.parse_args()

def bgr2rgb(im):
    #assert len(im.shape)
    return im[:,:,::-1]

crnn_model = model.NN_model(args, training=False)

try:
    crnn_model.load_weights(args.weights)
    print("Load pretrained weight from: ", args.weights)
except:
    raise Exception("No weight file!")
test_path = '/data1000G/steven/ML_PLATE/data/test_crop/'
test_im_names = [f for f in os.listdir(test_path) if os.path.isfile(join(test_path, f)) and ('png' in f)]

ans_csv = {}
ans_confid = {} # debug

batch_ims = []
batch_ids = []

num_invalid_len=0
from tqdm import tqdm
for i, test_im_name in enumerate(tqdm(test_im_names, desc='predict images from {} in batch {}'.format(test_path, BATCHSIZE))):
    im_id = int(test_im_name.split('_')[0])
    im = cv2.imread(join(test_path, test_im_name))
    im = cv2.resize(im, (para.image_w, para.image_h))
    if args.bgr2rgb:
        im = bgr2rgb(im)

    im = np.array(im, dtype=np.float32) / 255.0 # preprocessing
    
    batch_ims.append(im)
    batch_ids.append(im_id)
    if i == (len(test_im_names)-1): #last 
        batch_ims = np.array(batch_ims, dtype=np.float32)
    elif i%BATCHSIZE != 0:
        continue   
    elif i%BATCHSIZE == 0:
        batch_ims = np.array(batch_ims, dtype=np.float32)     
    
    #im = np.expand_dims(im, axis=0)
    y_pred = crnn_model.predict(batch_ims)
    y_pred = y_pred[:,2:,:]
    #outs = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )[0][0])[:, :para.max_text_len]
    
    def confidence(preds):
        confidences = []
        chars = []
        for pred in preds:
            pred = pred[:,:]
            array = pred.argmax(axis=1)
            char = ''
            confidence = 0.0
            for i, index in enumerate(array):
                if index<(para.num_classes-1) and (i==0 or index != array[i-1]):
                    char += para.letters[index]
                    confidence += pred[i, index]
            if len(char) < 6 or len(char) > 7:
                print(''.join([para.letters[i] for i in array]), char)
                    
            confidence = confidence/len(char)
            confidences.append(confidence)
            chars.append(char)
        #char.replace('_', '') # redundant character '_'
       
        return chars, confidences

    batch_outs, batch_confidences = confidence(y_pred)
    
    for (im_id, out, confid) in zip(batch_ids, batch_outs, batch_confidences):
        if len(out) < 6 or len(out) > 7:
            num_invalid_len += 1
            print('less than 6 or more than 7 words', im_id, out, 'total:', num_invalid_len)
        if im_id in ans_csv:
            if out != ans_csv[im_id]:
                print('id: ', im_id)
                print("the two answers: ", out, ans_csv[im_id])
                # choose one ans that've higher confidence score
                if confid > ans_confid[im_id]: 
                    print('choose answer: ', out)
                    ans_csv[im_id] = out
                    ans_confid[im_id] = confid
                elif confid <= ans_confid[im_id]:
                    print('choose answer: ', ans_csv[im_id])
                    continue 
                #raise ValueError(' two answers comflict!! ')

            elif out == ans_csv[im_id]:
                continue
        elif im_id not in ans_csv:
            ans_csv[im_id] = out
            ans_confid[im_id] = confid
    batch_ims = []
    batch_ids = []

### check non detect image and fill 'unknown'
max_len = int(max(ans_csv.keys()))
for i in range(1, max_len+1): # from 1 ~ max_len
    if i not in ans_csv:
        ans_csv[i] = 'unknown'

#output csv
import csv
with open('sample-submission.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(['ID', 'Number'])
    
    for key, value in sorted(ans_csv.items()):
        w.writerow([key, value])
