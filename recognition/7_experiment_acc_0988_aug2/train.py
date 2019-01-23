
import random
import os
from os.path import join
import numpy as np
import json 

import common
import load_img
import model
import para
import callback

from argparse import ArgumentParser
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

from keras.applications.densenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Adadelta

parser = ArgumentParser(description='parser for NN model')

parser.add_argument(
    '--model', type=str, default='resnet18',
    help='select pretrained model: '
         '  resnet50'
         '  resnet18'
    )

parser.add_argument(
    '--experiment_dir', type=str, default='./experiment',
    help='the experiment directory that save log files.'
    )

parser.add_argument(
    '--weights', type=str, default=None,
    help='weight file active at resume')

parser.add_argument(
    '--epoch', type=common.positive_int, default=40,
    help='train iteration, default = 50')

parser.add_argument(
    '--decay_epoch', type=common.positive_int, default=20,
    help='train iteration that will decay by exponential, default = 30')

parser.add_argument(
    '--batch', type=common.positive_int, default=16,
    help='batch size, default = 16')

parser.add_argument(
    '--valid_split', type=common.probility_float, default=0.1,
    help='valid_split, default = 0.1')

parser.add_argument(
    '--lr', type=common.positive_float, default=1.2e-4,
    help='learning rate')

parser.add_argument(
    '--resume', action='store_true', default=False,
    help='resume')

def draw(args, history):
    plt.ioff()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(join(args.experiment_dir, "log.png"))

def train(args, crnn_model,  train_gen,val_gen,  train_step,val_step):
    callbacks = callback.get_callbacks(args)
    crnn_model.compile(optimizer=Adam(lr=args.lr), loss={'ctc': lambda y_true, y_pred: y_pred})
    history = crnn_model.fit_generator(train_gen, workers=1, steps_per_epoch=train_step, epochs=args.epoch, callbacks=callbacks, validation_data=val_gen,validation_steps=val_step)
    
    draw(args, history)

def main():
    args = parser.parse_args()
    print(args)
   
    if not os.path.isdir(args.experiment_dir):
        os.mkdir(args.experiment_dir)
    elif not args.resume :
        raise ValueError('experiment dir is existed! ')
        
    with open(join(args.experiment_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)
 
    #model
    crnn_model, input_length = model.NN_model(args, training=True)
    print(input_length)
    if args.resume == True:
        crnn_model.load_weights(args.weights)
        print("Load pretrained weight from: ", args.weights)
    elif not args.resume:
        print("No pretrained weight!")

    #callbacks
    callback.decay_epoch = args.decay_epoch 
    callback.total_epoch = args.epoch
    callback.lr_base = args.lr
    #data
    ims, lbls = load_img.load_crop_imgs("/data1000G/steven/ML_PLATE/data/train_resize_crop/", para.image_w,para.image_h, resize=False)

    train_ims, val_ims, train_lbls, val_lbls = train_test_split(ims, lbls, test_size=args.valid_split , random_state=para.SEED)
    
    trainNum = train_ims.shape[0]
    validNum = val_ims.shape[0]

    decode_train_lbls = np.zeros((trainNum, para.max_text_len), dtype=np.uint8)
    decode_valid_lbls = np.zeros((validNum, para.max_text_len), dtype=np.uint8)
    for i in range(trainNum):
        decode_train_lbls[i] = para.decodePlateVec(train_lbls[i,:,:])
    for i in range(validNum):
        decode_valid_lbls[i] = para.decodePlateVec(val_lbls[i,:,:])

    print('train imgs: ', trainNum)
    print('valid imgs: ', validNum)
    
    train_datagen = ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                rotation_range=30,
                fill_mode='nearest'
                )

    train_gen = train_datagen.flow(
            train_ims, decode_train_lbls,  # train_x, train_y
            batch_size=args.batch,
            shuffle=True
            )  # set as training data
    
    val_datagen = ImageDataGenerator(
                )

    val_gen = val_datagen.flow(
            val_ims, decode_valid_lbls,  # train_x, train_y
            batch_size=args.batch,
            )  # set as training data

    def multilbls_gen(flow_gen):
        while True:
            x, y = next(flow_gen)
            b=np.array(x).shape[0]
            inputs = {
            'the_input_imgs': x,  # (bs, 128, 64, 1)
            'the_labels': y,  # (bs, 8)
            'input_length': np.full((b,1), int(input_length-2)),  # (bs, 1) -> 모든 원소 value = 30
            'label_length': np.full((b,1), para.max_text_len)  # (bs, 1) -> 모든 원소 value = 8
            }
            outputs = {'ctc': np.zeros([b])}

            yield (inputs, outputs)
        

    multi_train_gen = multilbls_gen(train_gen)
    #a,b = next(multi_train_gen)
    #print(a, b)
    
    #return 
    multi_val_gen = multilbls_gen(val_gen)
    
    train_step = len(train_ims) // args.batch
    val_step = len(val_ims) // args.batch
    train(args, crnn_model, multi_train_gen, multi_val_gen, train_step, val_step)

if __name__ == '__main__':
    main()
