from LRTensorBoard import LRTensorBoard
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
import numpy as np
from os.path import join

decay_epoch = 30
total_epoch = 50
lr_base = 1e-4

def lr_scheduler(epoch):
        global decay_epoch
        global total_epoch
        global lr_base
        # exponential decay
        
        if epoch > decay_epoch:
            a = total_epoch - decay_epoch
            #(epoch / a) = x / 5
            x = 5. * (epoch / a)
            scale = np.exp(-x)
            lr = scale * lr_base
        else:
            lr = lr_base

        return lr

def get_callbacks(args):
    scheduler = LearningRateScheduler(lr_scheduler)
    modelckpt = ModelCheckpoint(filepath=join(args.experiment_dir, 'best-model.h5'), monitor='val_loss', save_best_only=True, mode='auto')
    lr_tensorboard = LRTensorBoard(log_dir=join(args.experiment_dir, 'TensorBoard'))

    callbacks = [modelckpt, lr_tensorboard, scheduler]
    return callbacks