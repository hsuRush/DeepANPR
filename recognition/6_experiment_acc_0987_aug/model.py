

import numpy as np
from argparse import ArgumentParser
import keras
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.layers import Input, Dense, Lambda, Reshape, BatchNormalization, Activation
from keras.layers.merge import add, concatenate
from keras import backend as K
import para
# input_shape = (h,w,c)
N_COL = para.image_h
N_ROW = para.image_w


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def NN_model(args, training=True):
    global N_COL
    global N_ROW
    
    if args.model =='densenet121':
        from keras.applications.densenet import DenseNet121
        input_tensor = Input(shape=(N_COL, N_ROW, 3))
        base_model = DenseNet121(input_shape=(N_COL,N_ROW,3), include_top=False, weights='imagenet', input_tensor=input_tensor, pooling=None)  
    
    elif args.model == 'resnet18':
        import resnet
        NOT_CARE = 1
        base_model = resnet.ResnetBuilder.build_resnet_18(input_shape=(N_COL,N_ROW,3), num_outputs=NOT_CARE, include_top=False)
    elif args.model == 'resnet18_2222':
        import resnet
        NOT_CARE = 1
        base_model = resnet.ResnetBuilder.build_resnet_18_2222(input_shape=(N_COL,N_ROW,3), num_outputs=NOT_CARE, include_top=False)
    elif args.model == 'resnet34':
        import resnet
        NOT_CARE = 1
        base_model = resnet.ResnetBuilder.build_resnet_34(input_shape=(N_COL,N_ROW,3), num_outputs=NOT_CARE, include_top=False)
    elif args.model == 'resnet50':
        import resnet
        NOT_CARE = 1
        base_model = resnet.ResnetBuilder.build_resnet_50(input_shape=(N_COL,N_ROW,3), num_outputs=NOT_CARE, include_top=False)
    elif args.model == 'resnet101':
        import resnet
        NOT_CARE = 1
        base_model = resnet.ResnetBuilder.build_resnet_101(input_shape=(N_COL,N_ROW,3), num_outputs=NOT_CARE, include_top=False)
    
    else:
        raise TypeError('model should be in the list of the supported model!')

    print('Input col: ', N_COL)
    print('Input row: ', N_ROW)

    x = base_model.output
    #CNN to RNN
    x = Lambda(lambda x: K.permute_dimensions(x,(0,2,1,3)))(x) # switchaxes from [b,h,w,c] to [b,w,h,c]
    conv_shape = x.get_shape() # b, h,w,c  resnet 18 -> (?, 16, 32, 256)
    print('conv_shape', conv_shape)
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])), name='reshape')(x)
    x = Dense(para.dense_size, activation='relu', kernel_initializer='he_normal', name='dense1')(x)
    #x = BatchNormalization()(x)
    # GRU RNN
    gru_1 = GRU(para.rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
    gru_1b = GRU(para.rnn_size, return_sequences=True, go_backwards=True, 
                init='he_normal', name='gru1_b')(x)
    gru1_merged = add([gru_1, gru_1b])
    gru1_merged = BatchNormalization()(gru1_merged)

    gru_2 = GRU(para.rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(para.rnn_size, return_sequences=True, go_backwards=True, 
                init='he_normal', name='gru2_b')(gru1_merged)
    gru2_merged = concatenate([gru_2, gru_2b])
    gru2_merged = BatchNormalization()(gru2_merged)

    inner = Dense(para.num_classes, kernel_initializer='he_normal',name='dense2')(gru2_merged)
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[para.max_text_len], dtype='float32') # (None ,7)
    input_length = Input(name='input_length', shape=[1], dtype='int64')     # (None, 1)
    label_length = Input(name='label_length', shape=[1], dtype='int64')     # (None, 1)

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) #(None, 1)

    if training:
        return Model(inputs=[base_model.input, labels, input_length, label_length], outputs=loss_out), conv_shape[1]
    else:
        return Model(inputs=[base_model.input], outputs=y_pred)
    
def main():
    parser = ArgumentParser(description='parser for NN model')

    parser.add_argument(
        '--model',type=str, default='resnet50',
        help='select pretrained model: '
            '  resnet50'
        )
    args = parser.parse_args()
    NN_model(args)
