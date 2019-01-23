import numpy as np
image_w = 128
image_h = 64
image_c = 3

rnn_size = 256
dense_size = 64
max_text_len = 7

CHAR_VECTOR = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
letters = [letter for letter in CHAR_VECTOR]

num_classes = len(letters)

SEED = 50
def decodePlateVec(y):
    vec = np.zeros((1, max_text_len), dtype=np.uint8)
    for i in range(7):
        vec[0, i] = np.argmax(y[:,i])
    return vec

def encode_plate(string):
    num = np.zeros((num_classes, max_text_len))
    
    for i in range(len(string)):
        for j in range(num_classes):

            if ( string[i] == letters[j] ):
                num[j,i] = 1
                
    if (len(string) == 6):
        num[num_classes-1, 6] = 1
    if (len(string) == 5):
        num[num_classes-1, 6] = 1
        num[num_classes-1, 5] = 1
    if (len(string) == 4):
        num[num_classes-1, 6] = 1
        num[num_classes-1, 5] = 1
        num[num_classes-1, 4] = 1
    
    #print(string, '\n', num)
        
    return num
