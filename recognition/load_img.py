import xml.etree.ElementTree as ET
import numpy as np
from os.path import join
import cv2

path_to_data = '/data1000G/steven/ML_PLATE/data/train/'

classes = ["plate"]

def images_crop_by_annotation(image_id, model_w,model_h,model_c, resize=False): 
    in_file = open(join(path_to_data, 'labels/plate_original/%s.xml'%(image_id)))  #輸入路徑
    im = cv2.imread(join(path_to_data, 'images/plate/%s.jpg'%(image_id)))
    im = bgr2rgb(im)
    tree = ET.parse(in_file)     #得到xml樹
    root = tree.getroot()      #得到根
    size = root.find('size')      #通過size標籤得到尺寸信息
    w = int(size.find('width').text)    #分別得到照片的寬和高
    h = int(size.find('height').text)

    bbox_crop_im_list = []
    _temp_img = None
    for i, obj in enumerate(root.iter('object')):   #查找到每一個標籤對象
        difficult = obj.find('difficult').text   #獲得difficult標籤內容
        cls = obj.find('name').text    #獲得name標籤內容
        if cls =='car':
            obj.find('name').text = 'plate'
            tree.write(join(path_to_data, 'labels/plate_original/%s.xml'%(image_id)), encoding="UTF-8")
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            #print(cls)
            #print(image_id)
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')  #獲取boundbox
        xmin, xmax, ymin, ymax = int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text)
        
        _temp_img = im[ymin:ymax, xmin:xmax].copy()
        if resize:
            _temp_img = cv2.resize(_temp_img, (model_w, model_h))
        #print(_temp_img.shape)
        
        cv2.imwrite(join("/data1000G/steven/ML_PLATE/data/train_crop/", image_id+'_'+str(i)+'.png'),  _temp_img)

     
    plate_label_str = image_id.split('_')[0]

    if len(bbox_crop_im_list) > 1:
        print("WTF")
        print(image_id)
        assert ValueError("more than one bbox in train img")
    
    if _temp_img is not None:
        return _temp_img, plate_label_str

    elif _temp_img is None:
        return None, None

def bgr2rgb(im):
    #assert len(im.shape)
    return im[:,:,::-1]

def encodePlate(str):
    num = np.zeros((37, 7))
   
    for i in range(len(str)):
        for j in range(36):
#            print(i,j)
            if ( str[i] == chars[j] ):
                num[j,i] = 1;
                
    if (len(str) == 6):
        num[36,6] = 1;
    if (len(str) == 5):
        num[36,6] = 1;
        num[36,5] = 1;
    
    print(str, '\n', num)
        
    return num

def read_crop_img_lbl(path, image_id, resize_w,resize_h, resize=False):
    im = cv2.imread(join(path, image_id))
    #os.mkdir("/data1000G/steven/ML_PLATE/data/train_resize_crop/")
    
    #im = bgr2rgb(im)
    if resize:
        im = cv2.resize(im, (resize_w, resize_h))
        #cv2.imwrite(join('/data1000G/steven/ML_PLATE/data/train_resize_crop/',image_id), im)
        #print('write', image_id)
    plate_label_str = image_id.split('_')[0]

    im = np.array(im, dtype=np.float32) / 255.0

    return im,  plate_label_str

### import module
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os

### parameters
import para 
w = para.image_w
h = para.image_h
c = para.image_c

### read the data 
def perform_crop(path):

    #imgs = np.array([])
    t = []
    for image_id in os.listdir(path):
    #     print(image_id)
        shape= images_crop_by_annotation(image_id.split('.')[0], w,h,c) #  no ".jpg"
        if shape is not None:
            t.append(np.array(shape))
        #imgs = np.append(imgs, im_box_only, axis=3) 
    X = np.array(t)
    print (np.mean(X, axis=0, keepdims=True)) # mean [[27.96106694 64.55460374  3.        ]]
    print(np.max(X,axis=0,keepdims=True)) # max [[ 76 147   3]]
   
    #return np.asarray(imgs, np.float32)   

def load_crop_imgs(path, resize_w, resize_h, resize, multiprocessing=True):
    ims,labels = [], []

    # no multiprocessing
    for image_id in os.listdir(path):
        #print(image_id)
        
        im,label = read_crop_img_lbl(path, image_id, resize_w, resize_h, resize)  #  no ".jpg"
        if im is not None and label is not None:
            ims.append(im)
            labels.append(para.encode_plate(label))
    #print(labels)
    return np.array(ims), np.array(labels, dtype=np.uint8)

if __name__ == '__main__':
    ims, lbls = load_crop_imgs("/data1000G/steven/ML_PLATE/data/train_crop/",w,h, resize=True)
    print(ims.shape, len(lbls))

"""
### ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

seed = 1
i = 0
for batch in datagen.flow(data, batch_size=1,seed=seed,
                          save_to_dir='E:/tmp/augment', save_format='jpg'):
    i += 1
    if i > 20:
        break  
        
i = 0
for batch in datagen.flow(label, batch_size=1,seed=seed,
                          save_to_dir='E:/tmp/augment', save_format='png'):
    i += 1
    if i > 20:
        break 

"""
