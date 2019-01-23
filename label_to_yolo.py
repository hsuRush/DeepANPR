import xml.etree.ElementTree as ET
import os
from os import listdir
from os.path import join
import random
#sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

path_to_label = '/data1000G/steven/ML_PLATE/data_old/train/'

classes = ["plate"]

valid_split = 0.1


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):  
    in_file = open(path_to_label + 'labels/plate_original/%s.xml'%(image_id))  #輸入路徑
    out_file = open(path_to_label + 'labels/plate/%s.txt'%(image_id), 'w')    #輸出路徑
    tree=ET.parse(in_file)     #得到xml樹
    root = tree.getroot()      #得到根
    size = root.find('size')      #通過size標籤得到尺寸信息
    w = int(size.find('width').text)    #分別得到照片的寬和高
    h = int(size.find('height').text)

    for obj in root.iter('object'):   #查找到每一個標籤對象
        difficult = obj.find('difficult').text   #獲得difficult標籤內容
        cls = obj.find('name').text    #獲得name標籤內容
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')  #獲取boundbox
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n') #給bb中數之間添加空格，並輸出

if not os.path.exists(path_to_label + 'labels/plate/'): #如果不存在要輸出的文件夾，則先創建
    os.makedirs(path_to_label + 'labels/plate/')

image_ids = [f[:-4] for f in os.listdir(path_to_label + 'images/plate/') if os.path.isfile(os.path.join(path_to_label + 'images/plate/', f)) and ('jpg' in f)]

for image_id in image_ids:   
    convert_annotation(image_id) 

random.shuffle(image_ids)        #打亂文件名

train_data_num = int((1-valid_split)*len(image_ids)) 
train_ids = image_ids #[0:train_data_num]       #取前384個為訓練集
train_list = open('train.txt', 'w')  #打開要輸出的圖片路徑信息
for train_id in train_ids:
    train_list.write(path_to_label + 'images/plate/%s.jpg\n'%(train_id))  #將訓練圖片的路徑寫入文件中
train_list.close()
"""
test_ids = image_ids[train_data_num:len(image_ids)]     #剩餘的為測試集
test_list = open('test.txt', 'w')  #打開要輸出的圖片路徑信息
for test_id in test_ids:
    test_list.write(path_to_label + '/images/plate/%s.jpg\n'%(test_id))  #將測試圖片的路徑寫入文件中
test_list.close()
"""
#os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
#os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")

