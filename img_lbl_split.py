#  original dataset 
# path_to_data / xml and imgs

# this program will split xml and imgs like this

# path_to_data / images  / plate / (imgs)             
# path_to_data / labels  / plate_original / (xmls)


import os
path_to_data = '/data1000G/steven/ML_PLATE/data/train/'

only_xml_files = [f for f in os.listdir(path_to_data) if os.path.isfile(os.path.join(path_to_data, f)) and ('xml' in f)]
only_jpg_files = [f for f in os.listdir(path_to_data) if os.path.isfile(os.path.join(path_to_data, f)) and ('jpg' in f)]

print(only_xml_files[:5])
print(len(only_xml_files))

print(only_jpg_files[:5])
print(len(only_jpg_files))

if not os.path.exists(os.join(path_to_data, "labels")):
    os.mkdir(os.join(path_to_data, "labels"))
if not os.path.exists(os.join(path_to_data, "images")):
    os.mkdir(os.join(path_to_data, "images"))
if not os.path.exists(os.join(path_to_data, "labels/plate_original/")):
    os.mkdir(os.join(path_to_data, "labels/plate_original/"))
if not os.path.exists(os.join(path_to_data, "images/plate/")):
    os.mkdir(os.join(path_to_data, "images/plate/"))
    
for jpg, xml in zip(only_jpg_files, only_xml_files):
    os.rename(os.path.join(path_to_data, xml), os.path.join(path_to_data+'labels/plate_original/', xml))
    os.rename(os.path.join(path_to_data, jpg), os.path.join(path_to_data+'images/plate/', jpg))
    
