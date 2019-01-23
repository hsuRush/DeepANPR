import glob
import xml.etree.ElementTree as ET
 
import numpy as np
 
from kmeans_lib import kmeans, avg_iou
 
ANNOTATIONS_PATH = "/data1000G/steven/ML_PLATE/data/train/labels/plate_original/"
CLUSTERS = 9

HEIGHT = 240
WIDTH = 320
 
classes = ["plate"]

def load_dataset(path):
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ET.parse(xml_file)
    
        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))
        if height != HEIGHT and width != WIDTH:       
            print(1)

        for obj in tree.iter("object"):
            cls = obj.find('name').text
            difficult = obj.find('difficult').text
            if cls not in classes or int(difficult) == 1:
                continue

            xmin = int(obj.findtext("bndbox/xmin")) / width
            ymin = int(obj.findtext("bndbox/ymin")) / height
            xmax = int(obj.findtext("bndbox/xmax")) / width
            ymax = int(obj.findtext("bndbox/ymax")) / height
        
            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)

        if xmax == xmin or ymax == ymin:
            print(xml_file)

        dataset.append([xmax - xmin, ymax - ymin])
    return np.array(dataset)
 
if __name__ == '__main__':
    times = 50
    best_acc = 0
    best_anchor = None
    
    #print(__file__)
    for i in range(times):
        data = load_dataset(ANNOTATIONS_PATH)
        out = kmeans(data, k=CLUSTERS)
        #clusters = [[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]]
        #out= np.array(clusters)/416.0
        #print(out)
        if avg_iou(data, out) * 100 > best_acc:
            best_acc = avg_iou(data, out) * 100

        print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))

        anchors_for_yolo = [[int(round(x)), int(round(y))]for x,y in zip(out[:, 0]*WIDTH, out[:, 1]*HEIGHT )]
        if best_acc == avg_iou(data, out) * 100:
            best_anchor = anchors_for_yolo
        
        END =',  '
        print('anchors = ', end='')
        for i, anchor in enumerate(anchors_for_yolo):
            if i == len(anchors_for_yolo)-1:
                END = '\n'

            print(*anchor, sep=',', end=END)

        #print("Boxes:\n {}-{}".format(out[:, 0]*WIDTH, out[:, 1]*HEIGHT))
        
        ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
        #print("Ratios:\n {}".format(sorted(ratios)))

    with open('k_means_anchor', 'a') as f:

        print("Accuracy: {:.2f}%".format(best_acc), file=f)
        print("Accuracy: {:.2f}%".format(best_acc))

        print('anchors = ', end='', file=f)
        print('anchors = ', end='')
        for i, anchor in enumerate(best_anchor):
            if i == len(anchors_for_yolo)-1:
                END = '\n'
            print(*anchor, sep=',', end=END, file=f)
            print(*anchor, sep=',', end=END)
            
