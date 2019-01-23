# -*- coding: UTF-8 -*-

import cv2
import numpy as np
import os
from argparse import ArgumentParser
import darknet.python.darknet as dn
import time
#cfg_path = "/home/czchen/stevenwork/plate_detection_ML_hw3/darknet/cfg/yolov3_plate.cfg"
#weights_path = "/data1000G/steven/backup_v0/yolov3_plate_final.weights"

parser = ArgumentParser(description='demo application for video')

parser.add_argument(
    '--cfg_path', type=str, default="/home/czchen/stevenwork/plate_detection_ML_hw3/darknet/cfg/yolov3_1_cls.cfg",
    help='path of the existed pokemon video. ')

parser.add_argument(
    '--weights_path', type=str, default="/data1000G/steven/backup/yolov3_1_cls_final.weights",
    help='path of the existed pokemon video. ')

parser.add_argument(
    '--meta_path', type=str, default="/home/czchen/stevenwork/plate_detection_ML_hw3/darknet/cfg/plate.data",
    help='path of the existed pokemon video. ')

parser.add_argument(
    '--input_video_dir', type=str, default='/data1000G/steven/video/',
    help='path of the existed pokemon video. ')
    
parser.add_argument(
    '--input_video_name', type=str, default='pokemon_1mp4',
    help='path of the existed pokemon video. ')

parser.add_argument(
    '--output_video_dir', type=str, default='video_output',
    help='output dir of the pokemon video after processing')

parser.add_argument(
    '--output_video_name', type=str, default='output',
    help='output dir of the pokemon video after processing')
    

def video_process(args):
    dn.perform_detect(args.cfg_path, args.weights_path, args.meta_path, None, init=True)
    cap = cv2.VideoCapture(os.path.join(args.input_video_dir, args.input_video_name))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if not os.path.isdir(args.output_video_dir):
        os.mkdir(args.output_video_dir)
    
    avi = True

    if avi == True:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_name = os.path.join(args.output_video_dir, args.output_video_name) + '.avi'
    else:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') 
        output_name = os.path.join(args.output_video_dir, args.output_video_name) + '.mp4'

    #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(output_name, fourcc, fps//5, size)
    
    iteration = 1
    while(cap.isOpened()):
        frame_buffer = []
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is false
        if ret == False and type(frame) is type(None):
            break

	    #horizontal_frame = cv2.flip(frame, 1)
        frame_buffer.append(frame)
        #frame_buffer.append(horizontal_frame)	

        # Our operations on the frame come here
        if iteration % 5 == 0:
        
            frame_buffer = np.array(frame_buffer)
            start_time=time.time()
            batch_imgs = perform_predict(args, frame_buffer, frame_buffer.shape[0])
            end_time=time.time()
            print(str(round(1/(end_time-start_time), 2))+'fps')
            for i in range(0,len(batch_imgs)):
                out.write(batch_imgs[i])

            frame_buffer = []

        # Display the resulting frame
        #cv2.imshow('frame', frame)
        iteration += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print('Video file has been written to ', output_name)

def draw_chinese_to_img(img, word, place, color, size):
    from PIL import ImageFont, ImageDraw, Image
    color += (0,)
    fontpath = "./font/SourceHanSerifTC-Bold.otf" # <== 這裡是宋體路徑 

    font_size = int(60 * (size[0]*size[1]) / (1280*720))

    font = ImageFont.truetype(fontpath, font_size)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(place,  word, font=font, fill=color)
    _img = np.array(img_pil)

    return _img

def perform_predict(args, _ims, batch):
    """
        im: 4D-array_batches image [batch, h, w, c]

        return type: 4D-array_batches image [batch, h, w, c] that bbox is drawed
    """

    det_ims = []
    for _im in _ims:
        det_im, ret = dn.perform_detect(args.cfg_path, args.weights_path, args.meta_path, _im)
        det_ims.append(det_im)

    return det_ims

def main():
    args = parser.parse_args()
    print('args--------------------')
    print(args)
    print('------------------------')
    print()
   
    video_process(args)

if __name__ == "__main__":
    main()
