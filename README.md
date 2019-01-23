# NTUT ML license plate recognition
- It's a Deep learning based Automatic number-plate recognition for Taiwanese plate using two stage methods, modified yolov3 and modified ResNet+GRU. I got 1st on [Kaggle Leaderboard](https://www.kaggle.com/c/ntut-ml-2018-computer-vision/leaderboard) in NTUT Machine Learning course 2018 FALL.
# Requirement
Python
* python 3.6.5
* scikit-learn==0.20.0
* opencv-python==3.4.3.18
* numpy==1.15.2
* matplotlib==3.0.0
* Keras==2.2.4
* tensorflow-gpu==1.11.0
* tqdm==4.28.1

## Outline
* Yolov3 
* ResNet18+GRU
* Preparation
* Training 
* Testing
* Conclusion
* References
* Appendix
    * Experiments (1)
    * Problems
    * Experiments (2)
    * TODO

## Yolov3 
&nbsp;  &nbsp; Yolo(You Only Look Once)[[0]](https://arxiv.org/abs/1506.02640)  is a well-known real-time detection model for object detection. Unlike RPN, R-CNN, fast R-CNN.. use region proposal network to extract thousands of region to do classification, Yolo "only look once". The grid cell and the boudingbox regressor allow yolo to perform the object classification and the object detection simultaneously.

&nbsp; &nbsp; Yolo v2[[1]](https://arxiv.org/abs/1612.08242) mainly improve with three aspects.
* Batch Normalization 
* Convolutional With Anchor Boxes instead of grid cell
* k-means anchor boxes
* High Resolution Classifier
* network architecture Darknet19

&nbsp; &nbsp; Yolo v3[[2]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) improve with two major aspects.
* network architecture Darknet53
* a better loss for boundingbox location.


## ResNet18+GRU
&nbsp;  &nbsp; Residual Neural Network [[3]](https://arxiv.org/abs/1512.03385) is also very popular network use for image feature extraction problem, the residual block let the network avoid the gradient vanishing problems and make losses smoother[[4]](https://arxiv.org/abs/1712.09913).
&nbsp;  &nbsp; Gated recurrent units [[5]](https://arxiv.org/abs/1406.1078) (GRUs) are a gating mechanism in recurrent neural networks, introduced in 2014 by Kyunghyun Cho et al. It's very similiar to LSTM, but GRUs are more efficient there're a nice [comment](https://datascience.stackexchange.com/a/14585) by Abhishek Jaiswal.


There're some awesome websites to help you understand.
[[Lecture] Evolution: from vanilla RNN to GRU & LSTMs](https://towardsdatascience.com/lecture-evolution-from-vanilla-rnn-to-gru-lstms-58688f1da83a) by Supervise.ly.

&nbsp;  &nbsp; After the CNN feature extractor, I reshape the feature map from `[height,width,channel]` to `[width, height*channel]`. I got `[32,16,256]` in the output of the resnet18 model. After reshaping into [32, 16\*256], I connect a fully-connected layer to reduce the dimension to [32,64] features and input into the GRU rnn model, and finally a Softmax out layer for onehot encode output as a string.

<p align="center"><img src ="https://github.com/MachineLearningNTUT2018/computer-vision-107368002/blob/master/demo/transpose.png?raw=true"width="480" title="reshape" ></p>

&nbsp;  &nbsp; Because of the Variation of the label length and the maxmium label length, I padding all of the length labels to be 7. (7 is the maximum length of Taiwan plate)
    
    ABC123 -> ABC123_
    DE2345 -> DE2345_
&nbsp;  &nbsp; I use [ctc loss](https://distill.pub/2017/ctc/) to train this model and discard the first two outputs which seem as junks so input length will be 30 instead. Last, Using greedy Algorithm to minimize the input length 30 into a string. In addition, don't forget to discard the `_` char.  
    
    A_C__DD__1__22__44__5 -> ACD245

<p align="center"><img src ="https://xmfbit.github.io/img/warpctc_intro.png" width="480" title="ctc loss" ></p>

a [ctc demo website](https://distill.pub/2017/ctc/) to understand more.

## Preparation
### Yolov3
* #### kmeans
  &nbsp; &nbsp; yolov2 and later use anchor boxes instead of grid cell, but we need to initial some nice anchor boxes to improve the training process, so we need to run k-means on the boundingbox of our dataset.
```console
$ cd kmeans/
$ python run_kmeans.py
// write result in kmeans/k_means_anchor file.
// like this:
//  Accuracy: 89.91%
//  anchors = 69,25,  78,33,  71,29,  44,19,  75,37,  47,23,  58,27,  91,42,  55,22
// and paste into yolov3.cfg.
```
* #### Format Issue
    labelimg format is not suitable for darknet, so we need to write a convert program to fix the issue.
``` console
//run img_lbl_split.py to split xml and imgs.
$ python img_lbl_split.py
// from 
// path_data/[xmls&imgs]
// to
// path_to_data/images/plate/[imgs]
// path_to_data/labels/plate/[xmls]

//run the convert script
$ python label_to_yolo.py
/* the formula
x = (xmin + (xmax-xmin)/2) * 1.0 / image_w
y = (ymin + (ymax-ymin)/2) * 1.0 / image_h
w = (xmax-xmin) * 1.0 / image_w
h = (ymax-ymin) * 1.0 / image_h 
     */
```

* #### Size Issue
  &nbsp; &nbsp;  The original image width and height is 608x608, but get 320x240 in our dataset. There will be a upsample error cause by 240. Yolov3 downsample /2 5times, so the 240/2<sup>5</sup> = 7.5 but get 8 instead. so the upsample 8\*2 = 16 can't concentrate with 15(240/2<sup>4</sup>) by residual block.

* #### Solution 
  &nbsp; &nbsp; Therefore, I modify Yolov3, I call `yolov3_1_cls.cfg` in darknet/cfg/. I remove one 2-strides(downsample) conv layers and add more conv layers, and also use my custom anchor boxes that I calculated in [kmeans](#kmeans).
  
  

 <center>Yolov3 Architecture  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Modified Yolov3 Architecture </center>
  <div style="text-align:center">
  <img src ="https://github.com/MachineLearningNTUT2018/computer-vision-107368002/blob/master/demo/yolov3_structure.png?raw=true"height="540" title="Yolov3 structure" />
  <img src ="https://github.com/MachineLearningNTUT2018/computer-vision-107368002/blob/master/demo/my_yolo_struct.png?raw=true"height="540" title="modified Yolov3 structure" />
  </div>
  
### ResNet18+GRU
* Size Issue

  &nbsp;  &nbsp; I use ResNet18 as the image feature extractor and set input image width height as 128x64. Thus,  I modify some conv layers and remove Maxpooling layers due to the image size of the plate(224x224 in the original paper).
  
* CNN Architecture 

  &nbsp;  &nbsp; the original residual block in resnet18:
  
  <div align="center"><img src ="https://github.com/MachineLearningNTUT2018/computer-vision-107368002/blob/master/demo/resnet_struct.png?raw=true"height="360" title="ResNet structure" /></div>
  
  &nbsp;  &nbsp; I increase the conv layers by changing the residual block from `[2,2,2,2]` to `[2,4,4,2]` and minize the filters=32,64,128,256. Last, I remove the 7x7 /2 conv layers and Maxpooling layers and add 5x5 conv instead.
  
  [EDIT] I use [2,2,2,2] and filters=64,128,256,512 get 98.86% performance.(best on kaggle PLB)

  <p align="center"><img src ="https://github.com/MachineLearningNTUT2018/computer-vision-107368002/blob/master/demo/my_resnet_struct.png?raw=true"height="480" title="modified ResNet structure" ></p>
  
* RNN Architecture
  &nbsp;  &nbsp; I feed the datas into two GRUs(GRU, GRU_b) with one reverse sequence, then `add`,`batch normalization`. Next I repeat the GRU procedure with replacing `add` to `concatenate`.(GRU1, GRU2)
  <p align="center"><img src ="https://github.com/MachineLearningNTUT2018/computer-vision-107368002/blob/master/demo/rnn_structure.png?raw=true" height="640" title="rnn structure"></p>
* crop the image by true labels  in order to get the plate image and resize to 128x64.

        implement in recognition/load_img.py
        
### Dataset 

  &nbsp;  &nbsp; There're will thousands of labels are not precisely, like AFG1929 ADB2531 and so on...
  My two stage methods extremely depend on the ground truth, since the final accuracy is multiplication of two accuracies. the labeled data is **extremely** important for me.
  <p align="center"><img src ="https://github.com/MachineLearningNTUT2018/computer-vision-107368002/blob/master/demo/bad_label.png?raw=true" width="360" title="loss"></p>
  
  &nbsp;  &nbsp; I re-labelled 5098 images.
## Training
create a `train.txt` contains the absolute path to the images.
and need to change the path in `darknet/cfg/plate.data`.
### Yolov3
```console
$ cd darknet/ 
$ sh train_1_cls.sh
```
the training parameters is setting in the `dakrnet/cfg/yolov3_1_cls.cfg`.
```python
learning_rate=0.001
batch=64
max_batches = 4700
steps=3800,4100
scales=.1,.1
 ```   
decay in 3800 and 4100 by lr\*0.1.

<p align="center"><img src ="https://github.com/MachineLearningNTUT2018/computer-vision-107368002/blob/master/demo/yolo_loss.png?raw=true" width="360" title="loss"></p>

Because of the validation problem on darknet, I train all of the dataset without any split, so I write a code to demo on youtube videos([source](https://www.youtube.com/watch?v=fx-CQ-XISYA)), here's a demo below, the output will be lightblue ![](https://placehold.it/15/3BB9FF/000000?text=+)
 boundingboxes.
<p align="center"><img src ="https://github.com/MachineLearningNTUT2018/computer-vision-107368002/blob/master/demo/yo.gif?raw=true" title="demo for my Yolo"></p>

### ResNet18+GRU

see some config in `train.sh`, feel free to change it.
```console
//recognition/rain.sh
python train.py \
    --model resnet18 \
    --experiment_dir ./experiment \
    --epoch 40 \
    --decay_epoch 20 \
    --batch 16 \
    --lr 1e-4 \
    --valid_split 0.1 
```
Train command
``` console
$ cd recognition/
$ sh train.sh 
//there are some options in train.sh
//check train.py
```


<p align="center"><img src ="https://github.com/MachineLearningNTUT2018/computer-vision-107368002/blob/master/recognition/6_experiment_acc_0987_aug/log.png?raw=true" height="360" title="loss"></p>


I use LearningRateScheduler and perform an exponential decay fomr decay_poch to final epoch.

<p align="center"><img src ="https://github.com/MachineLearningNTUT2018/computer-vision-107368002/blob/master/demo/lr.png?raw=true" height="300" title="learning rate"></p>


Because the detection model won't detect perfectly every time, I train the model with some image augmentation so the model will be more robust.

```python
train_datagen = ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.1,
                rotation_range=30,
                fill_mode='nearest',
                )
```

Last but not least, I save the model with lowest validation loss by the Keras callback function.
## Testing
## Yolov3
change the `cfg/yolov3_1_cls.cfg` to

    #Testing
    batch=1
    subdivisions=1
    # Training
    #batch=64
    #subdivisions=16
    
and then run the script to crop test images.

Because I think there are only **few** background images, I set the threshold to [0.45,0.4,0.35.....0.1] recurrently and discard the image that's detected to ensure there's a detection in each image.
```console
$ python demo_image.py
```

## ResNet+GRU
```console
$ cd recognition/
$ python test.py
```
There are multiple detections in a image sometimes, so I fusion the result according to the confidence score.
## Conclusion
&nbsp; &nbsp;  I use ResNet18+GRU with yolov3 and get 98.8% acurracy in Kaggle public leaderboard. I am especially appreciate to Pro.Liao and TAs that deliver a fantastic ML course and Kaggle Competitions.

## References:
* Code
  1. http://jarvus.dragonbeef.net/note/notePlateCTC.php
  2. https://github.com/qjadud1994/CRNN-Keras
  3. https://github.com/ypwhs/captcha_break
  4. https://github.com/raghakot/keras-resnet
* Papers

  [0]. [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) by Joseph Redmon et al.
  
  [1]. [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) by Joseph Redmon et al.
   
  [2]. [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf) by Joseph Redmon et al.
  
  [3]. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Kaiming He et al.

  [4]. [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913) by Hao Li et al.

  [5]. [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) by Kyunghyun Cho et al.
  
* others
    1. [comment about GRU](https://datascience.stackexchange.com/a/14585) by Abhishek Jaiswal
    2. demo video from https://www.youtube.com/watch?v=fx-CQ-XISYA 
    
-----
# *Appendix*
**All the results are testing on Kaggle Public Leaderboard.**

To run the result I've trained, please overwrite para.py, resnet.py, model.py from the each `recognition/experiment` folder to `recognition/` and change the weight path and select the certain model name in test.py.
## Experiments (no switchaxes)
No switchaxes in `model.py`
```python
x = Lambda(lambda x: K.permute_dimensions(x,(0,2,1,3)))(x) # switchaxes from [b,h,w,c] to [b,w,h,c]
```
* I  **accidentally** reshape from `[height,width,channel]` to `[height, width*channel]` and turn out 98.43%, not bad actually.
* I reshape from `[height,width,channel]` to `[height, width*channel]` and turn out to be 98.43% accuracy.

## Problem about axes
I've spotted that the shape `[h,w,c]` reshape to `[w,h*c]` is different from `[w,h,c]` reshaping to `[w,h*c]`. the `[w,h,c]` is the correct method. So there're newer experiments below.

## Experiment with switchaxes
* Using ResNet50 as the Backbone CNN extractor, It drop to 97.7% performance.
* I change the rnn size from 256 to 128, and get 98.1% performance.
* I change dense size from 64 to 128, the training and validation loss was lower than usual, but the get 98.3% performance.
* I've change the resnet18's block to `[2,2,2,2]` and get 98.0% performance.
* I change `height_shift_range` and `shear_range` both from 0.1 to 0.2, improve ~0.5% performance.
* I've change the resnet18's block to `[2,2,2,2]` and filter=64 instead of 32, get 98.86% performance(best).(8_experiment folder weight [link](https://drive.google.com/file/d/1_y2UTB70SeG3-SU8gVvhjSUHkFWrqA9O/view?usp=sharing))
* Continued the last experiment, I add bn layer after the Dense layer between CNN and RNN, get 98.83% accuracy.   
* I want to know the effect of the relabelled data,So I train the same yolo with original data(no re-label), and the 8_experiment's model with origin data(no re-label) as well, I get 98.4% accuracy, So the relabelled data did few impact.

## TODO
* I've seen the [Group normalization](https://arxiv.org/abs/1803.08494) papers which might be useful in our case.
* There's a method named : [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025) which can implement in yolo and resnet to improve the size, angle misalignment problems.
* finetune with parameters in `para.py`, such as the size of the Dense layer (between CNN and RNN), RNN size and change the composition of the residual block. 
* ~~put a BN layer after the Dense layer.~~ (NOT WORK)
* change rnn Architecture.
* a demo code that including detection and recognition by inputting video streams. (for now, detection only)
