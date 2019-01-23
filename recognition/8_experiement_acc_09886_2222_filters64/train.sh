
python train.py \
    --model resnet18_2222_64 \
    --experiment_dir ./experiment \
    --epoch 40 \
    --decay_epoch 20 \
    --batch 16 \
    --lr 1e-4 \
    --valid_split 0.1 \
    

#    --weights GRU.h5
