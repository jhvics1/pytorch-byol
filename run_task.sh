#!/bin/bash
python byol_finetune.py --gpu-id 0,1 --image_folder ../../dataset/neu-split/  --epoch 100 --from-scratch  --num-classes 6 --tune-all  --lr 0.001 --board-tag scratch --depth 101 --batch-size 128
wait
python byol_finetune.py --gpu-id 0,1 --image_folder ../../dataset/neu-split/  --epoch 100 --from-scratch  --num-classes 6 --tune-all  --lr 0.001 --board-tag scratch --depth 101 --batch-size 256

