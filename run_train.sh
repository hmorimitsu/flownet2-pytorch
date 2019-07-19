#!/bin/bash
NET=FlowNet2S
TRAINING_DATASET=FlyingChairs
TRAINING_DATASET_ROOT=/datasets/FlyingChairs_release/data
VALIDATION_DATASET=MpiSintelClean
VALIDATION_DATASET_ROOT=/datasets/MPI-Sintel/training

python main.py --batch_size 8 --total_epochs 210 --model $NET --model_batchNorm False --optimizer=Adam --optimizer_lr=1e-4 \
    --loss=MultiScale --loss_norm=L2 --loss_numScales=5 --loss_startScale=4 \
    --crop_size 320 448 --training_dataset $TRAINING_DATASET --training_dataset_root $TRAINING_DATASET_ROOT \
    --validation_frequency 1 --validation_dataset $VALIDATION_DATASET --validation_dataset_root $VALIDATION_DATASET_ROOT \
    --validation_log_images --log_frequency 9999999 --extended_train_transforms