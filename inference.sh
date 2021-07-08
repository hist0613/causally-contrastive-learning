#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python inference_bert_imdb_pairwise_shellscript.py \
--model-path "checkpoints/SST-2/original_augmented_1x_output_scheduling_warmup_lambda_01_try3_1/best_epoch" \
--dataset-path "dataset/SST-2/ssmba_augmented_5x_sst2" \
--reps-path "reps" \
--batch-size 4 \
--epoch 2 \

