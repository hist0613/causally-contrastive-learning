#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2 python inference_bert_imdb_pairwise_shellscript.py \
--model-path "checkpoints/MultiNLI_telephone/original_augmented_1x_output_scheduling_warmup_lambda_01_2/best_epoch" \
--dataset-path "dataset/MultiNLI_telephone/ssmba_augmented_5x_mnli/" \
--reps-path "reps" \
--batch-size 6 \
--epoch 2 \

