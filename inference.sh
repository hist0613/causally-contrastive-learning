#!/bin/bash
python inference_bert_imdb_pairwise_shellscript.py \
--model-path "checkpoints/SST-2/original_augmented_1x_output_scheduling_warmup_lambda_01_try2_2/epoch_2" \
--dataset-path "dataset/SST-2/original_augmented_1x_sst2" \
--reps-path "reps" \
--batch-size 3 \
--epoch 2 \

