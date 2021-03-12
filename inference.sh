#!/bin/bash
python inference_bert_imdb_pairwise_shellscript.py \
--model-path "checkpoints/IMDb/original_augmented_1x_output_scheduling_warmup_2/epoch_2" \
--dataset-path "dataset/IMDb/original_augmented_1x_aclImdb_full" \
--reps-path "reps" \
--batch-size 2 \
--epoch 15 \

