#!/bin/bash
python inference_bert_imdb_pairwise_shellscript.py \
--model-path "checkpoints/FineFood_full/original_augmented_1x_output_scheduling_warmup_2/epoch_2" \
--dataset-path "dataset/FineFood_full/original_augmented_1x_finefood" \
--reps-path "reps" \
--batch-size 3 \
--epoch 2 \

