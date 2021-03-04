#!/bin/bash
python inference_bert_imdb_pairwise_shellscript.py \
--model-path "checkpoints/not_augmented_output_scheduling_warmup/epoch_14" \
--dataset-path "dataset/triplet_5word_augmented_1x_aclImdb" \
--reps-path "reps" \
--batch-size 16 \
--epoch 15 \

