#!/bin/bash

for i in "0" "1" "2"
do
python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/triplet_2word_augmented_1x_aclImdb" \
--output-path "checkpoints/triplet_2word_augmented_1x_output_scheduling_warmup_${i}_test" \
--batch-size 16 \
--epoch 15 \
--use-margin-loss
done


