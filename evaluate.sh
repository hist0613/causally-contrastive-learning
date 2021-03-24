#!/bin/bash
for i in "0" "1" "2"
do
python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  "dataset/FineFood/triplet_posneg_1word_augmented_1x_finefood" \
--checkpoint-path "checkpoints/IMDb_load/original_augmented_1x_output_scheduling_warmup_${i}" \
--batch-size 16 \
--epoch 3 
done

