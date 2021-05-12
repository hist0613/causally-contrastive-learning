#!/bin/bash
for i in "0" "1" "2"
do
python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  "dataset/SST-2/triplet_posneg_1word_augmented_1x_sst2" \
--checkpoint-path "checkpoints/IMDb/original_augmented_1x_output_scheduling_warmup_lambda_01_${i}" \
--batch-size 6 \
--epoch 3 
done

