#!/bin/bash
for i in "0" "1" "2"
do
python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  "dataset/FineFood_full/triplet_posneg_1word_augmented_1x_finefood" \
--checkpoint-path "checkpoints/IMDb/triplet_automated_averaged_gradient_propensity_TVD_uniform_1word_augmented_1x_output_scheduling_warmup_lambda_01_${i}" \
--batch-size 6 \
--epoch 3 
done

