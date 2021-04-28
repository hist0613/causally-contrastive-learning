#!/bin/bash
for i in "0" "1" "2"
do
python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  "dataset/IMDb/triplet_posneg_1word_augmented_1x_aclImdb" \
--checkpoint-path "checkpoints/SST-2/triplet_automated_averaged_gradient_propensity_TVD_uniform_THRESHOLD_01_1word_augmented_1x_output_scheduling_warmup_lambda_01_clean_try2_${i}" \
--batch-size 6 \
--epoch 3 
done

