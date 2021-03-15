#!/bin/bash

for i in "0" "1" "2"
do
python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplet_posneg_1word_augmented_1x_aclImdb" \
--output-path "checkpoints/IMDb/triplet_posneg_1word_augmented_1x_output_scheduling_warmup_lambda_01_${i}" \
--batch-size 16 \
--epoch 3 \
--use-margin-loss
done

echo "Model Running Completed. run expensive model..."
cd ../ResponseSelection
./running.sh

#--dataset-path "dataset/CFIMDb/triplet_automated_attention_1word_augmented_1x_aclImdb" \
#--output-path "checkpoints/CFIMDb/triplet_automated_attention_1word_augmented_1x_output_scheduling_warmup_${i}_renew" \
#--epoch 15 \
#--use-margin-loss
