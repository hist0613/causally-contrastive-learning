#!/bin/bash
TESTSET="dataset/CFIMDb/pure_test"

echo "\n\n\nModel:  Vanilla\n\n\n"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $TESTSET \
--checkpoint-path "checkpoints/IMDb/original_augmented_1x_output_scheduling_warmup_lambda_01_${i}" \
--batch-size 6 \
--epoch 3 
done

echo "\n\n\nModel: Gradient\n\n\n"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $TESTSET \
--checkpoint-path "checkpoints/IMDb/triplet_automated_averaged_gradient_1word_augmented_1x_output_scheduling_warmup_lambda_01_${i}" \
--batch-size 6 \
--epoch 3 
done

