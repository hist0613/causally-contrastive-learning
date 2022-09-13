#!/bin/bash
FINEFOOD_TESTSET="dataset/FineFood_full/triplet_posneg_1word_augmented_1x_finefood" 
IMDB_TESTSET="dataset/IMDb/triplet_posneg_1word_augmented_1x_aclImdb"
SST2_TESTSET="dataset/SST-2/triplet_posneg_1word_augmented_1x_sst2"

MODEL_PATH="checkpoints/IMDb/original_augmented_1x_output_scheduling_warmup_lambda_01"
echo "Vanilla"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $IMDB_TESTSET \
--checkpoint-path "${MODEL_PATH}_${i}" \
--batch-size 6 \
--epoch 3 
done

MODEL_PATH="checkpoints/IMDb/triplet_automated_averaged_gradient_1word_augmented_1x_output_scheduling_warmup_lambda_01"
echo "Grad"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $IMDB_TESTSET \
--checkpoint-path "${MODEL_PATH}_${i}" \
--batch-size 6 \
--epoch 3 
done

