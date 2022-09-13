#!/bin/bash

cd utils
python Wanglike_filtering.py
cd ..


FINEFOOD_TESTSET="dataset/FineFood_full/triplet_posneg_1word_augmented_1x_finefood" 
IMDB_TESTSET="dataset/IMDb/triplet_posneg_1word_augmented_1x_aclImdb"
SST2_TESTSET="dataset/SST-2/triplet_posneg_1word_augmented_1x_sst2"


MODEL_NAME="triplet_automated_averaged_wanglike_1word_augmented_1x_output_scheduling_warmup_lambda_01"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/SST-2/triplet_automated_averaged_gradient_wanglike_1word_augmented_1x_sst2/" \
--output-path "checkpoints/SST-2/${MODEL_NAME}_${i}" \
--batch-size 16 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.1
done
echo "FineFood"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $FINEFOOD_TESTSET \
--checkpoint-path "checkpoints/SST-2/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/SST-2/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done


