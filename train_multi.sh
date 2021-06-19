#!/bin/bash

FINEFOOD_TESTSET="dataset/FineFood_full/triplet_posneg_1word_augmented_1x_finefood" 
IMDB_TESTSET="dataset/IMDb/triplet_posneg_1word_augmented_1x_aclImdb"
SST2_TESTSET="dataset/SST-2/triplet_posneg_1word_augmented_1x_sst2"

MODEL_PATH="checkpoints/SST-2/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_output_scheduling_warmup_lambda_01_tmp"
TRAIN_DATASET="dataset/SST-2/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_sst2/"

MODEL_NAME="test"
echo "Test New Training"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplet_automated_averaged_gradient_propensity_TVD_uniform_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.3
done
echo "FineFood"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $FINEFOOD_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done



:<<"END"

echo "================================"
echo "LM dropoout 0.5, memory_test"
echo "================================"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
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
--checkpoint-path "${MODEL_PATH}_${i}" \
--batch-size 6 \
--epoch 3 
done
echo "IMDb"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $IMDB_TESTSET \
--checkpoint-path "${MODEL_PATH}_${i}" \
--batch-size 6 \
--epoch 3 
done

END
