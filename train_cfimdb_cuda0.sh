#!/bin/bash

TRAIN_SET="dataset/CFIMDb/triplet_automated_averaged_gradient_1word_augmented_1x_aclImdb"
ORIG_TEST="dataset/CFIMDb/original_augmented_1x_aclImdb/"
NEW_TEST="dataset/CFIMDb/revised_augmented_1x_aclImdb/"

echo "------------------"
echo "Lambda 0.1"
echo "------------------"
MODEL_NAME="triplet_automated_averaged_gradient_1word_augmented_1x_output_scheduling_warmup_lambda_01"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path $TRAIN_SET \
--output-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 4 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.1
done

echo "Test on ORIGINAL"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path $ORIG_TEST \
--checkpoint-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

echo "Test on Revised(CF)"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path $NEW_TEST \
--checkpoint-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done
