#!/bin/bash

ORIG_TEST="dataset/CFIMDb/original_augmented_1x_aclImdb/"
NEW_TEST="dataset/CFIMDb/revised_augmented_1x_aclImdb/"

echo "------------------"
echo "Vanilla"
echo "------------------"
MODEL_NAME="original_augmented_1x_output_scheduling_warmup_lambda_01"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path $ORIG_TEST \
--output-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 20
#--use-margin-loss \
#--lambda-weight 0.1
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

