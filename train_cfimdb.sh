#!/bin/bash

ORIG_TEST="dataset/CFIMDb/original_augmented_1x_aclImdb/"
NEW_TEST="dataset/CFIMDb/revised_augmented_1x_aclImdb/"

MODEL_NAME="original_augmented_1x_output_scheduling_warmup_lambda_01"


echo "Test on Revised(CF)"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path $NEW_TEST \
--checkpoint-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

