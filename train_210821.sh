#!/bin/bash

ORIG_TEST="dataset/CFIMDb/original_augmented_1x_aclImdb/"
NEW_TEST="dataset/CFIMDb/revised_augmented_1x_aclImdb/"

MODEL_NAME="ssmba_augmented_5x_output_scheduling_warmup_lambda_01"
echo "ssmba"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/CFIMDb/ssmba_augmented_5x_aclImdb/" \
--output-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.1
done

echo "Test on Revised(CF)"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path $NEW_TEST \
--checkpoint-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

MODEL_NAME="ssmba_softed_augmented_5x_output_scheduling_warmup_lambda_01"
echo "ssmba_softed"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/CFIMDb/ssmba_softed_augmented_5x_aclImdb/" \
--output-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.1
done

echo "Test on Revised(CF)"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path $NEW_TEST \
--checkpoint-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done



MODEL_NAME="triplet_automated_averaged_attention_1word_augmented_1x_output_scheduling_warmup_lambda_01"
echo "Attention"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/CFIMDb/triplet_automated_averaged_attention_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.1
done

echo "Test on Revised(CF)"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path $NEW_TEST \
--checkpoint-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done



MODEL_NAME="triplet_automated_averaged_gradient_propensity_flip_1word_augmented_1x_output_scheduling_warmup_lambda_01"
echo "Flip"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/CFIMDb/triplet_automated_averaged_gradient_propensity_flip_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.1
done

echo "Test on Revised(CF)"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path $NEW_TEST \
--checkpoint-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done



MODEL_NAME="triplet_automated_averaged_gradient_propensity_TVD_1word_augmented_1x_output_scheduling_warmup_lambda_01"
echo "TVD"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/CFIMDb/triplet_automated_averaged_gradient_propensity_TVD_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.1
done

echo "Test on Revised(CF)"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path $NEW_TEST \
--checkpoint-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done


