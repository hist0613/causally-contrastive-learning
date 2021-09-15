#!/bin/bash

ORIG_TEST="dataset/CFIMDb/original_augmented_1x_aclImdb/"
NEW_TEST="dataset/CFIMDb/revised_augmented_1x_aclImdb/"

MODEL_NAME="triplet_automated_averaged_gradient_wanglike_1word_augmented_1x_output_scheduling_warmup_lambda_01"
echo "CFIMDb-WangLike"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/CFIMDb/triplet_automated_averaged_gradient_wanglike_1word_augmented_1x_aclImdb/" \
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




TELEPHONE_TESTSET="dataset/MultiNLI_telephone/original_augmented_1x_mnli/" 
LETTERS_TESTSET="dataset/MultiNLI_letters/original_augmented_1x_mnli/"
FACETOFACE_TESTSET="dataset/MultiNLI_facetoface/original_augmented_1x_mnli/"


MODEL_PATH="checkpoints/MultiNLI_telephone/triplets_automated_averaged_gradient_wanglike_1word_augmented_1x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/MultiNLI_telephone/triplets_automated_averaged_gradient_wanglike_1word_augmented_1x_mnli/"

echo "nli wanglike"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.1
done

echo "Letters"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $LETTERS_TESTSET \
--checkpoint-path "${MODEL_PATH}_${i}" \
--batch-size 6 \
--epoch 3 
done
echo "FacetoFace"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $FACETOFACE_TESTSET \
--checkpoint-path "${MODEL_PATH}_${i}" \
--batch-size 6 \
--epoch 3 
done
