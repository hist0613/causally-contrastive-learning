#!/bin/bash

ORIG_TEST="dataset/CFNLI/original_augmented_1x_cfnli"
RH_TEST="dataset/CFNLI/revised_hypothesis_augmented_1x_cfnli/"
RP_TEST="dataset/CFNLI/revised_premise_augmented_1x_cfnli/"
RHP_TEST="dataset/CFNLI/revised_combined_augmented_1x_cfnli/"

MODEL_PATH="checkpoints/CFNLI/ssmba_softed_augmented_5x_output_scheduling_warmup_lambda_01_try2"
TRAIN_DATASET="dataset/CFNLI/ssmba_softed_augmented_5x_cfnli/"

echo "================================"
echo "Train SSMBA"
echo "================================"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
--batch-size 16 \
--epoch 20 
#--use-margin-loss \
#--lambda-weight 0.1
done

echo "Test on Revised(RP)"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path $RP_TEST \
--checkpoint-path "${MODEL_PATH}_${i}" \
--batch-size 6 \
--epoch 3 
done

echo "Test on Revised(RH)"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path $RH_TEST \
--checkpoint-path "${MODEL_PATH}_${i}" \
--batch-size 6 \
--epoch 3 
done


echo "Test on Revised(Combined)"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path $RHP_TEST \
--checkpoint-path "${MODEL_PATH}_${i}" \
--batch-size 6 \
--epoch 3 
done

