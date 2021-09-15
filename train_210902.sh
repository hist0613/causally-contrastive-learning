#!/bin/bash

ORIG_TEST="dataset/CFNLI/original_augmented_1x_cfnli"
RH_TEST="dataset/CFNLI/revised_hypothesis_augmented_1x_cfnli/"
RP_TEST="dataset/CFNLI/revised_premise_augmented_1x_cfnli/"
RHP_TEST="dataset/CFNLI/revised_combined_augmented_1x_cfnli/"

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_1word_augmented_1x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_1word_augmented_1x_cfnli/"
echo "================================"
echo "LM-based, lambda 0.1"
echo "================================"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
--batch-size 16 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.1
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

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_1word_augmented_1x_output_scheduling_warmup_lambda_03"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_1word_augmented_1x_cfnli/"
echo "================================"
echo "LM-based, lambda 0.3"
echo "================================"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
--batch-size 16 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.3
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

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_1word_augmented_1x_output_scheduling_warmup_lambda_05"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_1word_augmented_1x_cfnli/"
echo "================================"
echo "LM-based, lambda 0.5"
echo "================================"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
--batch-size 16 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.5
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

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_1word_augmented_1x_output_scheduling_warmup_lambda_07"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_1word_augmented_1x_cfnli/"
echo "================================"
echo "LM-based, lambda 0.7"
echo "================================"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
--batch-size 16 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.7
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

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_1word_augmented_1x_output_scheduling_warmup_lambda_1"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_1word_augmented_1x_cfnli/"
echo "================================"
echo "LM-based, lambda 1.0"
echo "================================"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
--batch-size 16 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 1.0
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

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_1word_augmented_1x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_1word_augmented_1x_cfnli/"
echo "================================"
echo "NO Neutral, lambda 0.1"
echo "================================"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
--batch-size 16 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.1
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


MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_1word_augmented_1x_output_scheduling_warmup_lambda_03"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_1word_augmented_1x_cfnli/"
echo "================================"
echo "NO Neutral, lambda 0.3"
echo "================================"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
--batch-size 16 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.3
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

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_1word_augmented_1x_output_scheduling_warmup_lambda_05"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_1word_augmented_1x_cfnli/"
echo "================================"
echo "No Neutral, lambda 0.5"
echo "================================"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
--batch-size 16 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.5
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

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_1word_augmented_1x_output_scheduling_warmup_lambda_07"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_1word_augmented_1x_cfnli/"
echo "================================"
echo "NO neutral, lambda 0.7"
echo "================================"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
--batch-size 16 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.7
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

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_1word_augmented_1x_output_scheduling_warmup_lambda_1"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_1word_augmented_1x_cfnli/"
echo "================================"
echo "No Neutral, lambda 1.0"
echo "================================"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
--batch-size 16 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 1.0
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

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_ver2_1word_augmented_1x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_ver2_1word_augmented_1x_cfnli/"
echo "================================"
echo "NO Neutral ver 2, lambda 0.1"
echo "================================"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
--batch-size 16 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.1
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

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_ver2_1word_augmented_1x_output_scheduling_warmup_lambda_03"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_ver2_1word_augmented_1x_cfnli/"
echo "================================"
echo "NO Neutral ver 2, lambda 0.3"
echo "================================"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
--batch-size 16 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.3
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

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_ver2_1word_augmented_1x_output_scheduling_warmup_lambda_05"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_ver2_1word_augmented_1x_cfnli/"
echo "================================"
echo "NO Neutral ver 2, lambda 0.5"
echo "================================"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
--batch-size 16 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.5
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

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_ver2_1word_augmented_1x_output_scheduling_warmup_lambda_07"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_ver2_1word_augmented_1x_cfnli/"
echo "================================"
echo "NO Neutral ver 2, lambda 0.7"
echo "================================"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
--batch-size 16 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.7
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

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_ver2_1word_augmented_1x_output_scheduling_warmup_lambda_1"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_ver2_1word_augmented_1x_cfnli/"
echo "================================"
echo "NO Neutral ver 2, lambda 1.0"
echo "================================"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
--batch-size 16 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 1.0
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







