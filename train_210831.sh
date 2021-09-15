#!/bin/bash

ORIG_TEST="dataset/CFNLI/original_augmented_1x_cfnli"
RH_TEST="dataset/CFNLI/revised_hypothesis_augmented_1x_cfnli/"
RP_TEST="dataset/CFNLI/revised_premise_augmented_1x_cfnli/"
RHP_TEST="dataset/CFNLI/revised_combined_augmented_1x_cfnli/"

:<<"END"
MODEL_PATH="checkpoints/CFNLI/original_augmented_1x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/CFNLI/original_augmented_1x_cfnli/"
echo "================================"
echo "Train Original"
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

MODEL_PATH="checkpoints/CFNLI/triplets_posneg_1word_augmented_1x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/CFNLI/triplets_posneg_1word_augmented_1x_cfnli/"
echo "================================"
echo "POS/NEG"
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
END

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_attention_1word_augmented_1x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_attention_1word_augmented_1x_cfnli/"
echo "================================"
echo "attention"
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

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_1word_augmented_1x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_1word_augmented_1x_cfnli/"
echo "================================"
echo "Gradient"
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

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_propensity_flip_1word_augmented_1x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_propensity_flip_1word_augmented_1x_cfnli/"
echo "================================"
echo "Propensity(HL)"
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

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_propensity_TVD_1word_augmented_1x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_propensity_TVD_1word_augmented_1x_cfnli/"
echo "================================"
echo "Propensity(SL)"
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

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_cfnli/"
echo "================================"
echo "LM-based"
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

MODEL_PATH="checkpoints/CFNLI/triplets_automated_averaged_gradient_wanglike_1word_augmented_1x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/CFNLI/triplets_automated_averaged_gradient_wanglike_1word_augmented_1x_cfnli/"
echo "================================"
echo "LM-based"
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

