#!/bin/bash

TELEPHONE_TESTSET="dataset/MultiNLI_telephone/original_augmented_1x_mnli/" 
LETTERS_TESTSET="dataset/MultiNLI_letters/original_augmented_1x_mnli/"
FACETOFACE_TESTSET="dataset/MultiNLI_facetoface/original_augmented_1x_mnli/"

MODEL_PATH="checkpoints/MultiNLI_telephone/triplets_automated_averaged_attention_1word_augmented_1x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/MultiNLI_telephone/triplets_automated_averaged_attention_1word_augmented_1x_mnli/"
:<<"END"
echo "================================"
echo "Attention"
echo "================================"
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


MODEL_PATH="checkpoints/MultiNLI_telephone/triplets_automated_averaged_gradient_propensity_1word_augmented_1x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/MultiNLI_telephone/triplets_automated_averaged_gradient_propensity_1word_augmented_1x_mnli/"

echo "================================"
echo "Hard Label"
echo "================================"
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

MODEL_PATH="checkpoints/MultiNLI_telephone/triplets_automated_averaged_gradient_propensity_TVD_1word_augmented_1x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/MultiNLI_telephone/triplets_automated_averaged_gradient_propensity_TVD_1word_augmented_1x_mnli/"

echo "================================"
echo "Soft-Label"
echo "================================"

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

MODEL_PATH="checkpoints/MultiNLI_telephone/triplets_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/MultiNLI_telephone/triplets_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_mnli/"

echo "================================"
echo "LM dropoout 0.5 Flip"
echo "================================"
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

MODEL_PATH="checkpoints/MultiNLI_telephone/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_1word_augmented_1x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/MultiNLI_telephone/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_1word_augmented_1x_mnli/"

echo "================================"
echo "LM dropoout 0.5 Flip"
echo "================================"
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


MODEL_PATH="checkpoints/MultiNLI_telephone/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_1word_augmented_1x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/MultiNLI_telephone/triplets_automated_averaged_gradient_LM_dropout_05_flip_sep_noneutral_1word_augmented_1x_mnli/"

echo "================================"
echo "LM dropoout 0.5 Flip, sep, no_neutral"
echo "================================"
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

END

MODEL_PATH="checkpoints/MultiNLI_telephone/ssmba_softed_augmented_5x_output_scheduling_warmup_lambda_01"
TRAIN_DATASET="dataset/MultiNLI_telephone/ssmba_softed_augmented_5x_mnli/"

echo "================================"
echo "SSMBA Training"
echo "================================"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript_stepsave.py \
--dataset-path ${TRAIN_DATASET} \
--output-path "${MODEL_PATH}_${i}" \
--batch-size 8 \
--epoch 3 
#--use-margin-loss \
#--lambda-weight 0.1
done

echo "Letters"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript_stepsave.py \
--dataset-path  $LETTERS_TESTSET \
--checkpoint-path "${MODEL_PATH}_${i}" \
--batch-size 6 \
--epoch 3 
done
echo "FacetoFace"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript_stepsave.py \
--dataset-path  $FACETOFACE_TESTSET \
--checkpoint-path "${MODEL_PATH}_${i}" \
--batch-size 6 \
--epoch 3 
done

