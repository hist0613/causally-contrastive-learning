#!/bin/bash

ORIG_TEST="dataset/CFIMDb/original_augmented_1x_aclImdb/"
NEW_TEST="dataset/CFIMDb/revised_augmented_1x_aclImdb/"

:<<"END"
echo "------------------"
echo "lambda 0.1"
echo "------------------"
MODEL_NAME="triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_output_scheduling_warmup_lambda_01"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/CFIMDb/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_aclImdb/" \
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

echo "------------------"
echo "lambda 0.2"
echo "------------------"
MODEL_NAME="triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_output_scheduling_warmup_lambda_02"
echo "ssmba_training"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/CFIMDb/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.2
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

echo "------------------"
echo "lambda 0.3"
echo "------------------"
MODEL_NAME="triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_output_scheduling_warmup_lambda_03"
echo "ssmba_training"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/CFIMDb/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.3
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

echo "------------------"
echo "lambda 0.4"
echo "------------------"
MODEL_NAME="triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_output_scheduling_warmup_lambda_04"
echo "ssmba_training"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/CFIMDb/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.4
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

echo "------------------"
echo "lambda 0.5"
echo "------------------"
MODEL_NAME="triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_output_scheduling_warmup_lambda_05"
echo "ssmba_training"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/CFIMDb/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.5
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

echo "------------------"
echo "lambda 0.6"
echo "------------------"
MODEL_NAME="triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_output_scheduling_warmup_lambda_06"
echo "ssmba_training"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/CFIMDb/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.6
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
echo "------------------"
echo "lambda 0.7"
echo "------------------"
MODEL_NAME="triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_output_scheduling_warmup_lambda_07"
echo "ssmba_training"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/CFIMDb/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/CFIMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 20 \
--use-margin-loss \
--lambda-weight 0.7
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
END

echo "------------------"
echo "Pos/Neg"
echo "------------------"
MODEL_NAME="triplet_posneg_1word_augmented_1x_output_scheduling_warmup_lambda_01"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/CFIMDb/triplet_posneg_1word_augmented_1x_aclImdb/" \
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

:<<"END"
echo "------------------"
echo "gradient"
echo "------------------"
MODEL_NAME="triplet_automated_averaged_gradient_1word_augmented_1x_output_scheduling_warmup_lambda_01"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/CFIMDb/triplet_automated_averaged_gradient_1word_augmented_1x_aclImdb/" \
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

END
