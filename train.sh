#!/bin/bash

FINEFOOD_TESTSET="dataset/FineFood_full/triplet_posneg_1word_augmented_1x_finefood" 
IMDB_TESTSET="dataset/IMDb/triplet_posneg_1word_augmented_1x_aclImdb"
SST2_TESTSET="dataset/SST-2/triplet_posneg_1word_augmented_1x_sst2"
:<<"END"
MODEL_NAME="triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_output_scheduling_warmup_lambda_03"
echo "Test lambda 03"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.3
done
echo "FineFood"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $FINEFOOD_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

MODEL_NAME="triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_output_scheduling_warmup_lambda_05"
echo "Test lambda 05"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.5
done
echo "FineFood"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $FINEFOOD_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

MODEL_NAME="triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_output_scheduling_warmup_lambda_07"
echo "Test lambda 07"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.7
done
echo "FineFood"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $FINEFOOD_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

MODEL_NAME="triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_output_scheduling_warmup_lambda_1"
echo "Test lambda 1"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 1.0
done
echo "FineFood"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $FINEFOOD_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

MODEL_NAME="triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_output_scheduling_warmup_lambda_05_try2"
echo "Test lambda 0.5"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.5
done
echo "FineFood"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $FINEFOOD_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

MODEL_NAME="triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_output_scheduling_warmup_lambda_04"
echo "Test lambda 0.4"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.4
done
echo "FineFood"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $FINEFOOD_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

MODEL_NAME="triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_output_scheduling_warmup_lambda_03_try2"
echo "Test lambda 0.3"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.3
done
echo "FineFood"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $FINEFOOD_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=4,5 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done


MODEL_NAME="ssmba_softed_augmented_5x_output_scheduling_warmup_lambda_01"
echo "ssmba_training"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript_stepsave.py \
--dataset-path "dataset/FineFood_full/ssmba_softed_augmented_5x_finefood/" \
--output-path "checkpoints/FineFood_full/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 
#--use-margin-loss \
#--lambda-weight 0.3
done
echo "IMDb"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $IMDB_TESTSET \
--checkpoint-path "checkpoints/FineFood_full/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done
echo "SST2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/FineFood_full/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done


MODEL_NAME="ssmba_softed_augmented_5x_output_scheduling_warmup_lambda_01"
echo "ssmba_training"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript_stepsave.py \
--dataset-path "dataset/IMDb/ssmba_softed_augmented_5x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 
#--use-margin-loss \
#--lambda-weight 0.3
done
echo "FineFood"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $FINEFOOD_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done
echo "SST2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

END

MODEL_NAME="ssmba_softed_augmented_5x_output_scheduling_warmup_lambda_01_ver1"
echo "ssmba_training"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/SST-2/ssmba_softed_augmented_5x_sst2.bak/" \
--output-path "checkpoints/SST-2/${MODEL_NAME}_${i}" \
--batch-size 16 \
--epoch 3 
#--use-margin-loss \
#--lambda-weight 0.3
done
echo "FineFood"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $FINEFOOD_TESTSET \
--checkpoint-path "checkpoints/SST-2/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done
echo "IMDb"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $IMDB_TESTSET \
--checkpoint-path "checkpoints/SST-2/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done


