#!/bin/bash

FINEFOOD_TESTSET="dataset/FineFood_full/triplet_posneg_1word_augmented_1x_finefood" 
IMDB_TESTSET="dataset/IMDb/triplet_posneg_1word_augmented_1x_aclImdb"
SST2_TESTSET="dataset/SST-2/triplet_posneg_1word_augmented_1x_sst2"
:<<"END"
MODEL_NAME="triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_2_1word_augmented_1x_output_scheduling_warmup_lambda_01"
echo "SentTokenize Training, lambda 01"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_2_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.1
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
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

MODEL_NAME="triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_2_1word_augmented_1x_output_scheduling_warmup_lambda_03"
echo "SentTokenize Training, lambda 03"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_2_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.3
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
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

MODEL_NAME="triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_2_1word_augmented_1x_output_scheduling_warmup_lambda_05"
echo "SentTokenize Training, lambda 05"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_2_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.5
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
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

MODEL_NAME="triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_2_1word_augmented_1x_output_scheduling_warmup_lambda_07"
echo "SentTokenize Training, lambda 07"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_2_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.7
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
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

MODEL_NAME="triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_2_1word_augmented_1x_output_scheduling_warmup_lambda_1"
echo "SentTokenize Training, lambda 1"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_2_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 1.0
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
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done
END

MODEL_NAME="triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_4_1word_augmented_1x_output_scheduling_warmup_lambda_01"
echo "SentTokenize Training, lambda 01"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_4_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.1
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
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

MODEL_NAME="triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_4_1word_augmented_1x_output_scheduling_warmup_lambda_03"
echo "SentTokenize Training, lambda 03"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_4_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.3
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
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

MODEL_NAME="triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_4_1word_augmented_1x_output_scheduling_warmup_lambda_05"
echo "SentTokenize Training, lambda 05"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_4_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.5
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
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

MODEL_NAME="triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_4_1word_augmented_1x_output_scheduling_warmup_lambda_07"
echo "SentTokenize Training, lambda 05"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_4_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.7
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
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done


MODEL_NAME="triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_4_1word_augmented_1x_output_scheduling_warmup_lambda_1"
echo "SentTokenize Training, lambda 1"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_aggregated_limit_4_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 1.0
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
echo "SST-2"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $SST2_TESTSET \
--checkpoint-path "checkpoints/IMDb/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

