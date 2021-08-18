#!/bin/bash

FINEFOOD_TESTSET="dataset/FineFood_full/triplet_posneg_1word_augmented_1x_finefood" 
IMDB_TESTSET="dataset/IMDb/triplet_posneg_1word_augmented_1x_aclImdb"
SST2_TESTSET="dataset/SST-2/triplet_posneg_1word_augmented_1x_sst2"
:<<"END"
echo "============================="
echo "Lambda 0.1, ver2"
echo "============================="
MODEL_NAME="triplet_automated_averaged_gradient_LM_dropout_05_flip_multi_2_same_neg_ver2_1word_augmented_1x_output_scheduling_warmup_lambda_01"
echo "Test New Training"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_multi_pairwise_shellscript.py \
--dataset-path "dataset/SST-2/triplet_automated_averaged_gradient_LM_dropout_05_flip_multi_2_same_neg_ver2_1word_augmented_1x_sst2" \
--output-path "checkpoints/SST-2/${MODEL_NAME}_${i}" \
--batch-size 16 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.1
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
echo "FineFood"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $FINEFOOD_TESTSET \
--checkpoint-path "checkpoints/SST-2/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

echo "============================="
echo "Lambda 0.05, ver2"
echo "============================="
MODEL_NAME="triplet_automated_averaged_gradient_LM_dropout_05_flip_multi_2_same_neg_ver2_1word_augmented_1x_output_scheduling_warmup_lambda_005"
echo "Test New Training"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_multi_pairwise_shellscript.py \
--dataset-path "dataset/SST-2/triplet_automated_averaged_gradient_LM_dropout_05_flip_multi_2_same_neg_ver2_1word_augmented_1x_sst2" \
--output-path "checkpoints/SST-2/${MODEL_NAME}_${i}" \
--batch-size 16 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.05
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
echo "FineFood"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $FINEFOOD_TESTSET \
--checkpoint-path "checkpoints/SST-2/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done

echo "============================="
echo "Lambda 0.05"
echo "============================="
MODEL_NAME="triplet_automated_averaged_gradient_LM_dropout_05_flip_multi_2_same_neg_1word_augmented_1x_output_scheduling_warmup_lambda_005"
echo "Test New Training"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_multi_pairwise_shellscript.py \
--dataset-path "dataset/SST-2/triplet_automated_averaged_gradient_LM_dropout_05_flip_multi_2_same_neg_1word_augmented_1x_sst2/" \
--output-path "checkpoints/SST-2/${MODEL_NAME}_${i}" \
--batch-size 16 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.05
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
echo "FineFood"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $FINEFOOD_TESTSET \
--checkpoint-path "checkpoints/SST-2/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done


echo "============================="
echo "Lambda 0.1"
echo "============================="
MODEL_NAME="triplet_automated_averaged_gradient_LM_dropout_05_flip_multi_2_same_neg_1word_augmented_1x_output_scheduling_warmup_lambda_01_try3"
echo "Test New Training"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python train_bert_imdb_multi_pairwise_shellscript.py \
--dataset-path "dataset/SST-2/triplet_automated_averaged_gradient_LM_dropout_05_flip_multi_2_same_neg_1word_augmented_1x_sst2/" \
--output-path "checkpoints/SST-2/${MODEL_NAME}_${i}" \
--batch-size 16 \
--epoch 3 \
--use-margin-loss \
--lambda-weight 0.1
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
echo "FineFood"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $FINEFOOD_TESTSET \
--checkpoint-path "checkpoints/SST-2/${MODEL_NAME}_${i}" \
--batch-size 6 \
--epoch 3 
done
END

