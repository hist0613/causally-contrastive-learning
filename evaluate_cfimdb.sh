#!/bin/bash
TESTSET="dataset/CFIMDb/pure_test"

echo "\n\n\nModel:  PosNeg\n\n\n"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=5,6,7 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $TESTSET \
--checkpoint-path "checkpoints/IMDb/triplet_posneg_1word_augmented_1x_output_scheduling_warmup_lambda_01_${i}" \
--batch-size 6 \
--epoch 3 
done

echo "\n\n\nModel: Attention\n\n\n"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=5,6,7 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $TESTSET \
--checkpoint-path "checkpoints/IMDb/triplet_automated_averaged_attention_1word_augmented_1x_output_scheduling_warmup_lambda_01_${i}" \
--batch-size 6 \
--epoch 3 
done

echo "\n\n\nModel: HL\n\n\n"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=5,6,7 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $TESTSET \
--checkpoint-path "checkpoints/IMDb/triplet_automated_averaged_gradient_propensity_flip_1word_augmented_1x_output_scheduling_warmup_lambda_01_${i}" \
--batch-size 6 \
--epoch 3 
done

echo "\n\n\nModel: SL\n\n\n"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=5,6,7 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $TESTSET \
--checkpoint-path "checkpoints/IMDb/triplet_automated_averaged_gradient_propensity_TVD_1word_augmented_1x_output_scheduling_warmup_lambda_01_0/_${i}" \
--batch-size 6 \
--epoch 3 
done


echo "\n\n\nModel: MR\n\n\n"
for i in "0" "1" "2"
do
CUDA_VISIBLE_DEVICES=5,6,7 python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  $TESTSET \
--checkpoint-path "checkpoints/IMDb/triplet_automated_averaged_gradient_LM_dropout_05_flip_1word_augmented_1x_output_scheduling_warmup_lambda_05_try2_${i}" \
--batch-size 6 \
--epoch 3 
done

