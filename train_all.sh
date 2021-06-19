#!/bin/bash
#echo "Model: Propensity_Uniform"
#for i in "0" "1" "2"
#do
#python train_bert_imdb_pairwise_shellscript.py \
#--dataset-path "dataset/FineFood_full/triplet_automated_averaged_gradient_propensity_TVD_uniform_1word_augmented_1x_finefood/" \
#--output-path "checkpoints/FineFood_full/triplet_automated_averaged_gradient_propensity_TVD_uniform_1word_augmented_1x_output_scheduling_warmup_lambda_01_${i}" \
#--batch-size 8 \
#--epoch 3 \
#--use-margin-loss
#done

echo "Model: Gradient"
for i in "0" "1" "2"
do
python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplet_automated_averaged_gradient_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/triplet_automated_averaged_gradient_1word_augmented_1x_output_scheduling_warmup_lambda_01_${i}" \
--batch-size 8 \
--epoch 3 \
--use-margin-loss
done

echo "Model: VANILLA"
for i in "0" "1" "2"
do
python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb/triplet_automated_averaged_gradient_1word_augmented_1x_aclImdb/" \
--output-path "checkpoints/IMDb/original_augmented_1x_output_scheduling_warmup_lambda_01_${i}" \
--batch-size 8 \
--epoch 3 
done
