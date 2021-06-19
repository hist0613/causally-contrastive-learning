#!/bin/bash
:<<"END"
echo "Wang-like, partition 025"
for i in "2"
do
python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb//triplet_automated_averaged_gradient_wanglike_1word_augmented_1x_partition_025_aclImdb/" \
--output-path "checkpoints/IMDb/triplet_automated_averaged_gradient_wanglike_1word_augmented_1x_partition_025_output_scheduling_warmup_lambda_01_try2_${i}" \
--batch-size 8 \
--epoch 12 \
--use-margin-loss
done
END

echo "Wang-like, partition 050"
for i in "2"
do
python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb//triplet_automated_averaged_gradient_wanglike_1word_augmented_1x_partition_050_aclImdb/" \
--output-path "checkpoints/IMDb/triplet_automated_averaged_gradient_wanglike_1word_augmented_1x_partition_050_output_scheduling_warmup_lambda_01_${i}" \
--batch-size 8 \
--epoch 6 \
--use-margin-loss
done

echo "Wang-like, partition 075"
for i in "2"
do
python train_bert_imdb_pairwise_shellscript.py \
--dataset-path "dataset/IMDb//triplet_automated_averaged_gradient_wanglike_1word_augmented_1x_partition_075_aclImdb/" \
--output-path "checkpoints/IMDb/triplet_automated_averaged_gradient_wanglike_1word_augmented_1x_partition_075_output_scheduling_warmup_lambda_01_${i}" \
--batch-size 8 \
--epoch 4 \
--use-margin-loss
done

