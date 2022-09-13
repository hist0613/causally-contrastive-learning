#!/bin/bash
#for i in "0" "1" "2"
#do
#python train_bert_imdb_pairwise_shellscript.py \
#--dataset-path "dataset/SST-2/triplet_automated_averaged_gradient_propensity_TVD_uniform_1word_augmented_1x_sst2/" \
#--output-path "checkpoints/SST-2/original_augmented_1x_output_scheduling_warmup_lambda_01_try3_${i}" \
#--batch-size 16 \
#--epoch 3 
#done

for i in "0" "1" "2"
do
python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  "dataset/IMDb/triplet_posneg_1word_augmented_1x_aclImdb" \
--checkpoint-path "checkpoints/SST-2/original_augmented_1x_output_scheduling_warmup_lambda_01_try3_${i}" \
--batch-size 6 \
--epoch 3 
done

for i in "0" "1" "2"
do
python evaluate_bert_imdb_pairwise_shellscript.py \
--dataset-path  "dataset/FineFood_full/triplet_posneg_1word_augmented_1x_finefood" \
--checkpoint-path "checkpoints/SST-2/original_augmented_1x_output_scheduling_warmup_lambda_01_try3_${i}" \
--batch-size 6 \
--epoch 3 
done


#--dataset-path "dataset/SST-2/triplet_automated_averaged_gradient_propensity_1word_augmented_1x_partition_02_sst2/" \
#--output-path "checkpoints/SST-2/triplet_automated_averaged_gradient_propensity_1word_augmented_1x_partition_02_output_scheduling_warmup_lambda_01_${i}" \

#echo "Model Running Completed. run expensive model..."
#cd ../ResponseSelection
#./running.sh

#--dataset-path "dataset/CFIMDb/triplet_automated_attention_1word_augmented_1x_aclImdb" \
#--output-path "checkpoints/CFIMDb/triplet_automated_attention_1word_augmented_1x_output_scheduling_warmup_${i}_renew" \
#--epoch 15 \
#--use-margin-loss


#--dataset-path "dataset/SST-2/triplet_automated_gradient_1word_augmented_1x_sst2" \
#--output-path "checkpoints/SST-2/triplet_automated_gradient_1word_augmented_1x_output_scheduling_warmup_lambda_001_${i}" \