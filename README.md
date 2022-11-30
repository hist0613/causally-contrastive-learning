# C2L: Causally Contrastive Learning for Robust Text Classification
Official pytorch implementation of [**C2L: Causally Contrastive Learning for Robust Text Classification**](https://ojs.aaai.org/index.php/AAAI/article/download/21296/version/19583/21045) (AAAI 2022) by *Seungtaek Choi*, Myeongho Jeong*, Hojae Han, Seung-won Hwang*.

## Setup
In this repository, we only treat CFIMDb dataset as an example. This also works on another datasets which are mentioned in our paper.
### Requirements
```
pip install -r requirements.txt
```

### Download Dataset
You can download dataset from the repository of [Learning the Difference that Makes a Difference with Counterfactually-Augmented Data](https://github.com/acmi-lab/counterfactually-augmented-data).
Then, please put the dataset into `dataset/CFIMDb/aclImdb/` and run the python script below.
```
cd utils
python reform_cfimdb_dataset.py
cd ..
```

However, we already pre-process the data for our training code. you can just clone this repository and run the training script below.
Also, there also exists the dataset augmented with our approach. You can train with this dataset directly.

## Train & Evaluate
### Train vanilla model
```
bash train_cfimdb_public.sh
```
### Generate counterfactually masked samples
To generate counterfactually masked samples, we provide a notebook [pairing-data-ours-public.ipynb](https://github.com/hist0613/counterfactual-robustness/blob/public_clean_code/pairing-data-ours-public.ipynb). Please run all shells sequentially. After that, please run the code below to reform the output to trainable dataset.
```
cd utils
python triplets_masking_dataset.py
cd ..
```
### Train model with C2L
```
bash train_cfimdb_ours_public.sh
```
