import pickle
import os
DATASET_NAME = "IMDb"
PICKLE_PATH = f"../dataset/{DATASET_NAME}/cf_augmented_examples"
PICKLE_NAME = "triplets_automated_averaged_gradient_LM_dropout_05_sentTokenize_sampling1_augmenting1_train"

with open(os.path.join(PICKLE_PATH, f"{PICKLE_NAME}_cuda0.pickle"), 'rb') as f:
    data0 = pickle.load(f)

with open(os.path.join(PICKLE_PATH, f"{PICKLE_NAME}_cuda1.pickle"), 'rb') as f:
    data1 = pickle.load(f)

with open(os.path.join(PICKLE_PATH, f"{PICKLE_NAME}_cuda2.pickle"), 'rb') as f:
    data2 = pickle.load(f)

data = data0 + data1 + data2

with open(os.path.join(PICKLE_PATH, f"{PICKLE_NAME}.pickle"), 'wb') as f:
    pickle.dump(data, f)

