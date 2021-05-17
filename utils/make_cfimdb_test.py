import json
import pickle
import os

OUTPUT_PATH = "../dataset/CFIMDb/pure_test"
os.makedirs(OUTPUT_PATH)

with open("../dataset/CFIMDb/cf_augmented_examples/paired_test.pickle", 'rb') as f:
    data = pickle.load(f)

output = []
cnt = 0
for d in data:
    for s in d:
        sample = dict()
        sample['id'] = cnt
        sample['anchor_text'] = s[1]
        sample['positive_text'] = ''
        sample['negative_text'] = ''
        sample['triplet_sample_mask'] = False
        if s[0] == "Negative":
            sample['label'] = [1., 0.]
        else:
            sample['label'] = [0., 1.]

        output.append(sample)
        cnt += 1

with open(os.path.join(OUTPUT_PATH, "test.json"), 'w') as f:
    json.dump(output, f)


