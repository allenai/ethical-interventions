import json
import os,sys
from tqdm import tqdm
sys.path.append("./..")
import analysis
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split

all_examples = {}
for intv in ['ethical', 'adversarial', 'irrelevant']:
    preds = json.load(open(f'./../data/gender/robertalarge_gender-{intv}.output.json', 'r'))
    paired = analysis.pairup_ex(preds)
    print('In {0}, {1} example pairs extracted.'.format(intv, len(paired)))
    all_examples.update(paired)
    
def noIntvOverlap(all_examples):
    intv2expairs = defaultdict(list)
    for key in all_examples:
        intv_single = key[-1]
        intv2expairs[intv_single].append(key)

    allkeys = list(intv2expairs.keys())
    train_dev, test = train_test_split(allkeys, test_size=0.2, random_state=42)
    train, dev = train_test_split(train_dev, test_size=0.2, random_state=42)
    print("train exp:", train[:3])
    assert set(train) & set(dev) & set(test) != ()
    print(f"{len(train)} train intv, {len(dev)} dev intv, {len(test)} test intv")
    train_keys = [x for intvs in train for x in intv2expairs[intvs]]
    dev_keys = [x for intvs in dev for x in intv2expairs[intvs]]
    test_keys = [x for intvs in test for x in intv2expairs[intvs]]
    random.shuffle(train_keys)
    random.shuffle(dev_keys)
    random.shuffle(test_keys)
    assert set(train_keys) & set(dev_keys) & set(test_keys) != ()
    train_exp = {"|".join(list(x[0]) + list(x[1:])): all_examples[x]  for x in train_keys}
    dev_exp = {"|".join(list(x[0]) + list(x[1:])): all_examples[x]  for x in dev_keys}
    test_exp = {"|".join(list(x[0]) + list(x[1:])): all_examples[x]  for x in test_keys}

    json.dump(train_exp, open('./../data/noIntvOverlap/train.json', 'w'))
    json.dump(dev_exp, open('./../data/noIntvOverlap/dev.json', 'w'))
    json.dump(test_exp, open('./../data/noIntvOverlap/test.json', 'w'))


def random_split(all_examples):
    allkeys = list(all_examples.keys())
    train_dev, test = train_test_split(allkeys, test_size=0.2, random_state=42)
    train, dev = train_test_split(train_dev, test_size=0.2, random_state=42)
    print(f"{len(train)} train, {len(dev)} dev, {len(test)} test")
    train_examples = {"|".join(list(x[0]) + list(x[1:])): all_examples[x] for x in train}
    dev_examples = {"|".join(list(x[0]) + list(x[1:])): all_examples[x] for x in dev}
    test_examples = {"|".join(list(x[0]) + list(x[1:])): all_examples[x] for x in test}
    json.dump(train_examples, open('./../data/gender/randomsplit/train.json', 'w'))    
    json.dump(dev_examples, open('./../data/gender/randomsplit/dev.json', 'w'))
    json.dump(test_examples, open('./../data/gender/randomsplit/test.json', 'w'))

if __name__=='__main__':
    random_split(all_examples)