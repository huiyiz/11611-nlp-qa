# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
Used to preprocess toy dataset - first 100 lines of psgs_w100.tsv
NQ dataset only. Trivia removed.

"""
import sys
import json
import parser
from pathlib import Path
import numpy as np
import util

def select_examples_NQ(data, index, passages, passages_index):
    selected_data = []
    for i, k in enumerate(index):
        ctxs = [
                {
                    'id': idx,
                    'title': passages[idx][1],
                    'text': passages[idx][0],
                }
                for idx in passages_index[str(i)] if passages.get(idx, 0) != 0
            ]
        dico = {
            'question': data[k]['question'],
            'answers': data[k]['answer'],
            'ctxs': ctxs,
        }
        selected_data.append(dico)

    return selected_data

if __name__ == "__main__":
    dir_path = Path(sys.argv[1])
    save_dir = Path(sys.argv[2])

    passages = util.load_passages(save_dir/'psgs_w100_100lines.tsv')
    passages = {p[0]: (p[1], p[2]) for p in passages}

    #load NQ question idx
    NQ_idx = {}
    NQ_passages = {}
    for split in ['train', 'dev', 'test']:
        with open(dir_path/('NQ.' + split + '.idx.json'), 'r') as fin:
            NQ_idx[split] = json.load(fin)
        with open(dir_path/'nq_passages' /  (split + '.json'), 'r') as fin:
            NQ_passages[split] = json.load(fin)


    originaltrain, originaldev = [], []
    with open(dir_path/'NQ-open.dev.jsonl') as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            originaldev.append(example)
    
    with open(dir_path/'NQ-open.train.jsonl') as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            originaltrain.append(example)

    NQ_train = select_examples_NQ(originaltrain, NQ_idx['train'], passages, NQ_passages['train'])
    NQ_dev = select_examples_NQ(originaltrain, NQ_idx['dev'], passages, NQ_passages['dev'])
    NQ_test = select_examples_NQ(originaldev, NQ_idx['test'], passages, NQ_passages['test'])

    NQ_save_path = save_dir / 'NQ'
    NQ_save_path.mkdir(parents=True, exist_ok=True)

    with open(NQ_save_path/'train.json', 'w') as fout:
        json.dump(NQ_train, fout, indent=4)
    with open(NQ_save_path/'dev.json', 'w') as fout:
        json.dump(NQ_dev, fout, indent=4)
    with open(NQ_save_path/'test.json', 'w') as fout:
        json.dump(NQ_test, fout, indent=4)
