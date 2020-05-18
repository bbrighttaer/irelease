# Author: bbrighttaer
# Project: GPMT
# Date: 5/18/2020
# Time: 3:36 PM
# File: smiles_worker.py

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import json

if __name__ == '__main__':
    file = 'analysis/GPMT-pretraining-Stack-RNN_eval.json'
    all_smiles = []
    if os.path.exists(file):
        with open(file, 'r') as f:
            data = json.load(f)
        for k in data:
            for seed_data in data[k]:
                for gen in seed_data:
                    all_smiles.extend(seed_data[gen][0]['smiles'])
    print(f'Number of smiles aggregated ={len(all_smiles)}')
    with open(file + '.smi', 'w') as f:
        for sm in all_smiles:
            f.write(sm + '\n')
