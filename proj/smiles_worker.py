# Author: bbrighttaer
# Project: GPMT
# Date: 5/18/2020
# Time: 3:36 PM
# File: smiles_worker.py

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import trange

from irelease.predictor import RNNPredictor, XGBPredictor

if torch.cuda.is_available():
    dvc_id = 3
    use_cuda = True
    device = f'cuda:{dvc_id}'
    torch.cuda.set_device(dvc_id)
else:
    device = 'cpu'
    use_cuda = None


def smiles_from_json_data(file):
    val_smiles = []
    inv_smiles = []
    if os.path.exists(file):
        with open(file, 'r') as f:
            data = json.load(f)
        for k in data:
            if data[k]:
                for seed_data in data[k]:
                    for gen in seed_data:
                        val_smiles.extend(seed_data[gen][0]['valid_smiles'])
                        inv_smiles.extend(seed_data[gen][1]['invalid_smiles'])
    return val_smiles, inv_smiles


def get_drd2_evaluator():
    predictor = RNNPredictor({'model_dir': './model_dir/expert_rnn_bin',
                              'd_model': 128,
                              'rnn_num_layers': 2,
                              'dropout': 0.8,
                              'is_bidirectional': True,
                              'unit_type': 'lstm'}, device, is_binary=True)
    return predictor


def get_logp_evaluator():
    predictor = RNNPredictor({'model_dir': './model_dir/expert_rnn_reg',
                              'd_model': 128,
                              'rnn_num_layers': 2,
                              'dropout': 0.8,
                              'is_bidirectional': False,
                              'unit_type': 'lstm'}, device)
    return predictor


def get_jak2_evaluator():
    return XGBPredictor('./model_dir/expert_xgb_reg')


if __name__ == '__main__':
    batch_size = 500
    eval_files = [f for f in os.listdir('./analysis') if 'eval' in f]
    eval_func = {'drd2': get_drd2_evaluator,
                 'logp': get_logp_evaluator,
                 'jak2': get_jak2_evaluator}
    for i in trange(len(eval_files), desc='Processing SMILES...'):
        file = eval_files[i]
        valid_smiles, invalid_smiles = smiles_from_json_data('./analysis/' + file)
        eval_dict = {'SMILES': [], 'prediction': []}
        evaluator = eval_func[file.split('_')[0].lower()]()
        for j in range(0, len(valid_smiles), batch_size):
            smiles = valid_smiles[j:j + batch_size]
            preds = evaluator(smiles)[1]
            eval_dict['SMILES'].extend(smiles)
            eval_dict['prediction'].extend(np.array(preds).ravel().tolist())
        df = pd.DataFrame(eval_dict)
        df.to_csv('./analysis/' + file.replace('json', 'csv'), index=False)
