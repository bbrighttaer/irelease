# Author: bbrighttaer
# Project: GPMT
# Date: 5/18/2020
# Time: 3:36 PM
# File: smiles_worker.py

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import trange

from irelease.data import GeneratorData
from irelease.predictor import RNNPredictor, XGBPredictor
from irelease.utils import get_default_tokens

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


def batch_eval(out_dict, smiles, evaluator, batch_size=500):
    for j in range(0, len(smiles), batch_size):
        _smiles = smiles[j:j + batch_size]
        preds = evaluator(_smiles)[1]
        out_dict['SMILES'].extend(_smiles)
        out_dict['prediction'].extend(np.array(preds).ravel().tolist())


if __name__ == '__main__':
    eval_files = [f for f in os.listdir('./analysis/stack_rnn_tl_baseline') if 'eval.json' in f]
    eval_func = {'drd2': get_drd2_evaluator,
                 'logp': get_logp_evaluator,
                 'jak2': get_jak2_evaluator}
    unbiased_smiles_file = '../data/unbiased_smiles.smi'
    biased_smiles_file_dict = {'drd2': '../data/drd2_active_filtered.smi',
                               'logp': '../data/logp_smiles_biased.smi',
                               'jak2_min': '../data/jak2_min_smiles_biased.smi',
                               'jak2_max': '../data/jak2_max_smiles_biased.smi'}
    for i in trange(len(eval_files), desc='Processing SMILES...'):
        file = eval_files[i]
        valid_smiles, invalid_smiles = smiles_from_json_data('./analysis/stack_rnn_tl_baseline/' + file)
        eval_dict = {'SMILES': [], 'prediction': []}
        lbl = file.split('_')[0].lower()
        evaluator = eval_func[lbl]()
        batch_eval(eval_dict, valid_smiles, evaluator)
        pd.DataFrame(eval_dict).to_csv('./analysis/stack_rnn_tl_baseline/' + file.replace('json', 'csv'), index=False)

        # Unbiased SMILES
        unbiased_data_gen = GeneratorData(training_data_path=unbiased_smiles_file,
                                          delimiter='\t',
                                          cols_to_read=[0],
                                          keep_header=True,
                                          pad_symbol=' ',
                                          max_len=120,
                                          tokens=get_default_tokens(),
                                          use_cuda=use_cuda)
        unbiased_smiles = unbiased_data_gen.random_training_set_smiles(10000)
        unb_eval_dict = {'SMILES': [], 'prediction': []}
        batch_eval(unb_eval_dict, unbiased_smiles, evaluator)
        pd.DataFrame(unb_eval_dict).to_csv(f'./analysis/{lbl}_unbiased.csv', index=False)

        # Biased SMILES
        if lbl == 'jak2':
            lbl += '_' + file.split('_')[1]
        biased_data_gen = GeneratorData(training_data_path=biased_smiles_file_dict[lbl],
                                        delimiter='\t',
                                        cols_to_read=[0],
                                        keep_header=True,
                                        pad_symbol=' ',
                                        max_len=120,
                                        tokens=get_default_tokens(),
                                        use_cuda=use_cuda)
        biased_smiles = biased_data_gen.random_training_set_smiles(10000)
        b_eval_dict = {'SMILES': [], 'prediction': []}
        batch_eval(b_eval_dict, biased_smiles, evaluator)
        pd.DataFrame(b_eval_dict).to_csv(f'./analysis/{lbl}_biased.csv', index=False)
