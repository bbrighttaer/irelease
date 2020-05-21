# From: https://github.com/isayev/ReLeaSE/blob/master/release/predictor.py
# Included in this project to serve as a proof of concept expert model.

from __future__ import division
from __future__ import print_function

import os
import numpy as np
import rdkit.Chem as Chem
import torch
from tqdm import tqdm

from gpmt.model import RNNPredictorModel
from gpmt.utils import get_default_tokens


class RNNPredictor(object):
    def __init__(self, hparams, device):
        expert_model_dir = hparams['model_dir']
        assert (os.path.isdir(expert_model_dir)), 'Expert model(s) should be in a dedicated folder'
        self.models = []
        self.tokens = get_default_tokens()
        self.device = device
        model_paths = os.listdir(expert_model_dir)
        for model_file in model_paths:
            model = RNNPredictorModel(d_model=hparams['d_model'],
                                      tokens=self.tokens,
                                      num_layers=hparams['rnn_num_layers'],
                                      dropout=hparams['dropout'],
                                      bidirectional=hparams['is_bidirectional'],
                                      unit_type=hparams['unit_type'],
                                      device=device).to(device)
            model.load_state_dict(torch.load(os.path.join(expert_model_dir, model_file),
                                             map_location=torch.device(device)))
            self.models.append(model)

    @torch.no_grad()
    def predict(self, smiles, use_tqdm=False):
        """
        Original implementation of this function:
        https://github.com/isayev/ReLeaSE/blob/batch_training/release/rnn_predictor.py

        :param smiles: list
        :param use_tqdm: bool
        :return:
        """
        canonical_smiles = []
        invalid_smiles = []
        if use_tqdm:
            pbar = tqdm(range(len(smiles)))
        else:
            pbar = range(len(smiles))
        for i in pbar:
            sm = smiles[i]
            if use_tqdm:
                pbar.set_description("Calculating predictions...")
            try:
                sm = Chem.MolToSmiles(Chem.MolFromSmiles(sm, sanitize=False))
                if len(sm) == 0:
                    invalid_smiles.append(sm)
                else:
                    canonical_smiles.append(sm)
            except:
                invalid_smiles.append(sm)
        if len(canonical_smiles) == 0:
            return canonical_smiles, [], invalid_smiles
        prediction = []
        for i in range(len(self.models)):
            prediction.append(self.models[i](canonical_smiles).detach().cpu().numpy())
        prediction = np.array(prediction)
        prediction = np.min(prediction, axis=0)
        return canonical_smiles, prediction, invalid_smiles

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


def get_reward_logp(smiles, predictor, invalid_reward=0.0):
    mol, prop, nan_smiles = predictor.predict([smiles])
    if len(nan_smiles) == 1:
        return invalid_reward
    if (prop[0] >= 1.0) and (prop[0] <= 4.0):
        return 11.0
    else:
        return 1.0
