# From: https://github.com/isayev/ReLeaSE/blob/master/release/predictor.py
# Included in this project to serve as a proof of concept expert model.

from __future__ import division
from __future__ import print_function

import os

import joblib
import numpy as np
import rdkit.Chem as Chem
import torch
from tqdm import tqdm
from xgboost import DMatrix

from irelease.drd2 import DRD2Model
from irelease.model import RNNPredictorModel
from irelease.utils import get_default_tokens, get_fp


class Predictor:
    def predict(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


class RNNPredictor(Predictor):
    def __init__(self, hparams, device, is_binary=False):
        expert_model_dir = hparams['model_dir']
        assert (os.path.isdir(expert_model_dir)), 'Expert model(s) should be in a dedicated folder'
        self.models = []
        self.tokens = get_default_tokens()
        self.device = device
        model_paths = os.listdir(expert_model_dir)
        self.transformer = None
        self.is_binary = is_binary
        for model_file in model_paths:
            if 'transformer' in model_file:
                with open(os.path.join(expert_model_dir, model_file), 'rb') as f:
                    self.transformer = joblib.load(f)
                    continue
            model = RNNPredictorModel(d_model=hparams['d_model'],
                                      tokens=self.tokens,
                                      num_layers=hparams['rnn_num_layers'],
                                      dropout=hparams['dropout'],
                                      bidirectional=hparams['is_bidirectional'],
                                      unit_type=hparams['unit_type'],
                                      device=device).to(device)
            if is_binary:
                model = torch.nn.Sequential(model, torch.nn.Sigmoid()).to(device)
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
            y_pred = self.models[i](canonical_smiles).detach().cpu().numpy()
            if not self.is_binary and self.transformer is not None:
                y_pred = self.transformer.inverse_transform(y_pred)
            prediction.append(y_pred)
        prediction = np.array(prediction)
        pool = np.mean if self.is_binary else np.min
        prediction = pool(prediction, axis=0)
        return canonical_smiles, prediction, invalid_smiles


class SVRPredictor(Predictor):
    def __init__(self, expert_model_dir):
        assert (os.path.isdir(expert_model_dir)), 'Expert model(s) should be in a dedicated folder'
        self.models = []
        model_paths = os.listdir(expert_model_dir)
        self.transformer = None
        for model_file in model_paths:
            if 'transformer' in model_file:
                with open(os.path.join(expert_model_dir, model_file), 'rb') as f:
                    self.transformer = joblib.load(f)
                    continue
            with open(os.path.join(expert_model_dir, model_file), 'rb') as f:
                model = joblib.load(f)
                self.models.append(model)

    def predict(self, smiles, get_features=get_fp, use_tqdm=False):
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
        x, _, _ = get_features(canonical_smiles, sanitize=False)
        for i in range(len(self.models)):
            y_pred = self.models[i].predict(x)
            if self.transformer is not None:
                y_pred = self.transformer.inverse_transform(y_pred)
            prediction.append(y_pred)
        prediction = np.array(prediction)
        prediction = np.min(prediction, axis=0)
        return canonical_smiles, prediction, invalid_smiles


class XGBPredictor(Predictor):
    def __init__(self, expert_model_dir):
        assert (os.path.isdir(expert_model_dir)), 'Expert model(s) should be in a dedicated folder'
        self.models = []
        model_paths = os.listdir(expert_model_dir)
        self.transformer = None
        for model_file in model_paths:
            if 'transformer' in model_file:
                with open(os.path.join(expert_model_dir, model_file), 'rb') as f:
                    self.transformer = joblib.load(f)
                    continue
            with open(os.path.join(expert_model_dir, model_file), 'rb') as f:
                model = joblib.load(f)
                self.models.append(model)

    def predict(self, smiles, get_features=get_fp, use_tqdm=False):
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
        x, _, _ = get_features(canonical_smiles, sanitize=False)
        x = DMatrix(x)
        for i in range(len(self.models)):
            y_pred = self.models[i].predict(x)
            if self.transformer is not None:
                y_pred = self.transformer.inverse_transform(y_pred)
            prediction.append(y_pred)
        prediction = np.array(prediction)
        prediction = np.mean(prediction, axis=0)
        return canonical_smiles, prediction, invalid_smiles


class SVCPredictor(Predictor):
    def __init__(self, svc_path):
        self.svc = DRD2Model(svc_path)

    def predict(self, smiles, use_tqdm=False):
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
        for smiles in canonical_smiles:
            prediction.append(self.svc(smiles))
        prediction = np.array(prediction)
        return canonical_smiles, prediction, invalid_smiles


def get_logp_reward(smiles, predictor, invalid_reward=0.0):
    mol, pred, nan_smiles = predictor.predict([smiles])
    if len(nan_smiles) == 1:
        return invalid_reward
    if (pred[0] >= 1.0) and (pred[0] <= 4.0):
        return 11.0
    else:
        return 1.0


def get_drd2_activity_reward(smiles, predictor, invalid_reward=0.0):
    mol, pred, nan_smiles = predictor.predict([smiles])
    if len(nan_smiles) == 1:
        return invalid_reward
    return pred[0]


def get_jak2_max_reward(smiles, predictor, invalid_reward=0.0):
    mol, pred, nan_smiles = predictor.predict([smiles], get_features=get_fp)
    if len(nan_smiles) == 1:
        return invalid_reward
    return np.exp(pred[0] / 3)


def get_jak2_min_reward(smiles, predictor, invalid_reward=0.0):
    mol, prop, nan_smiles = predictor.predict([smiles], get_features=get_fp)
    if len(nan_smiles) == 1:
        return invalid_reward
    return np.exp(-prop[0] / 3 + 3)
