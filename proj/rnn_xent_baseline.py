# Author: bbrighttaer
# Project: GPMT
# Date: 3/23/2020
# Time: 12:03 PM
# File: pretrain.py

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import math
import os
import random
import time
from datetime import datetime as dt
import copy
import numpy as np
import torch
import torch.nn as nn
from irelease.mol_metrics import verify_sequence, get_mol_metrics
from irelease.predictor import RNNPredictor, XGBPredictor, DummyPredictor
from ptan.common.utils import TBMeanTracker
from sklearn.metrics import accuracy_score
from soek import CategoricalParam, LogRealParam, RealParam, DiscreteParam, DataNode, RandomSearch, \
    BayesianOptSearch
from soek.bopt import GPMinArgs
from soek.template import Trainer
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from irelease.data import GeneratorData
from irelease.model import Encoder, StackRNN, RNNLinearOut, StackedRNNLayerNorm, StackedRNNDropout, OneHotEncoder, \
    RNNGenerator
from irelease.utils import Flags, parse_optimizer, ExpAverage, GradStats, Count, init_hidden, init_cell, init_stack, \
    generate_smiles, time_since, get_default_tokens, canonical_smiles

currentDT = dt.now()
date_label = currentDT.strftime("%Y_%m_%d__%H_%M_%S")

seeds = [1]

if torch.cuda.is_available():
    dvc_id = 1
    use_cuda = True
    device = f'cuda:{dvc_id}'
    torch.cuda.set_device(dvc_id)
else:
    device = 'cpu'
    use_cuda = None


class RNNBaseline(Trainer):
    @staticmethod
    def initialize(hparams, demo_data_gen, unbiased_data_gen, prior_data_gen, *args, **kwargs):
        prior_data_gen.set_batch_size(hparams['batch_size'])
        # Create main model
        encoder = OneHotEncoder(vocab_size=prior_data_gen.n_characters, return_tuple=False, device=device)

        # Create RNN layers
        model = nn.Sequential(encoder,
                              RNNGenerator(input_size=prior_data_gen.n_characters,
                                           hidden_size=hparams['d_model'],
                                           unit_type=hparams['unit_type'],
                                           num_layers=hparams['num_layers'],
                                           dropout=hparams['dropout']),
                              RNNLinearOut(out_dim=prior_data_gen.n_characters,
                                           hidden_size=hparams['d_model'],
                                           bidirectional=False,
                                           bias=True))
        if use_cuda:
            model = model.cuda()
        optimizer = parse_optimizer(hparams, model)
        gen_args = {'num_layers': hparams['num_layers'],
                    'hidden_size': hparams['d_model'],
                    'num_dir': 1,
                    'has_stack': False,
                    'has_cell': hparams['unit_type'] == 'lstm',
                    'device': device,
                    'expert_model': {'pretraining': DummyPredictor(),
                                     'drd2': RNNPredictor(hparams['drd2'],
                                                          device, True),
                                     'logp': RNNPredictor(hparams['logp'], device),
                                     'jak2_max': XGBPredictor(hparams['jak2']),
                                     'jak2_min': XGBPredictor(hparams['jak2'])
                                     }.get(hparams['exp_type']),
                    'demo_data_gen': demo_data_gen,
                    'unbiased_data_gen': unbiased_data_gen,
                    'prior_data_gen': prior_data_gen,
                    'exp_type': hparams['exp_type'],
                    }
        return model, optimizer, gen_args

    @staticmethod
    def data_provider(k, flags):
        tokens = get_default_tokens()
        demo_data = GeneratorData(training_data_path=flags.demo_file,
                                  delimiter=',',
                                  cols_to_read=[0],
                                  keep_header=False,
                                  pad_symbol=' ',
                                  max_len=120,
                                  tokens=tokens,
                                  use_cuda=use_cuda)
        unbiased_data = GeneratorData(training_data_path=flags.unbiased_file,
                                      delimiter='\t',
                                      cols_to_read=[0],
                                      keep_header=True,
                                      pad_symbol=' ',
                                      max_len=120,
                                      tokens=tokens,
                                      use_cuda=use_cuda)
        prior_data = GeneratorData(training_data_path=flags.prior_data,
                                   delimiter='\t',
                                   cols_to_read=[0],
                                   keep_header=True,
                                   pad_symbol=' ',
                                   max_len=120,
                                   tokens=tokens,
                                   use_cuda=use_cuda)
        return {'demo_data': demo_data, 'unbiased_data': unbiased_data, 'prior_data': prior_data}

    @staticmethod
    def evaluate(res_dict, generated_smiles, ref_smiles):
        smiles = []
        for s in generated_smiles:
            if verify_sequence(s):
                smiles.append(s)
        mol_metrics = get_mol_metrics()
        for metric in mol_metrics:
            res_dict[metric] = np.mean(mol_metrics[metric](smiles, ref_smiles))
        score = res_dict['internal_diversity']
        return score

    @staticmethod
    def train(generator, optimizer, rnn_args, pretrained_net_path=None, pretrained_net_name=None, n_iters=5000,
              sim_data_node=None, tb_writer=None, is_hsearch=False, is_pretraining=True, grad_clipping=5):
        expert_model = rnn_args['expert_model']
        tb_writer = tb_writer()
        best_model_wts = generator.state_dict()
        best_score = -1000
        best_epoch = -1
        demo_data_gen = rnn_args['demo_data_gen']
        unbiased_data_gen = rnn_args['unbiased_data_gen']
        prior_data_gen = rnn_args['prior_data_gen']
        score_exp_avg = ExpAverage(beta=0.6)
        exp_type = rnn_args['exp_type']

        if is_pretraining:
            num_batches = math.ceil(prior_data_gen.file_len / prior_data_gen.batch_size)
        else:
            num_batches = math.ceil(demo_data_gen.file_len / demo_data_gen.batch_size)
        n_epochs = math.ceil(n_iters / num_batches)
        grad_stats = GradStats(generator, beta=0.)

        # learning rate decay schedulers
        # scheduler = sch.StepLR(optimizer, step_size=500, gamma=0.01)

        # pred_loss functions
        criterion = nn.CrossEntropyLoss(ignore_index=prior_data_gen.char2idx[prior_data_gen.pad_symbol])

        # sub-nodes of sim data resource
        loss_lst = []
        train_loss_node = DataNode(label="train_loss", data=loss_lst)

        # collect mean predictions
        unbiased_smiles_mean_pred, biased_smiles_mean_pred, gen_smiles_mean_pred = [], [], []
        unbiased_smiles_mean_pred_data_node = DataNode('baseline_mean_vals', unbiased_smiles_mean_pred)
        biased_smiles_mean_pred_data_node = DataNode('biased_mean_vals', biased_smiles_mean_pred)
        gen_smiles_mean_pred_data_node = DataNode('gen_mean_vals', gen_smiles_mean_pred)
        if sim_data_node:
            sim_data_node.data = [train_loss_node,
                                  unbiased_smiles_mean_pred_data_node,
                                  biased_smiles_mean_pred_data_node,
                                  gen_smiles_mean_pred_data_node]

        # load pretrained model
        if pretrained_net_path and pretrained_net_name:
            print('Loading pretrained model...')
            weights = RNNBaseline.load_model(pretrained_net_path, pretrained_net_name)
            generator.load_state_dict(weights)
            print('Pretrained model loaded successfully!')

        start = time.time()
        try:
            demo_score = np.mean(expert_model(demo_data_gen.random_training_set_smiles(1000))[1])
            baseline_score = np.mean(expert_model(unbiased_data_gen.random_training_set_smiles(1000))[1])
            step_idx = Count()
            gen_data = prior_data_gen if is_pretraining else demo_data_gen
            with TBMeanTracker(tb_writer, 1) as tracker:
                mode = 'Pretraining' if is_pretraining else 'Fine tuning'
                for epoch in range(n_epochs):
                    epoch_losses = []
                    epoch_mean_preds = []
                    epoch_per_valid = []
                    with grad_stats:
                        for b in trange(0, num_batches, desc=f'{mode} in progress...'):
                            inputs, labels = gen_data.random_training_set()
                            optimizer.zero_grad()

                            predictions = generator(inputs)[0]
                            predictions = predictions.permute(1, 0, -1)
                            predictions = predictions.contiguous().view(-1, predictions.shape[-1])
                            labels = labels.contiguous().view(-1)

                            # calculate loss
                            loss = criterion(predictions, labels)
                            epoch_losses.append(loss.item())

                            # backward pass
                            loss.backward()
                            if grad_clipping:
                                torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clipping)
                            optimizer.step()

                            # for sim data resource
                            n_to_generate = 200
                            with torch.set_grad_enabled(False):
                                samples = generate_smiles(generator, demo_data_gen, rnn_args, num_samples=n_to_generate)
                            samples_pred = expert_model(samples)[1]

                            # metrics
                            eval_dict = {}
                            eval_score = RNNBaseline.evaluate(eval_dict, samples,
                                                              demo_data_gen.random_training_set_smiles(1000))
                            # TBoard info
                            tracker.track('loss', loss.item(), step_idx.IncAndGet())
                            for k in eval_dict:
                                tracker.track(f'{k}', eval_dict[k], step_idx.i)
                            mean_preds = np.mean(samples_pred)
                            epoch_mean_preds.append(mean_preds)
                            if exp_type == 'drd2':
                                per_qualified = float(len([v for v in samples_pred if v >= 0.8])) / len(samples_pred)
                                score = mean_preds
                            elif exp_type == 'logp':
                                per_qualified = np.sum((samples_pred >= 1.0) & (samples_pred < 5.0)) / len(samples_pred)
                                score = mean_preds
                            elif exp_type == 'jak2_max':
                                per_qualified = np.sum((samples_pred >= demo_score)) / len(samples_pred)
                                diff = mean_preds - demo_score
                                score = np.exp(diff)
                            elif exp_type == 'jak2_min':
                                per_qualified = np.sum((samples_pred <= demo_score)) / len(samples_pred)
                                diff = demo_score - mean_preds
                                score = np.exp(diff)
                            else:  # pretraining
                                score = -loss.item()
                                per_qualified = 0.
                            per_valid = len(samples_pred) / n_to_generate
                            epoch_per_valid.append(per_valid)
                            unbiased_smiles_mean_pred.append(float(baseline_score))
                            biased_smiles_mean_pred.append(float(demo_score))
                            gen_smiles_mean_pred.append(float(mean_preds))
                            tb_writer.add_scalars('qsar_score', {'sampled': mean_preds,
                                                                 'baseline': baseline_score,
                                                                 'demo_data': demo_score}, step_idx.i)
                            tb_writer.add_scalars('SMILES stats', {'per. of valid': per_valid,
                                                                   'per. of qualified': per_qualified},
                                                  step_idx.i)
                            avg_len = np.nanmean([len(s) for s in samples])
                            tracker.track('Average SMILES length', avg_len, step_idx.i)

                            score_exp_avg.update(score)
                            if score_exp_avg.value > best_score:
                                best_model_wts = copy.deepcopy(generator.state_dict())
                                best_score = score_exp_avg.value
                                best_epoch = epoch
                        # End of mini=batch iterations.

                        smiles = generate_smiles(generator=generator, gen_data=gen_data, init_args=rnn_args,
                                                 num_samples=3)
                        print(f'{time_since(start)}: Epoch {epoch}/{n_epochs}, loss={np.mean(epoch_losses)},'
                              f'sample SMILES = {smiles}, Mean value of predictions = {np.mean(epoch_mean_preds)}, '
                              f'% of valid SMILES = {np.mean(epoch_per_valid)}')

        except ValueError as e:
            print(str(e))

        duration = time.time() - start
        print('Model training duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))
        generator.load_state_dict(best_model_wts)
        return {'model': generator, 'score': round(best_score, 3), 'epoch': best_epoch}

    @staticmethod
    @torch.no_grad()
    def evaluate_model(model, gen_data, rnn_args, sim_data_node=None, num_smiles=1000):
        start = time.time()
        model.eval()

        # Samples SMILES
        samples = []
        step = 100
        count = 0
        for _ in range(int(num_smiles / step)):
            samples.extend(generate_smiles(generator=model, gen_data=gen_data, init_args=rnn_args,
                                           num_samples=step, is_train=False, verbose=True))
            count += step
        res = num_smiles - count
        if res > 0:
            samples.extend(generate_smiles(generator=model, gen_data=gen_data, init_args=rnn_args,
                                           num_samples=res, is_train=False, verbose=True))
        smiles, valid_vec = canonical_smiles(samples)
        valid_smiles = []
        invalid_smiles = []
        for idx, sm in enumerate(smiles):
            if len(sm) > 0:
                valid_smiles.append(sm)
            else:
                invalid_smiles.append(samples[idx])
        v = len(valid_smiles)
        valid_smiles = list(set(valid_smiles))
        print(f'Percentage of valid SMILES = {float(len(valid_smiles)) / float(len(samples)):.2f}, '
              f'Num. samples = {len(samples)}, Num. valid = {len(valid_smiles)}, '
              f'Num. requested = {num_smiles}, Num. dups = {v - len(valid_smiles)}')

        # sub-nodes of sim data resource
        smiles_node = DataNode(label="valid_smiles", data=valid_smiles)
        invalid_smiles_node = DataNode(label='invalid_smiles', data=invalid_smiles)

        # add sim data nodes to parent node
        if sim_data_node:
            sim_data_node.data = [smiles_node, invalid_smiles_node]

        duration = time.time() - start
        print('\nModel evaluation duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))

    @staticmethod
    def save_model(model, path, name):
        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, name + ".mod")
        torch.save(model.state_dict(), file)

    @staticmethod
    def load_model(path, name):
        return torch.load(os.path.join(path, name), map_location=torch.device(device))


def main(flags):
    sim_label = f'RNN_XEnt_Generator_Baseline_{flags.exp_type}'
    if flags.eval:
        sim_label += '_eval'
    sim_data = DataNode(label=sim_label)
    nodes_list = []
    sim_data.data = nodes_list

    # For searching over multiple seeds
    hparam_search = None

    pretraining = flags.exp_type == 'pretraining'

    for seed in seeds:
        summary_writer_creator = lambda: SummaryWriter(log_dir="irelease_tb"
                                                               "/{}_{}_{}/".format(sim_label, seed, dt.now().strftime(
            "%Y_%m_%d__%H_%M_%S")))

        # for data collection of this round of simulation.
        data_node = DataNode(label="seed_%d" % seed)
        nodes_list.append(data_node)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        print('--------------------------------------------------------------------------------')
        print(f'{device}\n{sim_label}\tDemonstrations file: {flags.prior_data if pretraining else flags.demo_data}')
        print('--------------------------------------------------------------------------------')

        trainer = RNNBaseline()
        k = 1
        if flags["hparam_search"]:
            print("Hyperparameter search enabled: {}".format(flags["hparam_search_alg"]))

            # arguments to callables
            extra_init_args = {}
            extra_data_args = {"flags": flags}
            extra_train_args = {"is_hsearch": True,
                                "n_iters": 50000,
                                "tb_writer": summary_writer_creator}

            hparams_conf = get_hparam_config(flags)
            if hparam_search is None:
                search_alg = {"random_search": RandomSearch,
                              "bayopt_search": BayesianOptSearch}.get(flags["hparam_search_alg"],
                                                                      BayesianOptSearch)
                search_args = GPMinArgs(n_calls=20, random_state=seed)
                hparam_search = search_alg(hparam_config=hparams_conf,
                                           num_folds=1,
                                           initializer=trainer.initialize,
                                           data_provider=trainer.data_provider,
                                           train_fn=trainer.train,
                                           save_model_fn=trainer.save_model,
                                           alg_args=search_args,
                                           init_args=extra_init_args,
                                           data_args=extra_data_args,
                                           train_args=extra_train_args,
                                           data_node=data_node,
                                           split_label='',
                                           sim_label=sim_label,
                                           dataset_label='ChEMBL_SMILES',
                                           results_file="{}_{}_gpmt_{}.csv".format(
                                               flags["hparam_search_alg"], sim_label, date_label))

            stats = hparam_search.fit(model_dir="models", model_name='irelease')
            print(stats)
            print("Best params = {}".format(stats.best()))
        else:
            hyper_params = default_hparams(flags)
            data_gens = trainer.data_provider(k, flags)
            model, optimizer, rnn_args = trainer.initialize(hyper_params, data_gens['demo_data'],
                                                            data_gens['unbiased_data'],
                                                            data_gens['prior_data'])
            if flags.eval:
                load_model = trainer.load_model(flags.model_dir, flags.eval_model_name)
                model.load_state_dict(load_model)
                # trainer.evaluate_model(model, gen_data, rnn_args, data_node, num_smiles=10000)
            else:
                results = trainer.train(generator=model,
                                        optimizer=optimizer,
                                        rnn_args=rnn_args,
                                        n_iters=1500000,
                                        sim_data_node=data_node,
                                        tb_writer=summary_writer_creator,
                                        is_pretraining=pretraining,
                                        pretrained_net_path=flags.model_dir,
                                        pretrained_net_name=flags.pretrained_model)
                trainer.save_model(results['model'], flags.model_dir,
                                   name=f'rnn_xent_gen_baseline_{flags.exp_type}_{hyper_params["unit_type"]}_'
                                        f'{date_label}_{results["score"]}_{results["epoch"]}')

    # save simulation data resource tree to file.
    sim_data.to_json(path="./analysis/")


def default_hparams(args):
    return {
        'unit_type': 'lstm',
        'num_layers': 3,
        'dropout': 0.2,
        'd_model': 1024,
        'batch_size': 128,
        'optimizer': 'adam',
        'exp_type': args.exp_type,
        'drd2': {'model_dir': './model_dir/expert_rnn_bin',
                 'd_model': 128,
                 'rnn_num_layers': 2,
                 'dropout': 0.8,
                 'is_bidirectional': True,
                 'unit_type': 'lstm'},
        'logp': {'model_dir': './model_dir/expert_rnn_reg',
                 'd_model': 128,
                 'rnn_num_layers': 2,
                 'dropout': 0.8,
                 'is_bidirectional': False,
                 'unit_type': 'lstm'},
        'jak2': './model_dir/expert_xgb_reg'
    }


def get_hparam_config(args):
    config = {
    }
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RNN SMILES Generator')
    parser.add_argument('--exp_type',
                        required=True,
                        choices=['pretraining', 'drd2', 'logp', 'jak2_max', 'jak2_min'],
                        help='The type of experiment')
    parser.add_argument('--demo', dest='demo_file', type=str,
                        help='File containing SMILES strings which are demonstrations of the required objective')
    parser.add_argument('--unbiased', dest='unbiased_file', type=str,
                        help='File containing SMILES generated with the pretrained (prior) model.')
    parser.add_argument('--prior_data', dest='prior_data', type=str,
                        help='File containing SMILES used to train the prior model.')
    parser.add_argument('--model_dir',
                        type=str,
                        default='./model_dir',
                        help='Directory containing models')
    parser.add_argument('--pretrained_model',
                        type=str,
                        default=None,
                        help='The name of the pretrained model')
    parser.add_argument("--hparam_search",
                        action="store_true",
                        help="If true, hyperparameter searching would be performed.")
    parser.add_argument("--hparam_search_alg",
                        type=str,
                        default="bayopt_search",
                        help="Hyperparameter search algorithm to use. One of [bayopt_search, random_search]")
    parser.add_argument("--eval",
                        action="store_true",
                        help="If true, a saved model is loaded and evaluated")
    parser.add_argument("--eval_model_name",
                        default=None,
                        type=str,
                        help="The filename of the model to be loaded from the directory specified in --model_dir")

    args = parser.parse_args()
    flags = Flags()
    args_dict = args.__dict__
    for arg in args_dict:
        setattr(flags, arg, args_dict[arg])
    main(flags)
