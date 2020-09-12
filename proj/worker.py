# Author: bbrighttaer
# Project: jova
# Date: 7/17/19
# Time: 6:00 PM
# File: worker.py


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score


def get_resources(root, queries):
    """Retrieves a list of resources under a root."""
    q_res = []
    for p in queries:
        res = get_resource(p, root)
        q_res.append(res)
    return q_res


def get_resource(p, root):
    """Retrieves a single resource under a root."""
    els = p.split('/')
    els.reverse()
    res = finder(root, els)
    return res


def finder(res_tree, nodes):
    """
    Uses recursion to locate a leave resource.

    :param res_tree: The (sub)resource containing the desired leave resource.
    :param nodes: A list of nodes leading to the resource.
    :return: Located resource content.
    """
    if len(nodes) == 1:
        return res_tree[nodes[0]]
    else:
        cur_node = nodes.pop()
        try:
            return finder(res_tree[cur_node], nodes)
        except TypeError:
            return finder(res_tree[int(cur_node)], nodes)


def retrieve_resource_cv(k, seeds, r_name, r_data, res_names):
    """
    Aggregates cross-validation data for analysis.

    :param k: number of folds.
    :param seeds: A list of seeds used for the simulation.
    :param r_name: The name of the root resource.
    :param r_data: The json data.
    :param res_names: A list resource(s) under each fold to be retrieved.
                      Each record is a tuple of (leave resource path, index of resource path under the given CV fold)
    :return: A dict of the aggregated resources across seeds and folds.
    """
    query_results = dict()
    for res, idx in res_names:
        query_results[res] = []
        for i, seed in enumerate(seeds):
            for j in range(k):
                path = "{}/{}/seed_{}cv/{}/fold-{}/{}/{}".format(r_name, i, seed, j, j, idx, res)
                r = get_resource(path, r_data)
                query_results[res].append(r)
    return {k: np.array(query_results[k]) for k in query_results}


def get_bc_resources():
    return [('loss', 0),
            ('metrics/accuracy_score', 1),
            ('metrics/precision_score', 1),
            ('metrics/recall_score', 1),
            ('metrics/f1_score', 1),
            ("score", 2),
            ("predictions/y_true", 3),
            ("predictions/y_pred", 3)
            ]


def get_reg_resources():
    return [('loss', 0),
            ('metrics/mean_squared_error', 1),
            ('metrics/root_mean_squared_error', 1),
            ('metrics/r2_score', 1),
            ("score", 2),
            ("predictions/y_true", 3),
            ("predictions/y_pred", 3)
            ]


if __name__ == '__main__':
    mode = 'reg'  # regression or classification, options = ['reg', 'bin']
    folder = "analysis"
    qualifier = "reg"
    files = [f for f in os.listdir(folder) if qualifier in f and ".json" in f]
    print('Number of files loaded=', len(files))
    files.sort()
    results_folder = 'analysis/'
    os.makedirs(results_folder, exist_ok=True)
    for file in files:
        print(file)
        with open(os.path.join(folder, file), "r") as f:
            data = json.load(f)

        metadata = json.loads(data['metadata'])
        root_name = metadata['sim_label']
        num_folds = metadata['num_folds']
        data_dict = retrieve_resource_cv(k=num_folds, seeds=metadata['seeds'], r_name=root_name,
                                         r_data=data,
                                         res_names=get_bc_resources() if mode == 'bin' else get_reg_resources())
        if mode == 'bin':
            acc_score = data_dict['metrics/accuracy_score'].mean()
            acc_score_std = data_dict['metrics/accuracy_score'].std()
            prec_score = data_dict['metrics/precision_score'].mean()
            prec_score_std = data_dict['metrics/precision_score'].std()
            recall = data_dict['metrics/recall_score'].mean()
            recall_std = data_dict['metrics/recall_score'].std()
            f1_score = data_dict['metrics/f1_score'].mean()
            f1_score_std = data_dict['metrics/f1_score'].std()
            auc_score = [roc_auc_score(data_dict['predictions/y_true'][k], data_dict['predictions/y_pred'][k])
                         for k in range(num_folds)]
            print(f'accuracy = {acc_score}, precision = {prec_score}, recall = {recall}, '
                  f'f1-score = {f1_score}, auc={np.mean(auc_score)}')

            with open(os.path.join(results_folder, root_name + '_results.txt'), "w") as txt_file:
                txt_file.writelines([f'accuracy={acc_score}, std={acc_score_std}\n',
                                     f'precision score={prec_score}, std={prec_score_std}\n',
                                     f'recall={recall}, std={recall_std}\n',
                                     f'f1-score={f1_score}, std={recall_std}\n',
                                     f'auc-score={np.mean(auc_score)}, std={np.std(auc_score)}'])
        elif mode == 'reg':
            mse = data_dict['metrics/mean_squared_error'].mean()
            mse_std = data_dict['metrics/mean_squared_error'].std()
            rmse = data_dict['metrics/root_mean_squared_error'].mean()
            rmse_std = data_dict['metrics/root_mean_squared_error'].std()
            r2_score = data_dict['metrics/r2_score'].mean()
            r2_score_std = data_dict['metrics/r2_score'].std()
            print(f'mse = {mse}, rmse = {rmse}, r2 = {r2_score}')
            with open(os.path.join(results_folder, root_name + '_results.txt'), "w") as txt_file:
                txt_file.writelines([f'mse={mse}, std={mse_std}\n',
                                     f'rmse={rmse}, std={rmse_std}\n',
                                     f'r2={r2_score}, std={r2_score_std}'])

        print('-' * 300)
