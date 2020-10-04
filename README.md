# IReLeaSE
This repository contains the source codes of the 
[Deep Inverse Reinforcement Learning for Structural Evolution of Small Molecules](https://arxiv.org/abs/2008.11804)
paper. The work proposes a framework for training compound generators using 
Deep Inverse Reinforcement Learning.

<img style="max-width: 300px; height: auto; " src="./proj/framework.png" />

## Requirements
### Dependencies

|Library/Project | Version     |
|----------------|-------------|
| [pytorch](https://pytorch.org/get-started/locally/)          | 1.3.0       |
|[numpy](https://pypi.org/project/numpy/) | 1.18.4|
|[ptan](https://github.com/Shmuma/ptan) | 0.6|
| [tqdm](https://github.com/tqdm/tqdm)                         | 4.35.0      |
|[scikit-learn](https://scikit-learn.org/stable/install.html)|0.23.1
|[joblib](https://pypi.org/project/joblib/)|0.13.2|
| [soek](https://github.com/bbrighttaer/soek)                  | 0.0.1       |
|[pandas](https://pypi.org/project/pandas/)|1.0.3|
|[xgboost](https://pypi.org/project/xgboost/)|0.90|
| [rdkit](https://anaconda.org/rdkit/rdkit)                    | 2019.09.3.0 |
|[gym](https://github.com/openai/gym)|0.15.6|

To install the dependencies, we suggest you install 
[Anaconda](https://www.anaconda.com/products/individual) 
first and then follow the commands below:

1. Create anaconda environment
    ```bash
    $ conda create -n irelease python=3.7
    ```
2. Activate environment
   ```bash
   $ conda activate irelease
   ```
3. Install the dependencies above according to their official websites or documentations.
For instance, you can install `XGBoost` using the command
   ```bash
   $ pip install xgboost==0.90
   ```

## Datasets
The demonstrations dataset used in the experiments are as follows:

|Experiment|Dataset|
|----------|--------| 
|DRD2 Activity|[drd2_active_filtered.smi](./data/drd2_active_filtered.smi)|
|LogP       | [logp_smiles_biased.smi](./data/logp_smiles_biased.smi)|
|JAK2 Max |[jak2_max_smiles_biased.smi](./data/jak2_max_smiles_biased.smi)|
|JAK2 Min |[jak2_min_smiles_biased.smi](./data/jak2_min_smiles_biased.smi)|

The datasets used for training the models used as evaluation functions are:

|Experiment|Dataset|
|----------|--------| 
|DRD2 Activity|[drd2_bin_balanced.csv](./data/drd2_bin_balanced.csv)|
|LogP       | [logP_labels.csv](./data/logP_labels.csv)|
|JAK2 Max and Min |[jak2_data.csv](./data/jak2_data.csv)|

Pretraining dataset: [chembl.smi](./data/chembl.smi)

## Usage
Install the project as a standard python package from the project directory:
```bash
$ pip install -e .
```

Then `cd` into the `proj` directory:
```bash
$ cd proj/
```

### Pretraining
The Stack-RNN model used in our work could be pretrained with the following command:
```bash
$ cd proj
$ python pretrain_rnn.py --data ../data/chembl.smi
```
The pretrained model we used could be downloaded from [here](https://www.dropbox.com/sh/54novmbmyi1p75x/AAAk3JiGYyJ3Z_FEdC7Dcxd4a?dl=0).

### Evaluation Functions
#### DRD2 Activity
The evaluation function for the DRD2 experiment is an RNN classifier trained with
the BCE loss function. The following is the command to train the model using 
5-fold cross validation:
```bash
$ python expert_rnn_bin.py --data_file ../data/drd2_bin_balanced.csv --cv
```
After training, the evaluation can be done using:
```bash
$ python expert_rnn_bin.py --data_file ../data/drd2_bin_balanced.csv --cv --eval --eval_model_dir ./model_dir/expert_rnn_bin/
```
___
**Note:**
The value of the `--eval_model_dir` flag is a directory which contains the 5 models
saved from the CV training stage.

#### LogP
The evaluation function for the LogP optimization experiment is an RNN model trained
using the MSE loss function.
The following command invokes training:
```bash
$ python expert_rnn_reg.py --data_file ../data/logP_labels.csv --cv
```
After training, the evaluation can be done using:
```bash
$ python expert_rnn_reg.py --data_file ../data/logP_labels.csv --cv --eval --eval_model_dir ./model_dir/expert_rnn_reg/
```

#### JAK2
We trained XGBoost models for the JAK2 maximization experiment. 
The same XGBoost models were used for the JAK2 minimization experiment, as 
mentioned in the paper.

The following invokes the training process: 
```bash
$ python expert_xgb_reg.py --data_file ../data/jak2_data.csv --cv
```

And evaluation could be done using:
```bash
$ python expert_xgb_reg.py --data_file ../data/jak2_data.csv --cv --eval --eval_model_dir ./model_dir/expert_xgb_reg/
```

### Training
The following files are used for PPO training for both DIRL and IRL:

- DRD2 Activity: `ppo_rl_drd2.py`
- LogP Optimization: `ppo_rl_logp.py`
- JAK2 Maximization: `ppo_rl_jak2_minmax.py`
- JAK2 Minimization: `ppo_rl_jak2_min.py`

For DRL training, the following files are used:
 
- DRD2 Activity: `reinforce_rl_drd2.py`
- LogP Optimization: `reinforce_rl_logp.py`
- JAK2 Maximization: `reinforce_rl_jak2_minmax.py`
- JAK2 Minimization: `ppo_rl_jak2_min.py`

These files mostly share command line flags for training. For instance, to train
a generator with the DRD2 demonstrations (DIRL) the following command could be used:
```bash
$ python ppo_rl_drd2.py  --exp_name drd2 --demo ../data/drd2_active_filtered.smi --unbiased ../data/unbiased_smiles.smi --prior_data ../data/chembl.smi --pretrained_model irelease_prior.mod
```
For DRL just add the flag `--use_true_reward`,
```bash
$ python ppo_rl_drd2.py  --exp_name drd2 --demo ../data/drd2_active_filtered.smi --unbiased ../data/unbiased_smiles.smi --prior_data ../data/chembl.smi --pretrained_model irelease_prior.mod --use_true_reward
```

### Compound Sampling
Assuming the training phase produces the model `biased_generator.mod`, compound
samples, in the form of SMILES, could be generated using:
```bash
$ python pretrain_rnn.py --data ../data/chembl.smi --eval --eval_model_name biased_generator.mod --num_smiles 1000
```
The `--num_smiles` flag controls the number of SMILES (valid and invalid) that would be sampled from the
generator.

After the generation, a JSON file is produced which contains valid and invalid
SMILES. In our experiments, we process this `.json` file using 
[smiles_worker.py](./proj/smiles_worker.py) to save the valid SMILES into a CSV file. 

A sample file JSON file produced after SMILES generation is 
[here](./proj/analysis/DRD2_activity_smiles_biased_ppo_grl_eval.json).
The corresponding processed CSV file containing the valid SMILES and 
the evaluation function's 
predictions is also [here](./proj/analysis/DRD2_activity_smiles_biased_ppo_grl_eval.csv)

## Credits
We thank the authors of [ReLeaSE](https://advances.sciencemag.org/content/4/7/eaap7885?intcmp=trendmd-adv) for their 
[original implementation of Stack-RNN](https://github.com/isayev/ReLeaSE).
We thank Maxim Lapan for 
[his book on DRL](https://www.packtpub.com/product/deep-reinforcement-learning-hands-on-second-edition/9781838826994) 
and the [ptan](https://github.com/Shmuma/ptan) project.
We also acknowledge the post of [int8](https://github.com/int8) on 
[Monte-Carlo Tree Search](https://int8.io/monte-carlo-tree-search-beginners-guide/).

## Cite
```bibtex
@article{agyemang2020deep,
  title={Deep Inverse Reinforcement Learning for Structural Evolution of Small Molecules},
  author={Agyemang, Brighter and Wu, Wei-Ping and Addo, Daniel and Kpiebaareh, Michael Y and Nanor, Ebenezer and Haruna, Charles Roland},
  journal={arXiv preprint arXiv:2008.11804},
  year={2020}
}
```

