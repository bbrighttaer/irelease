# Original code from: https://github.com/isayev/ReLeaSE

import csv
import math
import time
import warnings

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Dataset
from tqdm import trange
from sklearn.metrics import mean_squared_error

from irelease.mol_graphs import ConvMol

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def get_fp(smiles, sanitize=True):
    fp = []
    processed_indices = []
    invalid_indices = []
    for i in range(len(smiles)):
        mol = smiles[i]
        tmp = np.array(mol2image(mol, n=2048, sanitize=sanitize))
        if np.isnan(tmp[0]):
            invalid_indices.append(i)
        else:
            fp.append(tmp)
            processed_indices.append(i)
    return np.array(fp), processed_indices, invalid_indices


def get_desc(smiles, calc):
    desc = []
    processed_indices = []
    invalid_indices = []
    for i in range(len(smiles)):
        sm = smiles[i]
        try:
            mol = Chem.MolFromSmiles(sm)
            tmp = np.array(calc(mol))
            desc.append(tmp)
            processed_indices.append(i)
        except:
            invalid_indices.append(i)

    desc_array = np.array(desc)
    return desc_array, processed_indices, invalid_indices


def normalize_desc(desc_array, desc_mean=None):
    desc_array = np.array(desc_array).reshape(len(desc_array), -1)
    ind = np.zeros(desc_array.shape)
    for i in range(desc_array.shape[0]):
        for j in range(desc_array.shape[1]):
            try:
                if np.isfinite(desc_array[i, j]):
                    ind[i, j] = 1
            except:
                pass
    for i in range(desc_array.shape[0]):
        for j in range(desc_array.shape[1]):
            if ind[i, j] == 0:
                desc_array[i, j] = 0
    if desc_mean is None:
        desc_mean = np.mean(desc_array, axis=0)
    for i in range(desc_array.shape[0]):
        for j in range(desc_array.shape[1]):
            if ind[i, j] == 0:
                desc_array[i, j] = desc_mean[j]
    return desc_array, desc_mean


def mol2image(x, n=2048, sanitize=False):
    try:
        m = Chem.MolFromSmiles(x, sanitize=sanitize)
        fp = Chem.RDKFingerprint(m, maxPath=4, fpSize=n)
        res = np.zeros(len(fp))
        DataStructs.ConvertToNumpyArray(fp, res)
        return res
    except:
        return [np.nan]


def sanitize_smiles(smiles, canonical=True, throw_warning=False):
    """
    Takes list of SMILES strings and returns list of their sanitized versions.
    For definition of sanitized SMILES check
    http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol

    Parameters
    ----------
    smiles: list
        list of SMILES strings

    canonical: bool (default True)
        parameter specifying whether SMILES will be converted to canonical
        format

    throw_warning: bool (default False)
        parameter specifying whether warnings will be thrown if a SMILES is
        invalid

    Returns
    -------
    new_smiles: list
        list of SMILES and NaNs if SMILES string is invalid or unsanitized.
        If canonical is True, returns list of canonical SMILES.

    When canonical is True this function is analogous to:
        canonical_smiles(smiles, sanitize=True).
    """
    new_smiles = []
    for sm in smiles:
        try:
            if canonical:
                new_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(sm, sanitize=True)))
            else:
                new_smiles.append(sm)
        except:
            if throw_warning:
                warnings.warn('Unsanitized SMILES string: ' + sm, UserWarning)
            new_smiles.append('')
    return new_smiles


def canonical_smiles(smiles, sanitize=True, throw_warning=False):
    """
    Takes list of SMILES strings and returns list of their canonical SMILES.

    Parameters
    ----------
    smiles: list
        list of SMILES strings to convert into canonical format

    sanitize: bool (default True)
        parameter specifying whether to sanitize SMILES or not.
            For definition of sanitized SMILES check
            http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol

    throw_warning: bool (default False)
        parameter specifying whether warnings will be thrown if a SMILES is
        invalid

    Returns
    -------
    new_smiles: list
        list of canonical SMILES and NaNs if SMILES string is invalid or
        unsanitized (when sanitize is True)

    When sanitize is True the function is analogous to:
        sanitize_smiles(smiles, canonical=True).
    """
    new_smiles = []
    valid_vec = []
    for sm in smiles:
        try:
            mol = Chem.MolFromSmiles(sm, sanitize=sanitize)
            new_smiles.append(Chem.MolToSmiles(mol))
            valid_vec.append(1)
        except:
            if throw_warning:
                warnings.warn(sm + ' can not be canonized: invalid '
                                   'SMILES string!', UserWarning)
            new_smiles.append('')
            valid_vec.append(0)
    return new_smiles, valid_vec


def save_smi_to_file(filename, smiles, unique=True):
    """
    Takes path to file and list of SMILES strings and writes SMILES to the specified file.

        Args:
            filename (str): path to the file
            smiles (list): list of SMILES strings
            unique (bool): parameter specifying whether to write only unique copies or not.

        Output:
            success (bool): defines whether operation was successfully completed or not.
       """
    if unique:
        smiles = list(set(smiles))
    else:
        smiles = list(smiles)
    f = open(filename, 'w')
    for mol in smiles:
        f.writelines([mol, '\n'])
    f.close()
    return f.closed


def read_smi_file(filename, unique=True, add_start_end_tokens=False):
    """
    Reads SMILES from file. File must contain one SMILES string per line
    with \n token in the end of the line.

    Args:
        filename (str): path to the file
        unique (bool): return only unique SMILES

    Returns:
        smiles (list): list of SMILES strings from specified file.
        success (bool): defines whether operation was successfully completed or not.

    If 'unique=True' this list contains only unique copies.
    """
    f = open(filename, 'r')
    molecules = []
    for line in f:
        if add_start_end_tokens:
            molecules.append('<' + line[:-1] + '>')
        else:
            molecules.append(line[:-1])
    if unique:
        molecules = list(set(molecules))
    else:
        molecules = list(molecules)
    f.close()
    return molecules, f.closed


# def tokenize(smiles, pad_symbol, tokens=None, tokens_reload=False):
#     """
#     Returns list of unique tokens, token-2-index dictionary and number of
#     unique tokens from the list of SMILES
#
#     Parameters
#     ----------
#         smiles: list
#             list of SMILES strings to tokenize.
#
#         tokens: list, str (default None)
#             list of unique tokens
#         tokens_reload: bool
#             Whether the resulting tokens dict should be pickled and loaded for subsequent runs.
#
#     Returns
#     -------
#         tokens: list
#             list of unique tokens/SMILES alphabet.
#
#         token2idx: dict
#             dictionary mapping token to its index.
#
#         num_tokens: int
#             number of unique tokens.
#     """
#     check_reload = False
#     if tokens is None:
#         if tokens_reload and os.path.exists('token2idx.pkl'):
#             with open('token2idx.pkl', 'rb') as f:
#                 token2idx = pickle.load(f)
#                 tokens = list(token2idx.keys())
#                 num_tokens = len(tokens)
#                 return tokens, token2idx, num_tokens
#         tokens = list(set(''.join(smiles)))
#         tokens = list(np.sort(tokens))
#         tokens = ''.join(tokens)
#         check_reload = True
#     token2idx = dict((token, i) for i, token in enumerate(tokens))
#     token2idx[pad_symbol] = len(token2idx)
#     tokens += pad_symbol
#     num_tokens = len(tokens)
#     if check_reload and tokens_reload:
#         with open('token2idx.pkl', 'wb') as f:
#             pickle.dump(dict(token2idx), f)
#     return tokens, token2idx, num_tokens

def tokenize(smiles, tokens=None):
    """
    Returns list of unique tokens, token-2-index dictionary and number of
    unique tokens from the list of SMILES

    Parameters
    ----------
        smiles: list
            list of SMILES strings to tokenize.

        tokens: list, str (default None)
            list of unique tokens

    Returns
    -------
        tokens: list
            list of unique tokens/SMILES alphabet.

        token2idx: dict
            dictionary mapping token to its index.

        num_tokens: int
            number of unique tokens.
    """
    if tokens is None:
        tokens = list(set(''.join(smiles)))
        tokens = list(np.sort(tokens))
        tokens = ''.join(tokens)
    token2idx = dict((token, i) for i, token in enumerate(tokens))
    num_tokens = len(tokens)
    return tokens, token2idx, num_tokens


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def cross_validation_split(x, y, n_folds=5, split='random', folds=None):
    assert (len(x) == len(y))
    x = np.array(x)
    y = np.array(y)
    if split not in ['random', 'stratified', 'fixed']:
        raise ValueError('Invalid value for argument \'split\': '
                         'must be either \'random\', \'stratified\' '
                         'or \'fixed\'')
    if split == 'random':
        cv_split = KFold(n_splits=n_folds, shuffle=True)
        folds = list(cv_split.split(x, y))
    elif split == 'stratified':
        cv_split = StratifiedKFold(n_splits=n_folds, shuffle=True)
        folds = list(cv_split.split(x, y))
    elif split == 'fixed' and folds is None:
        raise TypeError(
            'Invalid type for argument \'folds\': found None, but must be list')
    cross_val_data = []
    cross_val_labels = []
    if len(folds) == n_folds:
        for fold in folds:
            cross_val_data.append(x[fold[1]])
            cross_val_labels.append(y[fold[1]])
    elif len(folds) == len(x) and np.max(folds) == n_folds:
        for f in range(n_folds):
            left = np.where(folds == f)[0].min()
            right = np.where(folds == f)[0].max()
            cross_val_data.append(x[left:right + 1])
            cross_val_labels.append(y[left:right + 1])

    return cross_val_data, cross_val_labels


def read_object_property_file(path, delimiter=',', cols_to_read=[0, 1],
                              keep_header=False, **kwargs):
    f = open(path, 'r')
    reader = csv.reader(f, delimiter=delimiter)
    data_full = np.array(list(reader))
    if keep_header:
        start_position = 0
    else:
        start_position = 1
    assert len(data_full) > start_position
    data = [[] for _ in range(len(cols_to_read))]
    for i in range(len(cols_to_read)):
        col = cols_to_read[i]
        data[i] = data_full[start_position:, col]
    f.close()
    if len(cols_to_read) == 1:
        data = data[0]
    return data


def init_hidden_2d(batch_size, seq_length, d_hidden, dvc='cpu'):
    """
    Initialization of the hidden state of RNN.

    Returns
    -------
    hidden: tensor
        tensor filled with zeros of size (batch_size, seq. length, d_hidden)
    """
    return torch.zeros(batch_size, seq_length, d_hidden).to(dvc)


def init_stack_2d(batch_size, seq_length, stack_depth, stack_width, dvc='cpu'):
    """
    Initialization of the stack state.

    Returns
    -------
    stack: tensor
        tensor filled with zeros
    """
    return torch.zeros(batch_size, seq_length, stack_depth, stack_width).to(dvc)


def init_stack(batch_size, stack_width, stack_depth, dvc='cpu'):
    return torch.zeros(batch_size, stack_depth, stack_width).to(dvc)


def init_hidden(num_layers, batch_size, hidden_size, num_dir=1, dvc='cpu'):
    return torch.zeros(num_layers * num_dir, batch_size, hidden_size).to(dvc)


def init_cell(num_layers, batch_size, hidden_size, num_dir=1, dvc='cpu'):
    return init_hidden(num_layers, batch_size, hidden_size, num_dir, dvc)


class Flags(object):
    # enables using either object referencing or dict indexing to retrieve user passed arguments of flag objects.
    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        setattr(self, key, value)


def get_default_tokens():
    """Default SMILES tokens"""
    tokens = [' ', '<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
              '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
              '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']
    return tokens


def char_to_tensor(string, device, tokens=None):
    if tokens is None:
        tokens = get_default_tokens()
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = tokens.index(string[c])
    tensor = tensor.to(device)
    return tensor.clone()


def parse_optimizer(hparams, model):
    """
    Creates an optimizer for the given model using the argumentes specified in
    hparams.

    Arguments:
    -----------
    :param hparams: Hyperparameters dictionary
    :param model: An nn.Module object
    :return: a torch.optim object
    """
    # optimizer configuration
    optimizer = {
        "adadelta": torch.optim.Adadelta,
        "adagrad": torch.optim.Adagrad,
        "adam": torch.optim.Adam,
        "rmsprop": torch.optim.RMSprop,
        "adamax": torch.optim.Adamax,
        "rprop": torch.optim.Rprop,
        "sgd": torch.optim.SGD,
    }.get(hparams["optimizer"].lower(), None)
    assert optimizer is not None, "{} optimizer could not be found"

    # filter optimizer arguments
    optim_kwargs = dict()
    optim_key = hparams["optimizer"]
    for k, v in hparams.items():
        if "optimizer__" in k:
            attribute_tup = k.split("__")
            if optim_key == attribute_tup[1] or attribute_tup[1] == "global":
                optim_kwargs[attribute_tup[2]] = v
    optimizer = optimizer(model.parameters(), **optim_kwargs)
    return optimizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GradStats(object):
    def __init__(self, net, tb_writer=None, beta=.9, bias_cor=False):
        super(GradStats, self).__init__()
        self.net = net
        self.writer = tb_writer
        self._l2 = ExpAverage(beta, bias_cor)
        self._max = ExpAverage(beta, bias_cor)
        self._var = ExpAverage(beta, bias_cor)
        self._window = 1 // (1. - beta)

    @property
    def l2(self):
        return self._l2.value

    @property
    def max(self):
        return self._max.value

    @property
    def var(self):
        return self._var.value

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self._l2.reset()
        self._max.reset()
        self._var.reset()
        self.t = 0

    def stats(self, step_idx=None):
        grads = np.concatenate([p.grad.data.cpu().numpy().flatten() for p in self.net.parameters()
                                if p.grad is not None])
        l2 = np.sqrt(np.mean(np.square(grads)))
        self._l2.update(l2)
        mx = np.max(np.abs(grads))
        self._max.update(mx)
        vr = np.var(grads)
        self._var.update(vr)
        if self.writer:
            assert step_idx is not None, "step_idx cannot be none"
            self.writer.add_scalar("grad_l2", l2, step_idx)
            self.writer.add_scalar("grad_max", mx, step_idx)
            self.writer.add_scalar("grad_var", vr, step_idx)
        return "Grads stats (w={}): L2={}, max={}, var={}".format(int(self._window), self.l2, self.max, self.var)


def generate_smiles(generator, gen_data, init_args, prime_str='<', end_token='>', max_len=100, num_samples=5,
                    gen_type='rnn', is_train=True, return_probs=False, return_logits=False, verbose=False):
    """
    Generates SMILES strings using the model/generator given.

    Arguments:
    ------------
    :param generator: nn.Module
        The model for generating SMILES.
    :param gen_data:
        Object of ::class::data.GeneratorData.
    :param init_args: dict
        Arguments for facilitating the creation of initial states.
    :param prime_str: str
        Character for indicating the beginning of a SMILES string.
    :param end_token: str
        Character for indicating the end of a SMILES string.
    :param max_len: int
        The maximum length of a generated string.
    :param num_samples: int
        The number of samples to be generated. Notice that the number of SMILES returned may be lesser than this
        number since the sampled strings are filtered for validity.
    :param gen_type: str
        rnn or trans
    :param is_train: bool
        Whether the call is from a training procedure. If it is then the generator would be set back to train after
        generation.
    :return: list
        Generated string(s).
    """
    generator.eval()
    if gen_type == 'rnn':
        hidden_states = []
        num_layers = init_args['num_layers'] if 'num_layers' in init_args else 1
        for _ in range(init_args['num_layers']):
            hidden = init_hidden(num_layers=num_layers, batch_size=num_samples,
                                 hidden_size=init_args['hidden_size'],
                                 num_dir=init_args['num_dir'], dvc=init_args['device'])
            if init_args['has_cell']:
                cell = init_cell(num_layers=num_layers, batch_size=num_samples,
                                 hidden_size=init_args['hidden_size'],
                                 num_dir=init_args['num_dir'], dvc=init_args['device'])
            else:
                cell = None
            if init_args['has_stack']:
                stack = init_stack(num_samples, init_args['stack_width'], init_args['stack_depth'],
                                   dvc=init_args['device'])
            else:
                stack = None
            hidden_states.append((hidden, cell, stack))

    prime_input, _ = seq2tensor([prime_str] * num_samples, tokens=gen_data.all_characters, flip=False)
    prime_input = torch.from_numpy(prime_input).long().to(init_args['device'])
    new_samples = [[prime_str] * num_samples]

    # Use priming string to initialize hidden state
    if gen_type == 'rnn':
        for p in range(len(prime_str[0]) - 1):
            x_ = prime_input[:, p]
            if x_.ndim == 1:
                x_ = x_.view(-1, 1)
            outputs = generator([x_] + hidden_states)
            hidden_states = outputs[1:]
    inp = prime_input[:, -1]
    if inp.ndim == 1:
        inp = inp.view(-1, 1)

    # try:
    # Start sampling
    if verbose:
        loop = trange(max_len - 1, desc='Generating SMILES...')
    else:
        loop = range(max_len - 1)
    probs_out = []
    logits = []
    hidden_neurons = []
    for i in loop:
        if gen_type == 'rnn':
            outputs = generator([inp] + hidden_states)
            output, hidden_states = outputs[0], outputs[1:]
            output = output.detach().cpu()
            logits.append(output)
            hidden_neurons.append(hidden_states[-1][0])
        elif gen_type == 'trans':
            stack = init_stack_2d(num_samples, inp.shape[-1], init_args['stack_depth'],
                                  init_args['stack_width'],
                                  dvc=init_args['device'])
            output = generator([inp, stack])[-1, :, :]
            output = output.detach().cpu()

        # Sample the next character from the generator
        probs = torch.softmax(output.view(-1, output.shape[-1]), dim=-1).detach()
        probs_out.append(probs)
        top_i = torch.multinomial(probs, 1).cpu().numpy()

        # Add predicated character to string and use as next input.
        predicted_char = (np.array(gen_data.all_characters)[top_i].reshape(-1))
        predicted_char = predicted_char.tolist()
        new_samples.append(predicted_char)

        # Prepare next input token for the generator
        if gen_type == 'trans':
            predicted_char = np.array(new_samples).transpose()
        inp, _ = seq2tensor(predicted_char, tokens=gen_data.all_characters)
        inp = torch.from_numpy(inp).long().to(init_args['device'])
    # except:
    #     print('SMILES generation error')

    if is_train:
        generator.train()

    # Remove characters after end tokens
    string_samples = []
    new_samples = np.array(new_samples)
    if verbose:
        loop = trange(num_samples, desc='Processing SMILES...')
    else:
        loop = range(num_samples)
    for i in loop:
        sample = list(new_samples[:, i])
        if end_token in sample:
            end_token_idx = sample.index(end_token)
            string_samples.append(''.join(sample[1:end_token_idx]))
        else:
            string_samples.append(''.join(sample[1:]))
    probs_out = torch.cat(probs_out).cpu().numpy()
    if return_probs:
        return string_samples, probs_out
    if return_logits:
        logits = torch.cat(logits).numpy()
        hidden_neurons = torch.cat(hidden_neurons).numpy()
        return string_samples, logits, hidden_neurons
    return string_samples


def seq2tensor(seqs, tokens, flip=False):
    tensor = np.zeros((len(seqs), len(seqs[0])))
    for i in range(len(seqs)):
        for j in range(len(seqs[i])):
            if seqs[i][j] in tokens:
                tensor[i, j] = tokens.index(seqs[i][j])
            else:
                tokens = tokens + [seqs[i][j]]
                tensor[i, j] = tokens.index(seqs[i][j])
    if flip:
        tensor = np.flip(tensor, axis=1).copy()
    return tensor, tokens


def pad_sequences(seqs, max_length=None, pad_symbol=' '):
    if max_length is None:
        max_length = -1
        for seq in seqs:
            max_length = max(max_length, len(seq))
    lengths = []
    for i in range(len(seqs)):
        cur_len = len(seqs[i])
        lengths.append(cur_len)
        seqs[i] = seqs[i] + pad_symbol * (max_length - cur_len)
    return seqs, lengths


def process_gconv_view_data(mols, device):
    """Converts ConvMol sample(s) to pytorch tensor data"""
    mol_data = []
    multiConvMol = ConvMol.agglomerate_mols(mols)
    mol_data.append(torch.from_numpy(multiConvMol.get_atom_features()).to(device))
    mol_data.append(torch.from_numpy(multiConvMol.deg_slice).to(device))
    mol_data.append(torch.tensor(multiConvMol.membership).to(device))
    mol_data.append([mol.get_num_atoms() for mol in mols])
    for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
        mol_data.append(torch.from_numpy(multiConvMol.get_deg_adjacency_lists()[i]).to(device))
    return mol_data


def get_activation_func(activation):
    from irelease.model import NonsatActivation
    return {'relu': torch.nn.ReLU(),
            'leaky_relu': torch.nn.LeakyReLU(.2),
            'sigmoid': torch.nn.Sigmoid(),
            'tanh': torch.nn.Tanh(),
            'softmax': torch.nn.Softmax(),
            'elu': torch.nn.ELU(),
            'nonsat': NonsatActivation()}.get(activation.lower(), torch.nn.ReLU())


def parse_hparams(file, index):
    hdata = pd.read_csv(file, header=0)
    hdict = hdata.to_dict('index')
    hdict = hdict[list(hdict.keys())[index]]
    return _rec_parse_hparams(hdict)


def _rec_parse_hparams(p_dict):
    hparams = {}
    for k in p_dict:
        v = p_dict[k]
        try:
            new_v = eval(v.replace('./', '')) if isinstance(v, str) else v
            if isinstance(new_v, dict):
                new_v = _rec_parse_hparams(new_v)
            hparams[k] = new_v
        except NameError:
            hparams[k] = v
    return hparams


class ExpAverage(object):
    def __init__(self, beta, bias_cor=False):
        self.beta = beta
        self.value = 0.
        self.bias_cor = bias_cor
        self.t = 0

    def reset(self):
        self.t = 0
        self.value = 0

    def update(self, v):
        self.t += 1
        self.value = self.beta * self.value + (1. - self.beta) * v
        if self.bias_cor:
            self.value = self.value / (1. - pow(self.beta, self.t))


class Count(object):
    def __init__(self, i=-1):
        self.i = i

    def inc(self):
        self.i += 1

    def getAndInc(self):
        r = self.i
        self.inc()
        return r

    def IncAndGet(self):
        self.inc()
        return self.i


class DummyException(RuntimeError):
    """ Method or function hasn't been implemented yet. """

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass


class SmilesDataset(Dataset):
    def __init__(self, x, y):
        self.X, processed_indices, invalid_indices = get_fp(x)
        self.y = y[processed_indices]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]


def np_to_plot_data(y):
    y = y.squeeze()
    if y.shape == ():
        return [float(y)]
    else:
        return y.squeeze().tolist()


def root_mean_squared_error(*args, **kwargs):
    return mean_squared_error(*args, **kwargs, squared=False)
