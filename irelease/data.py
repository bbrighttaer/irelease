# Original GeneratorData code from: https://github.com/isayev/ReLeaSE

import numpy as np
import torch
from irelease.smiles_enumerator import SmilesEnumerator

from irelease.utils import read_smi_file, tokenize, read_object_property_file, seq2tensor, pad_sequences, ReplayBuffer, \
    get_default_tokens


class GeneratorData(object):

    def __init__(self, training_data_path, tokens=None, start_token='<',
                 end_token='>', pad_symbol=' ', max_len=120, use_cuda=None, seed=None,
                 **kwargs):
        """
        Constructor for the GeneratorData object.

        Parameters
        ----------
        training_data_path: str
            path to file with training dataset. Training dataset must contain
            a column with training strings. The file also may contain other
            columns.

        tokens: list (default None)
            list of characters specifying the language alphabet. If left
            unspecified, tokens will be extracted from data automatically.

        start_token: str (default '<')
            special character that will be added to the beginning of every
            sequence and encode the sequence start.

        end_token: str (default '>')
            special character that will be added to the end of every
            sequence and encode the sequence end.

        max_len: int (default 120)
            maximum allowed length of the sequences. All sequences longer than
            max_len will be excluded from the training data.

        use_cuda: bool (default None)
            parameter specifying if GPU is used for computations. If left
            unspecified, GPU will be used if available

        kwargs: additional positional arguments
            These include cols_to_read (list, default [0]) specifying which
            column in the file with training data contains training sequences
            and delimiter (str, default ',') that will be used to separate
            columns if there are multiple of them in the file.

        """
        super(GeneratorData, self).__init__()
        if seed:
            np.random.seed(seed)

        if 'cols_to_read' not in kwargs:
            kwargs['cols_to_read'] = []
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        if 'tokens_reload' in kwargs:
            self.tokens_reload = kwargs['tokens_reload']

        data = read_object_property_file(training_data_path, **kwargs)
        self.start_token = start_token
        self.end_token = end_token
        self.pad_symbol = pad_symbol
        self.file = []
        for i in range(len(data)):
            if len(data[i]) <= max_len:
                self.file.append(self.start_token + data[i].strip() + self.end_token)
        self.file_len = len(self.file)
        self.all_characters, self.char2idx, \
        self.n_characters = tokenize(self.file, tokens)
        self.pad_symbol_idx = self.all_characters.index(self.pad_symbol)
        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()

    def load_dictionary(self, tokens, char2idx):
        self.all_characters = tokens
        self.char2idx = char2idx
        self.n_characters = len(tokens)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def random_chunk(self, batch_size):
        """
        Samples random SMILES string from generator training data set.
        Returns:
            random_smiles (str).
        """
        index = np.random.randint(0, self.file_len - 1, batch_size)
        return [self.file[i][:-1] for i in index], [self.file[i][1:] for i in index]

    def random_training_set_smiles(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        assert (batch_size > 0)
        sels = np.random.randint(0, self.file_len - 1, batch_size)
        return [self.file[i][1:-1] for i in sels]

    def random_training_set(self, batch_size=None, return_seq_len=False):
        if batch_size is None:
            batch_size = self.batch_size
        assert (batch_size > 0)
        inp, target = self.random_chunk(batch_size)
        inp_padded, inp_seq_len = pad_sequences(inp)
        inp_tensor, self.all_characters = seq2tensor(inp_padded,
                                                     tokens=self.all_characters,
                                                     flip=False)
        target_padded, target_seq_len = pad_sequences(target)
        target_tensor, self.all_characters = seq2tensor(target_padded,
                                                        tokens=self.all_characters,
                                                        flip=False)
        self.n_characters = len(self.all_characters)
        inp_tensor = torch.tensor(inp_tensor).long()
        target_tensor = torch.tensor(target_tensor).long()
        if self.use_cuda:
            inp_tensor = inp_tensor.cuda()
            target_tensor = target_tensor.cuda()
        if return_seq_len:
            return inp_tensor, target_tensor, (inp_seq_len, target_seq_len)
        return inp_tensor, target_tensor

    def read_sdf_file(self, path, fields_to_read):
        raise NotImplementedError

    def update_data(self, path):
        self.file, success = read_smi_file(path, unique=True)
        self.file_len = len(self.file)
        assert success


class BinaryClassificationData:

    def __init__(self, buffer_size, enum=True, device='cpu'):
        self.enum = enum
        self.device = device
        self.smiles_enum = SmilesEnumerator()
        self._pos_buffer = ReplayBuffer(buffer_size)
        self._neg_buffer = ReplayBuffer(buffer_size)

    def populate_pos(self, samples):
        self._pos_buffer.populate(samples)

    def populate_neg(self, samples):
        self._neg_buffer.populate(samples)

    def _data_to_tensor(self, data):
        data, _ = pad_sequences(data)
        data, _ = seq2tensor(data, tokens=get_default_tokens())
        data = torch.from_numpy(data).long().to(self.device)
        return data

    def _augment(self, samples):
        new_samples = list(samples)
        for s in samples:
            new_samples.append(self.smiles_enum.randomize_smiles(s))
        return new_samples

    def sample(self, batch):
        assert (batch > 1), 'BC data batch size must be greater than 1'
        pos_data = list(set(self._pos_buffer.sample(batch, False)))
        # pos_data = self._augment(pos_data)
        neg_data = list(set(self._neg_buffer.sample(batch, False)))
        min_size = min(len(pos_data), len(neg_data))
        # aug_smiles = []
        # diff = len(neg_data) - len(pos_data)
        # if len(pos_data) < len(neg_data):
        #     while len(aug_smiles) < diff:
        #         for i in range(diff - len(aug_smiles)):
        #             sm = pos_data[np.random.choice(len(pos_data))]
        #             aug_smiles.append(self.smiles_enum.randomize_smiles(sm))
        #         # aug_smiles = [s for s in aug_smiles if s not in pos_data]
        # pos_data.extend(aug_smiles)
        pos_data = ['<' + s + '>' for s in pos_data]
        pos_data = np.array(pos_data, dtype=np.object)
        neg_data = np.array(neg_data, dtype=np.object)
        pos_data = pos_data[np.random.choice(np.arange(len(pos_data)), min_size, replace=False)]
        neg_data = neg_data[np.random.choice(np.arange(len(neg_data)), min_size, replace=False)]
        pos_neg_data = np.concatenate([pos_data, neg_data])
        x = self._data_to_tensor(pos_neg_data)
        t_pos_labels = torch.ones(len(pos_data), 1)
        t_neg_labels = torch.zeros(len(neg_data), 1)
        y = torch.cat([t_pos_labels, t_neg_labels]).to(self.device)
        return x, y
