# Author: bbrighttaer
# Project: GPMT
# Date: 5/28/2020
# Time: 4:22 PM
# File: create_dopamine_activity_data.py

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse

from tqdm import trange

from gpmt.data import GeneratorData
from gpmt.utils import get_default_tokens
from gpmt.drd2 import Activity_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Creates activity dataset for dopamine receptor type 2 (DRD2)')
    parser.add_argument('--svc', type=str, help='Path to the SVC pickled object')
    parser.add_argument('--threshold', type=float, default=0.8, help='Activity threshold for DRD2')
    parser.add_argument('--data', type=str, help='The SMILES data file to be used in creating the activity dataset')
    parser.add_argument('--save_dir', type=str, default='../data', help='The directory to save the created dataset')
    parser.add_argument('--filename', type=str, default='drd2_active.smi', help='The filename for the created dataset')
    args = parser.parse_args()

    assert(os.path.exists(args.svc))
    assert(os.path.exists(args.data))
    assert(0 < args.threshold < 1)

    # Load file containing SMILES
    gen_data = GeneratorData(training_data_path=args.data,
                             delimiter='\t',
                             cols_to_read=[0],
                             keep_header=True,
                             pad_symbol=' ',
                             max_len=120,
                             tokens=get_default_tokens(),
                             use_cuda=False)

    # Load classifier
    clf = Activity_model(args.svc)

    # Screen SMILES in data file and write active compounds to file.
    os.makedirs(args.save_dir, exist_ok=True)
    num_active = 0
    with open(os.path.join(args.save_dir, args.filename), 'w') as f:
        for i in trange(gen_data.file_len, desc='Screening compounds...'):
            smiles = gen_data.file[i][1:-1]
            p = clf(smiles)
            if p >= args.threshold:
                f.write(smiles + '\n')
                num_active += 1
    print(f'Total number of actives written to file = {num_active}')
