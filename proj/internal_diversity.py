# Author: bbrighttaer
# Project: irelease
# Date: 7/15/2020
# Time: 11:59 AM
# File: internal_diversity.py

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem


def verify_sequence(smile):
    mol = Chem.MolFromSmiles(smile)
    return smile != '' and mol is not None and mol.GetNumAtoms() > 1


def batch_internal_diversity(smiles):
    """
    Calculates internal diversity of the given compounds.
    See http://arxiv.org/abs/1708.08227

    :param smiles:
    :param set_smiles:
    :return:
    """
    rand_mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 4, nBits=2048) for m in rand_mols]
    vals = [bulk_tanimoto_distance(s, fps) if verify_sequence(s) else 0.0 for s in smiles]
    return np.mean(vals)


def bulk_tanimoto_distance(smile, fps):
    ref_mol = Chem.MolFromSmiles(smile)
    ref_fps = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 4, nBits=2048)
    dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
    return dist


if __name__ == '__main__':
    comps = ['COc1cc(Cc2ccccc2Cl)ncc1NC(=N)Nc1ccc(C(=O)N=C2NCCN2C)cc1',
             'Oc1ccc2ccccc2c1-c1nc2ccccc2[nH]1',
             'CN(C)CCn1c(Nc2ccccc2)nc2ccc(NC3CCCC3)cc21',
             'CCCCCCCCCCCCCC(=O)N(C(=O)CCCCCCCC(C)C)C(C)C',
             'COc1ccc(-c2ncnc(N3CCCC3C(=O)NC3CCOC3)n2)cc1']
    print(batch_internal_diversity(comps))
