# prepare_data_Human.py
#
# 1. This is a data pre-processing file for generating Human dataset in PyTorch format.
# 2. The dataset was not divided in its given format. Therefore, it was processed into
#    five subsets, each of which acted as test set once, i.e., 5-fold.
#
# Author: Shugang Zhang
# Created: Friday, Aug 6th, 2021
# Last update: Friday, Aug 6th, 2021

import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if (mol is None):
        print("bad smile:", smile)
    else:
        # 1.num of atoms
        num_atoms = mol.GetNumAtoms()
        # 2.features
        features = []
        for atom in mol.GetAtoms():
            feature = atom_features(atom)
            features.append(feature / sum(feature))
        # 3.edges
        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])

        # if(edge_index == []):
        #     print(smile)  # molecules that have no bonds

        return num_atoms, features, edge_index


def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x


datasets = ['Human']
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = 'data/' + dataset + '/' + dataset + '.txt'
    data = pd.read_table(fpath, sep=' ', header=None)

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}  # encode alphabet from 1
seq_dict_len = len(seq_dict)
max_seq_len = 1000

# 2. create graph for all SMILES
print("\nCreating graph for all SMILES...")
compound_iso_smiles = []
for dataset in datasets:
    df = pd.read_table('data/' + dataset + '/' + dataset + '.txt', sep=' ', header=None)
    compound_iso_smiles += list(df[0])  # the first column is drug SMILES
compound_iso_smiles = set(compound_iso_smiles)  # function set() can remove redundant automatically
smile_graph = {}

for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g
print("Finished.")

# 3. convert to PyTorch data format
print("\nConvert to pytorch data format...")
for dataset in datasets:
    df_raw = pd.read_table('data/' + dataset + '/' + dataset + '.txt', sep=' ', header=None)

    df = df_raw.drop(df[(df[0] == '[I]')
                        | (df[0] == '[Na+].[Na+].[Cl-].[Cl-].[Cl-].[Cl-].[Pt+2]')
                        | (df[0] == 'C.[Ti]')
                        | (df[0] == '[Na+].[Ag+]')
                        | (df[0] == '[H-].[Na+]')
                        | (df[0] == '[23Na]')
                        | (df[0] == '[Rb+].[I-]')
                        | (df[0] == '[SiH2]')
                        | (df[0] == '[43Ca]')
                        | (df[0] == 'O.O.[Zn]')
                        | (df[0] == '[O-2].[O-2].[O-2].[In+3].[In+3]')
                        | (df[0] == '[Al].[Si]')
                        | (df[0] == '[3He]')
                        | (df[0] == '[110Ag]')
                        | (df[0] == 'O.[Cu].[Cu]')
                        | (df[0] == '[77BrH]')
                        | (df[0] == '[43K]')
                        | (df[0] == '[78Se]')
                        | (df[0] == '[115Cd]')
                        | (df[0] == '[NH4+].F.[F-]')
                        | (df[0] == '[O-2].[O-2].[O-2].[Y+3].[Y+3]')
                        | (df[0] == '[125IH]')
                        | (df[0] == '[C]')
                        | (df[0] == '[Br]')
                        | (df[0] == '[85Sr+2]')
                        | (df[0] == '[11CH4]')
                        | (df[0] == '[NH2-].[Cl-].[Hg+2]')
                        | (df[0] == '[Na+].[Na+].[S-2]')
                        | (df[0] == '[O-2].[O-2].[O-2].[O-2].[Co+2].[Co+3].[Co+3]')
                        | (df[0] == '[129IH]')
                        | (df[0] == '[68Ge]')
                        | (df[0] == '[Ho]')
                        | (df[0] == '[132IH]')
                        | (df[0] == '[9Li]')
                        | (df[0] == '[56Ni]')
                        | (df[0] == '[214Pb]')
                        | (df[0] == '[15NH3]')
                        | (df[0] == '[As+5]')
                        | (df[0] == '[197Au]')
                        | (df[0] == '[F-].[F-].[K+].[K+]')
                        | (df[0] == '[Cl-].[Cl-].[Ni+2]')
                        | (df[0] == '[Li+].[Li+].[Li+].[Li+].[CH3-]')
                        | (df[0] == '[210Tl]')
                        | (df[0] == '[H+].[H+].[H+].[OH-]')
                        | (df[0] == '[63Cu]')
                        | (df[0] == 'N.N.N.N.N.N.[Cl-].[Cl-].[Cl-].[Co+3]')
                        | (df[0] == '[NH4+].[Br-]')
                        | (df[0] == 'O.[Ag].[Ag]')
                        | (df[0] == 'O.O.O.O.O.O.[Cl-].[Cl-].[Ni+2]')
                        | (df[0] == '[252Cf]')
                        | (df[0] == '[OH-].[OH-].[OH-].[OH-].[Ir]')
                        | (df[0] == '[178W]')
                        | (df[0] == 'O.[Hg].[Hg]')
                        | (df[0] == '[33SH2]')
                        | (df[0] == '[44Ca]')
                        | (df[0] == '[H]')
                        | (df[0] == '[89Sr+2]')
                        | (df[0] == '[H-].[Al+3]')
                        | (df[0] == 'O.O.O.O.[Mo]')
                        | (df[0] == '[Si+4]')
                        | (df[0] == '[Cl]')
                        | (df[0] == '[Li+].[AlH4-]')
                        | (df[0] == '[Zn]')
                        | (df[0] == 'O.O.O.O.[V]')
                        | (df[0] == '[O-2].[O-2].[O-2].[O-2].[O-2].[Ta+5].[Ta+5]')
                        | (df[0] == '[90Sr+2]')
                        | (df[0] == '[34SH2]')
                        | (df[0] == '[84BrH]')
                        | (df[0] == '[127IH]')
                        | (df[0] == '[Fe]')
                        | (df[0] == '[141Ce]')
                        | (df[0] == '[207Pb]')
                        | (df[0] == '[75Se]')
                        | (df[0] == '[17NH3]')
                        | (df[0] == '[209Pb]')
                        | (df[0] == '[H-].[Li+]')
                        | (df[0] == '[56Fe]')
                        | (df[0] == '[F]')
                        | (df[0] == 'N.N.[OH-].[OH-].[Cl-].[Cl-].[Pt+4]')
                        | (df[0] == '[Ge].[Fr]')
                        | (df[0] == '[11B]')
                        | (df[0] == '[99Mo]')
                        | (df[0] == '[2H]')
                        | (df[0] == '[173Ta]')
                        | (df[0] == '[7Li]')
                        | (df[0] == '[O-2].[O-2].[O-2].[O-2].[O-2].[Al+3].[Al+3].[Ca+2].[Ca+2]')
                        | (df[0] == '[12CH4]')
                        | (df[0] == '[109Cd]')
                        | (df[0] == '[125Sb]')
                        | (df[0] == '[Mg+2].[Br-].[Br-]')
                        | (df[0] == '[F-].[F-].[Ca+2]')
                        | (df[0] == '[79Br-]')
                        | (df[0] == 'S.[As]')
                        | (df[0] == '[29Si]')
                        | (df[0] == 'C.C.C.C.C.C.O.O.O.O.O.O.O')
                        | (df[0] == '[81Br-]')
                        | (df[0] == '[13NH3]')
                        | (df[0] == '[Br-].[Rb+]')
                        | (df[0] == '[16OH2]')
                        | (df[0] == '[60Fe]')
                        | (df[0] == '[128IH]')
                        | (df[0] == '[205Pb]')
                        | (df[0] == 'O.O.O.[Co]')
                        | (df[0] == '[102Ru]')
                        | (df[0] == '[45Ca+2]')
                        | (df[0] == '[Si]')
                        | (df[0] == '[80Se]')
                        | (df[0] == '[218Po]')
                        | (df[0] == 'N.N.N.N.N.N.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.[Mo].[Mo].[Mo].[Mo].[Mo].[Mo].[Mo]')
                        | (df[0] == '[NH4+].[I-]')
                        | (df[0] == '[10B]')
                        | (df[0] == '[Li+].[OH-]')
                        | (df[0] == '[Mg+2]')
                        | (df[0] == '[113In]')
                        | (df[0] == '[121Sn]')
                        | (df[0] == '[183W]')
                        | (df[0] == '[OH]')
                        | (df[0] == '[45K]')
                        | (df[0] == '[188W]')
                        | (df[0] == '[B-]')
                        | (df[0] == 'O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.[Si].[K].[K].[K].[K].[K].[Cu].[W].[W].[W].[W].[W].[W].[W].[W].[W].[W].[W]')
                        | (df[0] == '[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Si+4].[W].[W].[W].[W].[W].[W].[W].[W].[W]')

                        ].index)


    print(len(list(df_raw[0])))
    print(len(list(df[0])))
    portion = int(0.2 * len(df[0]))
    drugs, prots, Y = list(df[0]), list(df[1]), list(df[2])
    XT = [seq_cat(t) for t in prots]
    drugs, prots, Y = np.asarray(drugs), np.asarray(XT), np.asarray(Y)

    for fold in range(5):
        if fold < 4:
            # test set
            test_drugs = drugs[fold * portion:(fold + 1) * portion]
            test_prots = prots[fold * portion:(fold + 1) * portion]
            test_Y = Y[fold * portion:(fold + 1) * portion]
            # train set
            train_drugs = np.delete(drugs, range(fold * portion, (fold + 1) * portion))
            train_prots = np.delete(prots, range(fold * portion, (fold + 1) * portion), axis=0)
            train_Y = np.delete(Y, range(fold * portion, (fold + 1) * portion))

        if fold == 4:  # to avoid cases that the number of samples cannot be divided by 5
            # test set
            test_drugs = drugs[fold * portion:]
            test_prots = prots[fold * portion:]
            test_Y = Y[fold * portion:]
            # train set
            train_drugs = np.delete(drugs, range(fold * portion, len(drugs)))
            train_prots = np.delete(prots, range(fold * portion, len(drugs)), axis=0)
            train_Y = np.delete(Y, range(fold * portion, len(drugs)))

        # make data PyTorch Geometric ready
        # train set
        data_path = 'data/processed/' + dataset + '_train_' + str(fold + 1) + '.pt'
        if not os.path.isfile(data_path):  # if not exists
            print('preparing ', dataset + '_train_' + str(fold + 1) + '.pt in pytorch format!')
            TestbedDataset(root='data', dataset=dataset + '_train_' + str(fold + 1),
                           xd=train_drugs, xt=train_prots, y=train_Y, smile_graph=smile_graph)
            print(data_path, 'created successfully.')
        else:
            print(data_path, 'are already created.')
        # test set
        data_path = 'data/processed/' + dataset + '_test_' + str(fold + 1) + '.pt'
        if not os.path.isfile(data_path):  # if not exists
            print('preparing ', dataset + '_test_' + str(fold + 1) + '.pt in pytorch format!')
            TestbedDataset(root='data', dataset=dataset + '_test_' + str(fold + 1),
                           xd=test_drugs, xt=test_prots, y=test_Y, smile_graph=smile_graph)
        else:
            print(data_path, 'are already created.')
