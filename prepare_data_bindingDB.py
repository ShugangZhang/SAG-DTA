# prepare_data_bindingDB.py
#
# This is a data pre-processing file for generating BindingDB dataset in PyTorch format.
#
# Author: Shugang Zhang
# Created: Wednesday, Aug 4th, 2021
# Last update: Wednesday, Aug 4th, 2021

import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
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
            features.append( feature / sum(feature) )
        # 3.edges
        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])

        return num_atoms, features, edge_index

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  


# from DeepDTA data
all_prots = []
datasets = ['BindingDB']
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = 'data/' + dataset + '/'
    train_data = pd.read_table(fpath + "train.txt", sep=' ', header=None)  # len=5
    valid_data = pd.read_table(fpath + "dev.txt", sep=' ', header=None)
    test_data = pd.read_table(fpath + "test.txt", sep=' ', header=None)
    # ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)  # davis len=68
    # proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)  # davis len=442
    # affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')  # davis len=68
    drugs = []
    prots = []
    print('\ndataset:', dataset)
    print('train_fold:', len(train_data))
    print('valid_fold:', len(valid_data))
    print('test_fold:', len(test_data))
    all_prots += list(set(prots))
    
    
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}  # encode alphabet from 1
seq_dict_len = len(seq_dict)
max_seq_len = 1000

# 2.create graph for all SMILES
print("\nCreating graph for all SMILES...")
compound_iso_smiles = []
for dt_name in ['BindingDB']:
    opts = ['train', 'dev', 'test']
    for opt in opts:
        df = pd.read_table('data/' + dt_name + '/' + opt + '.txt', sep=' ', header=None)
        compound_iso_smiles += list(df[0])  # the first column is drug SMILES
compound_iso_smiles = set(compound_iso_smiles)  # function set() can remove redundant automatically
smile_graph = {}

for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g
print("Finished.")


datasets = ['BindingDB']
# convert to PyTorch data format
print("\nConvert to pytorch data format...")
for dataset in datasets:
    opts = ['train', 'dev', 'test']
    for opt in opts:
        df = pd.read_table('data/' + dt_name + '/' + opt + '.txt', sep=' ', header=None)
        processed_data = 'data/processed/' + dataset + '_' + opt + '.pt'
        if not os.path.isfile(processed_data):  # if not exists
            df = pd.read_table('data/' + dataset + '/' + opt + '.txt', sep=' ', header=None)
            drugs, prots, Y = list(df[0]), list(df[1]), list(df[2])
            XT = [seq_cat(t) for t in prots]
            drugs, prots, Y = np.asarray(drugs), np.asarray(XT), np.asarray(Y)

            # make data PyTorch Geometric ready
            print('preparing ', dataset + '_' + opt + '.pt in pytorch format!')
            valid_data = TestbedDataset(root='data', dataset=dataset + '_' + opt,
                                        xd=drugs, xt=prots, y=Y, smile_graph=smile_graph)
            print(processed_data, 'created successfully.')
        else:
            print(processed_data, 'are already created.')
