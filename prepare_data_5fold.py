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
datasets = ['kiba','davis']
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = 'data/' + dataset + '/'
    train_fold_origin = json.load(open(fpath + "folds/train_fold_setting1.txt"))  # len=5
    train_fold_origin = [e for e in train_fold_origin]  # 5-fold
    test_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))  # davis len=5010
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)  # davis len=68
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)  # davis len=442
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')  # davis len=68
    drugs = []
    prots = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)  # loading drugs
    for t in proteins.keys():
        prots.append(proteins[t])  # loading proteins
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]
    affinity = np.asarray(affinity)  # affinity shape=(68 drug,442 prot)


    opts = ['train','test']
    for opt in opts:
        if opt == 'train':
            for fold in range(5):
                train_rows, train_cols = np.where(np.isnan(affinity) == False)  # not NAN
                valid_rows, valid_cols = np.where(np.isnan(affinity) == False)  # not NAN
                train_folds = []
                valid_fold = train_fold_origin[fold]  # specify 1 fold as valid set
                for i in range(len(train_fold_origin)):
                    if i != fold:
                        train_folds += train_fold_origin[i]
                train_rows, train_cols = train_rows[train_folds], train_cols[train_folds]  # train fold包含了drug和prot的index信息
                valid_rows, valid_cols = valid_rows[valid_fold], valid_cols[valid_fold]

                with open('data/' + dataset + '_fold' + str(fold+1) + '_train.csv', 'w') as f:
                    f.write('compound_iso_smiles,target_sequence,affinity\n')
                    for pair_ind in range(len(train_rows)):
                        ls = []
                        ls += [drugs[train_rows[pair_ind]]]
                        ls += [prots[train_cols[pair_ind]]]
                        ls += [affinity[train_rows[pair_ind], train_cols[pair_ind]]]
                        f.write(','.join(map(str, ls)) + '\n')  # csv format

                with open('data/' + dataset + '_fold' + str(fold+1) + '_valid.csv', 'w') as f:
                    f.write('compound_iso_smiles,target_sequence,affinity\n')
                    for pair_ind in range(len(valid_rows)):
                        ls = []
                        ls += [drugs[valid_rows[pair_ind]]]
                        ls += [prots[valid_cols[pair_ind]]]
                        ls += [affinity[valid_rows[pair_ind], valid_cols[pair_ind]]]
                        f.write(','.join(map(str, ls)) + '\n')  # csv format

                print('train_fold_' + str(fold+1) + ':', len(train_folds))
                print('valid_fold_' + str(fold+1) + ':', len(valid_fold))

        elif opt == 'test':
            rows, cols = np.where(np.isnan(affinity) == False)  # not NAN
            rows, cols = rows[test_fold], cols[test_fold]
            with open('data/' + dataset + '_test.csv', 'w') as f:
                f.write('compound_iso_smiles,target_sequence,affinity\n')
                for pair_ind in range(len(rows)):
                    ls = []
                    ls += [ drugs[rows[pair_ind]]  ]
                    ls += [ prots[cols[pair_ind]]  ]
                    ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                    f.write(','.join(map(str,ls)) + '\n')  # csv format
    print('test_fold:', len(test_fold))

    print('len(set(drugs)),len(set(prots)):', len(set(drugs)), len(set(prots)))
    all_prots += list(set(prots))
    
    
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}  # encode alphabet from 1
seq_dict_len = len(seq_dict)
max_seq_len = 1000

# 2.create graph for all SMILES
print("\nCreating graph for all SMILES...")
compound_iso_smiles = []
for dt_name in ['kiba','davis']:
    for fold in range(5):
        df_train = pd.read_csv('data/' + dt_name + '_fold' + str(fold+1) + '_train.csv')
        df_valid = pd.read_csv('data/' + dt_name + '_fold' + str(fold+1) + '_valid.csv')
        compound_iso_smiles += list(df_train['compound_iso_smiles'])
        compound_iso_smiles += list(df_valid['compound_iso_smiles'])

    df = pd.read_csv('data/' + dt_name + '_test.csv')
    compound_iso_smiles += list(df['compound_iso_smiles'])
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g


datasets = ['davis','kiba']
# convert to PyTorch data format
for dataset in datasets:
    # 5-fold, train&valid sets
    for fold in range(5):
        # read in train file
        df = pd.read_csv('data/' + dataset + '_fold' + str(fold+1) + '_train.csv')
        train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])
        train_XT = [seq_cat(t) for t in train_prots]
        train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(train_XT), np.asarray(train_Y)
        # read in valid file
        df = pd.read_csv('data/' + dataset + '_fold' + str(fold+1) + '_valid.csv')
        valid_drugs, valid_prots, valid_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])
        valid_XT = [seq_cat(t) for t in valid_prots]
        valid_drugs, valid_prots, valid_Y = np.asarray(valid_drugs), np.asarray(valid_XT), np.asarray(valid_Y)

        processed_data_file_train = 'data/processed/' + dataset + '_fold' + str(fold+1) + '_train.pt'
        processed_data_file_valid = 'data/processed/' + dataset + '_fold' + str(fold+1) + '_valid.pt'
        # splitting
        if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_valid))):
            # print("shape:", train_drugs.shape, train_prots.shape, train_Y.shape)

            # make data PyTorch Geometric ready
            print('preparing', dataset + '_fold' + str(fold+1) + '_train.pt in pytorch format!')
            train_data = TestbedDataset(root='data', dataset=dataset + '_fold' + str(fold + 1) + '_train',
                                        xd=train_drugs, xt=train_prots, y=train_Y, smile_graph=smile_graph)
            print('preparing', dataset + '_fold' + str(fold+1) + '_valid.pt in pytorch format!')
            valid_data = TestbedDataset(root='data', dataset=dataset + '_fold' + str(fold + 1) + '_valid',
                                        xd=valid_drugs, xt=valid_prots, y=valid_Y, smile_graph=smile_graph)
            print(processed_data_file_train, processed_data_file_valid, 'are created successfully.\n')
        else:
            print(processed_data_file_train, processed_data_file_valid, 'are already created.')

    # test set
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if not os.path.isfile(processed_data_file_test):
        df = pd.read_csv('data/' + dataset + '_test.csv')
        drugs, prots, Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])
        XT = [seq_cat(t) for t in prots]
        drugs, prots, Y = np.asarray(drugs), np.asarray(XT), np.asarray(Y)

        # make data PyTorch Geometric ready
        print('preparing ', dataset + '_test.pt in pytorch format!')
        test_data = TestbedDataset(root='data', dataset=dataset + '_test',
                                    xd=drugs, xt=prots, y=Y, smile_graph=smile_graph)
        print(processed_data_file_test, 'created successfully.\n')
    else:
        print(processed_data_file_test, 'are already created.')
