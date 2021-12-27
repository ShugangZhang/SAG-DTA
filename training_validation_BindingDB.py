# training_validation_BindingDB.py
#
# This file contains training code for the BindingDB dataset.
#
# Author: Shugang Zhang
# Created: Wednesday, Aug 4th, 2021
# Last update: Thursday, Aug 5th, 2021

import sys
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from models.sag_global_pooling import SAGNet_GLOBAL
from models.sag_hierarchical_pooling import SAGNet_HIER
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from utils import *

# training function at each epoch
def train_back(model, device, train_loader, optimizer, epoch):
    # print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    # print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    epoch_loss = 0
    epoch_mse = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        # 1.zero grad
        optimizer.zero_grad()
        # 2.forward
        output = model(data)
        # 3.compute diff
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        # 4.backward
        loss.backward()   # propagate the gradient of trained weights backwardly
        optimizer.step()  # gradient update using a optimizer
        # 5.metrics
        current_batch_size = len(data.y)
        epoch_loss += loss.item()*current_batch_size

    print('Epoch {}: train_loss: {:.5f} '.format(epoch, epoch_loss/len(train_loader.dataset)), end='')


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    # print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


datasets = ['BindingDB']
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet, SAGNet_HIER, SAGNet_GLOBAL][5]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = ["cuda:0", "cuda:1"][int(sys.argv[3])]
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.001
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset)
    processed_train_data = 'data/processed/' + dataset + '_train.pt'
    processed_valid_data = 'data/processed/' + dataset + '_dev.pt'
    processed_test_data = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_train_data)) or (not os.path.isfile(processed_valid_data)) or (not os.path.isfile(processed_test_data))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset+'_train')
        valid_data = TestbedDataset(root='data', dataset=dataset + '_dev')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test')

        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.BCEWithLogitsLoss()  # https://blog.csdn.net/weixin_40522801/article/details/106616564
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_roc = 0
        best_epoch = -1
        model_file_name = 'model_' + model_st + '_' + dataset +  '.model'
        result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch+1)
            G, P = predicting(model, device, valid_loader)
            valid_roc = roc_auc_score(G, P)
            if valid_roc > best_roc:
                best_roc = valid_roc
                best_epoch = epoch+1  # start from 1 instead of 0
                torch.save(model.state_dict(), model_file_name)
                G, P = predicting(model, device, test_loader)
                tpr, fpr, _ = precision_recall_curve(G, P)
                ret = [roc_auc_score(G, P), auc(fpr, tpr)]
                with open(result_file_name,'w') as f:
                    f.write(','.join(map(str,ret)))
                test_roc = ret[0]
                test_prc = ret[1]

                print('AUROC improved at epoch ', best_epoch, '; best_valid_auc:{:.5f}'.format(best_roc),
                      '; test_auc:{:.5f}'.format(test_roc), '; test_prc:{:.5f}'.format(test_prc), model_st, dataset)
            else:
                print('No improvement since epoch ', best_epoch, '; best_valid_auc:{:.5f}'.format(best_roc),
                      '; test_auc:{:.5f}'.format(test_roc), '; test_prc:{:.5f}'.format(test_prc), model_st, dataset)

