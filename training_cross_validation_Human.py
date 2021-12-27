# training_cross_validation_Human.py
#
# This file contains training code for the Human dataset.
#
# Author: Shugang Zhang
# Created: Friday, Aug 6th, 2021
# Last update: Friday, Aug 6th, 2021

import sys
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from models.sag_global_pooling import SAGNet_GLOBAL
from models.sag_hierarchical_pooling import SAGNet_HIER
from utils import *


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
        loss = criterion(output, data.y.view(-1, 1).float().to(device))
        # 4.backward
        loss.backward()   # propagate the gradient of trained weights backwardly
        optimizer.step()  # gradient update using a optimizer
        # 5.metrics
        current_batch_size = len(data.y)
        epoch_loss += loss.item()*current_batch_size

    print('Epoch {}: train_loss(mse): {:.5f} '.format(epoch, epoch_loss/len(train_loader.dataset)), end='')


def predicting(model, device, loader):
    model.eval()
    total_pred_values = torch.Tensor()
    total_pred_labels = torch.Tensor()
    total_true_labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            predicted_values = torch.sigmoid(output)  # continuous
            predicted_labels = torch.round(predicted_values)  # binary

            total_pred_values = torch.cat((total_pred_values, predicted_values.cpu()), 0)  # continuous
            total_pred_labels = torch.cat((total_pred_labels, predicted_labels.cpu()), 0)  # binary
            total_true_labels = torch.cat((total_true_labels, data.y.view(-1, 1).cpu()), 0)

    return total_true_labels.numpy().flatten(), total_pred_values.numpy().flatten(), total_pred_labels.numpy().flatten()


datasets = [['Human'][0]]
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet, SAGNet_GLOBAL, SAGNet_HIER][5]
model_st = modeling.__name__

print("dataset:", datasets)
print("modeling:", modeling)


cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
VALID_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.001  # 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000


print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
dataset = datasets[0]
print('\nrunning on ', model_st + '_' + dataset)
valid_results = []

# train & valid
for fold in range(1, 6):
    processed_data_file_train = 'data/processed/' + dataset + '_train_' + str(fold) + '.pt'
    processed_data_file_valid = 'data/processed/' + dataset + '_test_' + str(fold) + '.pt'
    if ((not os.path.isfile(processed_data_file_train)) or
            (not os.path.isfile(processed_data_file_valid))):
        print('please run prepare_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset+'_train_' + str(fold))
        valid_data = TestbedDataset(root='data', dataset=dataset+'_test_' + str(fold))

        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=VALID_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # best_mse = 1000
        # best_test_mse = 1000
        # best_test_ci = 0
        best_roc = 0
        best_epoch = -1
        model_file_name = 'pretrained/model_' + model_st + '_' + dataset + '_fold' + str(fold) + '.model'  # model
        # result_file_name = 'pretrained/result_' + model_st + '_' + dataset + '_fold' + str(fold) + '.csv'  # result

        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch+1)
            G, P, _ = predicting(model, device, valid_loader)
            valid_roc = roc_auc_score(G, P)
            print('| val_loss(mse): {:.5f}'.format(valid_roc))
            if valid_roc > best_roc:
                best_roc = valid_roc
                best_epoch = epoch+1
                torch.save(model.state_dict(), model_file_name)
                G, P, _ = predicting(model, device, valid_loader)
                tpr, fpr, _ = precision_recall_curve(G, P)
                ret = [roc_auc_score(G, P), auc(fpr, tpr)]
                # with open(result_file_name,'w') as f:
                #     f.write(','.join(map(str,ret)))
                test_roc = ret[0]
                test_prc = ret[1]

                # best_acc = float('%0.4f'%metrics[0])
                # best_roc = float('%0.4f'%metrics[1])
                # best_mse = float(metrics[0])
                # best_ci = float(metrics[1])
                print('AUROC improved at epoch ', best_epoch, '; test_auc:{:.5f}'.format(test_roc), '; test_prc:{:.5f}'.format(test_prc), model_st, dataset)
            else:
                print('No improvement since epoch ', best_epoch, '; test_auc:{:.5f}'.format(test_roc), '; test_prc:{:.5f}'.format(test_prc), model_st, dataset)


        # reload the best model and test it on valid set again to get other metrics
        model.load_state_dict(torch.load(model_file_name))
        G, P_value, P_label = predicting(model, device, valid_loader)

        tpr, fpr, _ = precision_recall_curve(G, P_value)

        valid_metrics = [roc_auc_score(G, P_value), auc(fpr, tpr), precision_score(G, P_label), recall_score(G, P_label)]
        print('Fold-{} valid finished, auc: {:.5f} | prc: {:.5f} | precision: {:.5f} | recall: {:.5f}'.format(str(fold), valid_metrics[0], valid_metrics[1], valid_metrics[2], valid_metrics[3]))
        valid_results.append([valid_metrics[0], valid_metrics[1], valid_metrics[2], valid_metrics[3]])


valid_results = np.array(valid_results)
valid_results = [np.mean(valid_results, axis=0), np.std(valid_results, axis=0)]


print("5-fold cross validation finished. "
      "auc:{:.3f}±{:.4f} | prc:{:.3f}±{:.4f} | precision:{:.3f}±{:.4f} | recall:{:.3f}±{:.4f}"
      .format(valid_results[0][0], valid_results[1][0], valid_results[0][1], valid_results[1][1], valid_results[0][2], valid_results[1][2], valid_results[0][3], valid_results[1][3]))

result_file_name = 'pretrained/result_' + model_st + '_' + dataset + '.txt'  # result

with open(result_file_name, 'w') as f:
    f.write("auc:{:.3f}±{:.4f} | prc:{:.3f}±{:.4f} | precision:{:.3f}±{:.4f} | recall:{:.3f}±{:.4f}"
      .format(valid_results[0][0], valid_results[1][0], valid_results[0][1], valid_results[1][1], valid_results[0][2], valid_results[1][2], valid_results[0][3], valid_results[1][3]))