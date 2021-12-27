# predict_with_pretrained_model_BindingDB.py
#
# This file is for a quick check of the reported results in the paper.
# It contains replication results only for BindingDB.
#
# Author: Shugang Zhang
# Created: Wednesday, Aug 4th, 2021
# Last update: Thursday, Aug 5th, 2021

from models.sag_global_pooling import SAGNet_GLOBAL
from models.sag_hierarchical_pooling import SAGNet_HIER
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
from utils import *


def predicting(model, device, loader):
    model.eval()
    total_pred_values = torch.Tensor()
    total_pred_labels = torch.Tensor()
    total_true_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            # print("output.shape = ", output.shape)
            predicted_values = torch.sigmoid(output)  # continuous
            predicted_labels = torch.round(predicted_values)  # binary

            total_pred_values = torch.cat((total_pred_values, predicted_values.cpu()), 0)  # continuous
            total_pred_labels = torch.cat((total_pred_labels, predicted_labels.cpu()), 0)  # binary
            total_true_labels = torch.cat((total_true_labels, data.y.view(-1, 1).cpu()), 0)

    # return (G, P_value, P_label)
    return total_true_labels.numpy().flatten(), total_pred_values.numpy().flatten(), total_pred_labels.numpy().flatten()


datasets = ['BindingDB']
modelings = [SAGNet_HIER, SAGNet_GLOBAL]
cuda_name = "cuda:0"
print('cuda_name:', cuda_name)

TEST_BATCH_SIZE = 512

result = []
for dataset in datasets:
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if (not os.path.isfile(processed_data_file_test)):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        test_data = TestbedDataset(root='data', dataset=dataset+'_test')
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        for modeling in modelings:
            model_st = modeling.__name__
            print('\npredicting for ', dataset, ' using ', model_st)
            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = modeling().to(device)
            model_file_name = 'pretrained/model_' + model_st + '_' + dataset + '.model'
            if os.path.isfile(model_file_name):            
                model.load_state_dict(torch.load(model_file_name), strict=False)
                G, P_value, P_label = predicting(model, device, test_loader)
                tpr, fpr, _ = precision_recall_curve(G, P_value)

                ret = [roc_auc_score(G, P_value), auc(fpr, tpr), precision_score(G, P_label), recall_score(G, P_label)]
                ret = [dataset, model_st] + [round(e, 3) for e in ret]
                result += [ret]
                print('dataset, model, auc, prc, precision, recall')
                print(ret)
            else:
                print('model is not available!')
with open('result.csv', 'w') as f:
    f.write('dataset,model,auc,prc,precision,recall\n')
    for ret in result:
        f.write(','.join(map(str, ret)) + '\n')