import numpy as np
from sklearn.model_selection import train_test_split
from torch import optim

import os
import torch

def get_pn_data(root_path):
     pos= np.load(os.path.join(root_path,'pos_feature.npy'))
     neg = np.load(os.path.join(root_path,'neg_feature.npy'))
     p_label = [1]*len(pos)
     n_lael = [0]*len(neg)
     x = np.vstack((pos,neg))
     y = p_label+n_lael
     data = zip(x,y)
     data_train, data_test = train_test_split(data,test_size=0.2,random_state=20180911)
     return data_train,data_test


def get_pos_data(root_path):
    x = np.load (os.path.join (root_path, 'pos_feature.npy'))
    y = [1] * len (x)
    data_train = zip (x, y)
    return data_train
def get_neg_data(root_path):
    x = np.load (os.path.join (root_path, 'neg_feature.npy'))
    y = [0] * len (x)
    data_train = zip (x, y)
    return data_train



def acc_metric(true_y,pred_y):
    assert len(true_y) == len(pred_y)
    total = len(pred_y)
    true_num =0
    for i in range(total):
        if true_y[i] == pred_y[i]:
            true_num+=1
    acc =1.0*true_num/total
    return acc

def predict(model, test_data):
    model.eval ()
    true_y =[]
    pred_y = []
    for i, data in enumerate (test_data):
        x, y = data
        true_y.extend(y)
        out = model(x)
        _,index = map (lambda x: x.data.cpu ().numpy (), torch.max (out, 1))
        pred_y.extend(index) #
    return true_y, pred_y


if __name__ == '__main__':
    train_data, test_data = get_pn_data ('../data/gen_data')
    print(train_data[0])