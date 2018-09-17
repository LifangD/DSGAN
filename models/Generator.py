import torch
from torch import  nn,optim
from torch.autograd import Variable
from models.PCNN import PCNN
import logging
from  utils.data_utils import predict,acc_metric
import numpy as np


class Generator(object):
    def __init__(self,config):
        self.config = config
        self.trainModel = PCNN(config)
        if self.config.use_cuda:
            self.trainModel.cuda ()
        self.optimizer = self.select_opt(config.opt_method)

    def pretrain(self, train_loader, test_loader):
        n_epochs = self.config.n_epochs
        batch_size = self.config.batch_size
        criterion =nn.NLLLoss()
        best_acc = 0
        for epoch in range(n_epochs):
            train_loss = 0
            for i,batch_data in enumerate (train_loader):
                self.optimizer.zero_grad ()
                data,label = batch_data
                if self.config.use_cuda:
                    label = torch.LongTensor(label).cuda()
                else:
                    label = torch.LongTensor(label)
                out = self.trainModel(data)
                loss = criterion(out,Variable(label))
                loss.backward ()
                self.optimizer.step()
                train_loss += loss.data[0]
            print('Epoch{}/{}, Train_Loss={:.3f}'.format(epoch + 1, n_epochs, train_loss / batch_size))


            if epoch%self.config.epoch_per_test ==0:
                true_y,pred_y = predict(self.trainModel,test_loader)
                eval_acc = acc_metric(true_y,pred_y)
                if  best_acc < eval_acc:
                    best_acc = eval_acc
                    self.trainModel.save(self.config.model_name)
                    print ('gen_valid_acc is {:.3f}'.format (best_acc))




    def select_opt(self, opt_method):
        if opt_method == "Adagrad" or opt_method == "adagrad":
            optimizer = optim.Adagrad (self.trainModel.parameters (), lr=self.config.lr, lr_decay=self.config.lr_decay,weight_decay=self.config.weight_decay)
        elif opt_method == "Adadelta" or opt_method == "adadelta":
            optimizer = optim.Adadelta (self.trainModel.parameters (), lr=self.config.lr)
        elif opt_method == "Adam" or opt_method == "adam":
            optimizer = optim.Adam (self.trainModel.parameters (), lr=self.config.lr)
        else:
            optimizer = optim.SGD (self.trainModel.parameters (), lr=self.config.lr)
        return optimizer

    def gen_step(self,data,train=True,n_sample=2):
        out = self.trainModel(data)
        prob = torch.exp (out)
        pos,neg = [i.squeeze(1) for i in prob.split(1,1)]

        sample_high_id = torch.multinomial(pos, n_sample, replacement=True)
        data = np.array(data)
        high_id = sample_high_id.cpu().data.numpy()
        high_data =data[high_id]

        high_id = set (high_id)
        total = set(np.arange(len(prob)))
        low_id = np.array(list(total-high_id))
        low_data = data[low_id]

        rewards = yield high_data,low_data
        if train:
            self.trainModel.zero_grad ()
            log_probs = pos
            reinforce_loss = -torch.sum (rewards *log_probs)
            reinforce_loss.backward(retain_graph = True)
            self.optimizer.step ()
            #self.trainModel.constraint()
        yield None
