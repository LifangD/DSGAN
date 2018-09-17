import torch
from torch import  nn,optim
from torch.autograd import Variable
from models.PCNN import PCNN
from  utils.data_utils import acc_metric,predict
import numpy as np


class Discriminator(object):
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
                    print ('dis_valid_acc is {:.3f}'.format (best_acc))


    def predict(self,model, test_data):
        model.eval ()
        test_loss = 0
        true_y =[]
        pred_y = []
        for i, data in enumerate (test_data):
            x, y = data
            true_y.extend(y)
            out = self.trainModel(x)
            _,index = map (lambda x: x.data.cpu ().numpy (), torch.max (out, 1))
            pred_y.extend(index) #
        return true_y, pred_y

    def select_opt(self, opt_method):
        if opt_method == "Adagrad" or opt_method == "adagrad":
            optimizer = optim.Adagrad (self.trainModel.parameters (), lr=self.config.lr, lr_decay=self.config.lr_decay,
                                       weight_decay=self.config.weight_decay)
        elif opt_method == "Adadelta" or opt_method == "adadelta":
            optimizer = optim.Adadelta (self.trainModel.parameters (), lr=self.config.lr)
        elif opt_method == "Adam" or opt_method == "adam":
            optimizer = optim.Adam (self.trainModel.parameters (), lr=self.config.lr)
        else:
            optimizer = optim.SGD (self.trainModel.parameters (), lr=self.config.lr)
        return optimizer

    def dis_step(self,high_data,low_data,train=True):
        data = np.vstack((high_data,low_data))
        label = [0]*len(high_data)+[1]*len(low_data)
        criterion = nn.NLLLoss()
        if self.config.use_cuda:
            label = torch.LongTensor (label).cuda ()
        else:
            label = torch.LongTensor (label)
        out = self.trainModel(data)
        loss = criterion(out,Variable(label))

        prob_T = self.trainModel(high_data)[0]
        prob_T = torch.exp(prob_T)
        reward = -torch.sum(prob_T)


        if train:
            self.trainModel.zero_grad()
            losses = torch.sum(loss)
            losses.backward()
            self.optimizer.step()
            #self.trainModel.constraint()
        return losses, reward

