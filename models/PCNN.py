# -*- coding: utf-8 -*-

from models.BasicModule import BasicModule
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PCNN(BasicModule):
    def __init__(self, conf):
        super(PCNN, self).__init__()
        self.conf = conf
        self.word_embs = nn.Embedding(self.conf.vocab_size, self.conf.word_dim) 
        self.pos1_embs = nn.Embedding(self.conf.pos_size, self.conf.pos_dim) 
        self.pos2_embs = nn.Embedding(self.conf.pos_size, self.conf.pos_dim)
        feature_dim = self.conf.word_dim + self.conf.pos_dim * 2

        # for more filter size, and right now there's only one.
        self.convs = nn.ModuleList([nn.Conv2d(1, self.conf.filters_num, (k, feature_dim), padding=(int(k / 2), 0)) for k in self.conf.filters])

        all_filter_num = self.conf.filters_num * len(self.conf.filters)

        if self.conf.use_pcnn:
            all_filter_num = all_filter_num * 3 # 三段contact

        self.linear = nn.Linear(all_filter_num, self.conf.rel_num)
        self.dropout = nn.Dropout(self.conf.drop_out)
        self.relu = nn.ReLU(True)
        self.classify = nn.Linear(self.conf.rel_num,self.conf.type_num)

        self.init_model_weight()
        self.init_word_emb()

    def init_model_weight(self):
        '''
        use xavier to init
        '''
        for conv in self.convs:
            nn.init.xavier_uniform(conv.weight)
            nn.init.constant(conv.bias, 0.0)

        nn.init.xavier_uniform(self.linear.weight)
        nn.init.constant(self.linear.bias, 0.0)

    def init_word_emb(self):

        def p_2norm(path):
            v = torch.from_numpy(np.load(path))
            if self.conf.norm_emb:
                v = torch.div(v, v.norm(2, 1).unsqueeze(1))
                v[v != v] = 0.0
            return v

        w2v = p_2norm(self.conf.w2v_path)
        p1_2v = p_2norm(self.conf.p1_2v_path)
        p2_2v = p_2norm(self.conf.p2_2v_path)

        if self.conf.use_cuda:
            self.word_embs.weight.data.copy_(w2v.cuda())
            self.pos1_embs.weight.data.copy_(p1_2v.cuda()) 
            self.pos2_embs.weight.data.copy_(p2_2v.cuda())
        else:
            self.pos1_embs.weight.data.copy_(p1_2v)
            self.pos2_embs.weight.data.copy_(p2_2v)
            self.word_embs.weight.data.copy_(w2v)



    def piece_max_pooling(self, x, insPool):
        '''
        old version piecewise
        '''
        split_batch_x = torch.split(x, 1, 0)
        split_pool = torch.split(insPool, 1, 0)
        batch_res = []

        for i in range(len(split_pool)):
            ins = split_batch_x[i].squeeze()  # all_filter_num * max_len
            pool = split_pool[i].squeeze().data    # 2
            seg_1 = ins[:, :pool[0]].max(1)[0].unsqueeze(1)          # all_filter_num * 1
            seg_2 = ins[:, pool[0]: pool[1]].max(1)[0].unsqueeze(1)  # all_filter_num * 1
            seg_3 = ins[:, pool[1]:].max(1)[0].unsqueeze(1)
            piece_max_pool = torch.cat([seg_1, seg_2, seg_3], 1).view(1, -1)    # 1 * 3all_filter_num
            batch_res.append(piece_max_pool)

        out = torch.cat(batch_res, 0)
        assert out.size(1) == 3 * self.conf.filters_num
        return out

    def forward(self, x):
        insX, insPFs, insPool = zip(*x)
        xx= [insX,insPFs,insPool]
        if self.conf.use_cuda:
            xx = map (lambda x: Variable (torch.LongTensor (x).cuda()), xx)
        else:
            xx = map (lambda x: Variable (torch.LongTensor (x)), xx)
        insX,insPFs,insPool = xx
        insPF1,insPF2 = [i.squeeze(1) for i in torch.split(insPFs,1,1)]
        word_emb = self.word_embs(insX)
        pf1_emb = self.pos1_embs(insPF1)
        pf2_emb = self.pos2_embs(insPF2)
        x = torch.cat([word_emb, pf1_emb, pf2_emb], 2)
        x = x.unsqueeze(1)
        x = self.dropout(x)
        x = [F.tanh(conv(x)).squeeze(3) for conv in self.convs] #数据复制了多次 为下面的多个卷积核做准备
        if self.conf.use_pcnn:
            x = [self.piece_max_pooling(i, insPool) for i in x] # 多个卷积核
        else:
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.relu(x)
        out =self.classify(x)
        out = F.log_softmax(out,1)
        return out

    def constraint(self):
        for param in self.parameters():
            param.data.renorm_(2, 0, 1)

