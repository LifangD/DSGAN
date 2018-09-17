# -*- coding: utf-8 -*-

import torch
import time


class BasicModule(torch.nn.Module):
    '''
    封装了nn.Module,主要是提供了save和load两个方法
    '''

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name=str(type(self))  # model name

    def load(self, name):
        '''
        可加载指定路径的模型
        '''
        prefix = 'checkpoints/'
        name = prefix + name
        self.load_state_dict(torch.load(name))

    def save(self,name):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        '''
        prefix = 'checkpoints/'
        name = prefix + name
        torch.save(self.state_dict(), name)
        return name

