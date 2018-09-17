# encoding:utf-8
from models.Generator import Generator
from models.Discriminator import Discriminator
from torch.utils.data import DataLoader
from config import gen_config,dis_config,gan_config
from utils.data_utils import get_pn_data,get_pos_data,get_neg_data
import os,torch
from utils.data_utils import acc_metric,predict
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()

def collate_fn(batch):
    data, label = zip(*batch)
    return data, label

def gen_pretrain():
    print("start pre-training generator...")
    conf = gen_config()
    train_data,test_data = get_pn_data('data/gen_data')
    train_loader = DataLoader(train_data, conf.batch_size, shuffle=True, num_workers=conf.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader (train_data, conf.batch_size, shuffle=True, num_workers=conf.num_workers,collate_fn=collate_fn)
    Gen = Generator(conf)
    Gen.pretrain(train_loader,test_loader)

def dis_pretrain():
    print("start pre-training discriminator...")
    conf = dis_config ()
    train_data, test_data = get_pn_data ('data/dis_data')
    train_loader = DataLoader (train_data, conf.batch_size, shuffle=True, num_workers=conf.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader (train_data, conf.batch_size, shuffle=True, num_workers=conf.num_workers,collate_fn=collate_fn)
    Dis =  Discriminator(conf)
    Dis.pretrain (train_loader, test_loader)


def gan_train():
    print("start gan training...")
    conf = gan_config()
    gen = Generator(conf)
    dis = Discriminator(conf)
    gen.trainModel.load('generator.pkl')
    train_data = get_pos_data('data/gan_data')
    test_data = get_neg_data('data/dis_data')
    train_loader = DataLoader(train_data, conf.batch_size, shuffle=True, num_workers=conf.num_workers,collate_fn=collate_fn,drop_last=True)
    test_loader = DataLoader(test_data, conf.batch_size, shuffle=True, num_workers=conf.num_workers,collate_fn=collate_fn,drop_last=True)

    avg_reward = 0
    for epoch in range(conf.n_epochs):
        dis.trainModel.load('discriminator.pkl')
        epoch_loss = 0
        epoch_reward = 0
        for i,batch_data in enumerate(train_loader):
            data,label = batch_data
            gen_step = gen.gen_step(data)
            high,low = next(gen_step)
            losses,reward = dis.dis_step(high,low)
            reward = reward-avg_reward
            epoch_loss+=losses.data[0]
            epoch_reward+=reward
            gen_step.send(reward)
        avg_reward = epoch_reward/conf.batch_size

        print('Epoch{}/{}, Train_Loss={:.3f}'.format(epoch + 1, conf.n_epochs,  epoch_loss/ conf.batch_size))
        worst_acc = 1
        if epoch%conf.epoch_per_test ==0:
                true_y,pred_y = predict(dis.trainModel,test_loader)
                eval_acc = acc_metric(true_y,pred_y)
                if  worst_acc > eval_acc:
                    worst_acc = eval_acc
                    gen.trainModel.save(conf.model_name)
                    print ('gan_valid_acc is {:.3f}'.format (worst_acc))






if __name__ =="__main__":

    gen_pretrain()
    dis_pretrain()
    gan_train()
