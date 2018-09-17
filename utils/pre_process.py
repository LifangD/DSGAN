# encoding:utf-8
import numpy as np
import os

class Data_Prepare(object):
    def __init__(self,root_path,max_len=80, limit=50, pos_dim=5, pad=1):
        self.max_len = max_len
        self.limit = limit
        self.root_path = root_path
        self.pos_dim = pos_dim
        self.pad = pad

        self.pos_path = os.path.join (root_path, 'pos.txt')
        self.neg_path = os.path.join (root_path, 'neg.txt')

    '''
    word and offset embedding 
    '''
    def get_initial_embedding(self,dic_path):
        self.w2v_path = os.path.join (dic_path, 'vector.txt')
        self.word_path = os.path.join(dic_path,'dict.txt')
        self.w2v, self.word2id, self.id2word = self.load_w2v ()
        self.p1_2v, self.p2_2v = self.load_p2v ()
        np.save (os.path.join (dic_path, 'w2v.npy'), self.w2v)
        np.save (os.path.join (dic_path, 'p1_2v.npy'), self.p1_2v)
        np.save (os.path.join (dic_path, 'p2_2v.npy'), self.p2_2v)

    def load_p2v(self):
        pos1_vec = [np.zeros (self.pos_dim)]
        pos1_vec.extend ([np.random.uniform (low=-1.0, high=1.0, size=self.pos_dim) for _ in range (self.limit * 2 + 1)])
        pos2_vec = [np.zeros (self.pos_dim)]
        pos2_vec.extend ([np.random.uniform (low=-1.0, high=1.0, size=self.pos_dim) for _ in range (self.limit * 2 + 1)])
        return np.array (pos1_vec, dtype=np.float32), np.array (pos2_vec, dtype=np.float32)

    def load_w2v(self):
        wordlist = []
        vecs = []
        wordlist.append ('BLANK')
        wordlist.extend ([word.strip ('\n') for word in file(self.word_path)])
        for line in file(self.w2v_path):
            line = line.strip ('\n').split ()
            vec = map(float, line)
            vecs.append (vec)
       
        dim = len (vecs[0])
        vecs.insert (0, np.zeros (dim))
        wordlist.append ('UNK')
        vecs.append (np.random.uniform (low=-1.0, high=1.0, size=dim))
        word2id = {j: i for i, j in enumerate (wordlist)}
        id2word = {i: j for i, j in enumerate (wordlist)}
        return np.array(vecs, dtype=np.float32), word2id, id2word


    '''
    sentences features, including  ids, offsets of every word and the two entities' position
    '''

    def save_sen_feature(self):
        pos= self.extract_sen_feature(self.pos_path)
        neg= self.extract_sen_feature(self.neg_path)
        np.save(os.path.join(self.root_path,'pos_feature.npy'),pos)
        np.save (os.path.join(self.root_path,'neg_feature.npy'), neg)


    def extract_sen_feature(self,path):
        bags,_ = self.parse_sen(path)
        sen_pos = []
        for bag in bags:
            feature = zip(bag[2],bag[3],bag[4])
            sen_pos +=feature
        return np.array(sen_pos,dtype=np.int)

    def parse_sen(self, path):
        all_sens =[]
        all_labels =[]
        f = file(path)
        while 1:
            line = f.readline()
            if not line:
                break
            entities = map(int, line.split(' '))
            line = f.readline()
            bagLabel = line.split(' ')
            rel = map(int, bagLabel[0:-1])
            num = int(bagLabel[-1])
            positions = []
            sentences = []
            entitiesPos = []
            for i in range(0, num):
                sent = f.readline().split(' ')
                positions.append(map(int, sent[0:2]))
                epos = map(lambda x: int(x) + 1, sent[0:2])
                epos.sort()
                entitiesPos.append(epos)
                sentences.append(map(int, sent[2:-1]))
            bag = [entities, num, sentences, positions, entitiesPos]
            all_labels.append(rel)
            all_sens += [bag]
        f.close()
        bags_feature = self.get_sentence_feature(all_sens)
        return bags_feature, all_labels

    def get_sentence_feature(self, bags):
        update_bags = []
        for bag in bags:
            es, num, sens, pos, enPos = bag
            new_sen = []
            new_pos = []
            for idx, sen in enumerate(sens):
                new_pos.append(self.get_pos_feature(len(sen), pos[idx])) # get every word's offset
                new_sen.append(self.get_pad_sen(sen))
            update_bags.append([es, num, new_sen, new_pos, enPos])
        return update_bags

    def get_pad_sen(self, sen):
        '''
        padding the sentences
        '''
        sen.insert(0, self.word2id['BLANK'])
        if len(sen) < self.max_len + 2 * self.pad:
            sen += [self.word2id['BLANK']] * (self.max_len +2 * self.pad - len(sen))
        else:
            sen = sen[: self.max_len + 2 * self.pad]
        return sen

    def get_pos_feature(self, sen_len, ent_pos):
        '''
        clip the postion range:
        : -limit ~ limit => 0 ~ limit * 2+2
        : -51 => 1
        : -50 => 1
        : 50 => 101
        : >50: 102
        '''
        def padding(x):
            if x < 1:
                return 1
            if x > self.limit * 2 + 1:
                return self.limit * 2 + 1
            return x

        if sen_len < self.max_len:
            index = np.arange(sen_len)
        else:
            index = np.arange(self.max_len)

        pf1 = [0]
        pf2 = [0]
        pf1 += map(padding, index - ent_pos[0] + 2 + self.limit)
        pf2 += map(padding, index - ent_pos[1] + 2 + self.limit)

        if len(pf1) < self.max_len + 2 * self.pad:
            pf1 += [0] * (self.max_len + 2 * self.pad - len(pf1))
            pf2 += [0] * (self.max_len + 2 * self.pad - len(pf2))
        return [pf1, pf2]



if __name__ =="__main__":
    # DP = Data_Prepare(root_path = '../data/gen_data')
    # DP.get_initial_embedding(dic_path='../data/dic_data')
    # DP.save_sen_feature()

    data = np.load('../data/gen_data/pos_feature.npy')
    print type(data)







