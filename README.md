### 环境：
Pytorch 0.3   
Python 2.7

### 过程：
1. 数据预处理
    - 初始数据：origin_data
        利用别人的代码抽取pb文件数据转成tsv,see:
        https://github.com/abarthakur/clustering_riedel_dataset  
        (抽出来的tsv 中间的空格需要替换)
        https://github.com/knightBoy/riedel_dataset  
        (如果是python3上,读取时改成二进制读取/写入，因为Unicodecsv里会用到decode() )
    - txt转npy: pre_process.py
2. gen_pretrain: 用P,NG预训练生成器
3. dis_pretrain: 用P，ND预训练判别器
4. gan_train:
    - 每一轮判别器载入初始参数、只用P数据进行训练
    - 生成器采样生成高置信度的样本，标记为0，低置信度的样本，标记为1，送入判别器
    - 判别器判断正确率，当正确率不再下降时，保存此时的生成器（认为生成器此时的Probability已然合理）

### 细节&参数
1. 句子特征提取，论文中使用Simple CNN
(Relation Classification via Convolutional Deep Neural Network)
(Event detection and domain adaptation with convolutional neural networks)
2. 一个最大化使用数据的策略，如果sentence bag里所有句子被generator认为是negative,entity pair会被认为是negative的一类
3. 一些参数：
    - 关系数量：52+1
    - CNN Window cw, kernel size ck 3, 100
    - Word embedding de,|V| 50, 114042
    - Position embedding dp 5
    - Learning rate of G,D 1e-5, 1e-4
    - position [-30,30]
4. word embedding 来自于 Neural relation extraction with selective attention over instance

### 关于数据集
1. NYT dataset：
    - Freebase triples,New York Times Corpus,Standfor NER extract/pick the sentence through the NER.
    - heldout(包括train和test) kb_manual人工标注的数据
2. 论文中未提到ND和NG是如何划分的,我的想法：
P：heldout中的trainPositive和kb manual中的trainPostive (Positive个数相比Negative要少很多)
    - Ng: heldout中的trainNegative
    - Nd: kbmanual中的trainNegative 因为要求判别器一开始的准确率很高，所以采用人工标注的负例应该会更好一些
用训练好的生成器过滤P 生成true positive samples，接着用一般关系模型训练relation extractor，
最后再用 heldout中的testNegative 和testPositive测试用不同正例训练出来的extractor的好坏。


### 存在的疑问：
1. 两种reWard,最终怎么结合？还是只用到了T的reward?
2. performance evaluation:
    - 用ND中的三种关系展示Discriminator在gan_train时在bag中循环时性能的下降，因为ND中的例子没有参与gan的训练，
      （感觉，在评价时，仍通过G来挑选，挑到更高质量的N 样本，标记为1，这样Discriminator的正确率就下降了)
    - 对最终generator的性能好坏用P的F1-score来衡量。这里的测试数据是预先筛选出来的？
3. DSGAN 也同时解决了当所有例子都是负例的情况，是指在用别的模型之前，先用DSGAN过滤一遍吗？
4. 论文中提到NG和ND不一样，要怎么划分？（划分比例）

## 参考：
[1] Qin P, Xu W, Wang W Y. DSGAN: Generative Adversarial Training for Distant Supervision Relation Extraction[J]. arXiv preprint arXiv:1805.09929, 2018.   
[2] https://github.com/abarthakur/clustering_riedel_dataset  
[3] https://github.com/ShomyLiu/pytorch-relation-extraction  
[4] https://github.com/cai-lw/KBGAN  
