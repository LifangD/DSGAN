ps: 仅供参考，没有check结果
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
3. dis_pretrain: 用P,ND预训练判别器
4. gan_train:
    - 每一轮判别器载入初始参数、只用P数据进行训练
    - 生成器采样生成高置信度的样本，标记为0，低置信度的样本，标记为1，送入判别器
    - 判别器判断正确率，当正确率不再下降时，保存此时的生成器（认为生成器此时的Probability已然合理）

### 细节&参数
1. 句子特征提取，论文中使用Simple CNN
(Relation Classification via Convolutional Deep Neural Network)
(Event detection and domain adaptation with convolutional neural networks)
2. 一些参数：
    - 关系数量：52+1
    - CNN Window cw, kernel size ck 3, 100
    - Word embedding de,|V| 50, 114042
    - Position embedding dp 5
    - Learning rate of G,D 1e-5, 1e-4
    - position [-30,30]





### 参考：
[1] Qin P, Xu W, Wang W Y. DSGAN: Generative Adversarial Training for Distant Supervision Relation Extraction[J]. arXiv preprint arXiv:1805.09929, 2018.   
[2] https://github.com/abarthakur/clustering_riedel_dataset  
[3] https://github.com/ShomyLiu/pytorch-relation-extraction  
[4] https://github.com/cai-lw/KBGAN  



