# -*- coding: utf-8 -*-

from learner import Learner
from model import RNN, Transformer
        
if __name__ == '__main__':
    data_path = '/home/zhaoyu/Datasets/NLPBasics/Classification/SST-2'
#    lrner = Learner(data_path, RNN)
    lrner = Learner(data_path, Transformer)
    lrner.train()
