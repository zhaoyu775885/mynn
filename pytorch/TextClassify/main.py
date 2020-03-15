# -*- coding: utf-8 -*-

from learner import Learner
from model import RNN
        
if __name__ == '__main__':
    data_path = '/home/zhaoyu/Datasets/NLPBasics/sentiment'
    lrner = Learner(data_path, RNN)
    lrner.train()
