import torch
import torch.nn as nn
import torch.optim as optim
import string
from name_data import NameDataset
from network import RNNNaive, RNN
from learner import NameLearner
import string

BATCH_SIZE = 1
INIT_LR = 1e-3
MOMENTUM = 0.9
L2_REG = 1e-5

if __name__ == '__main__':
    data_path = '../data/names/'
    #data_path = '/home/zhaoyu/Datasets/NLPBasics/names/'
    
    # define dataset
    all_letters = string.ascii_letters + " .,;'"
    name_dataset = NameDataset(data_path, '/', all_letters)
    
    # define network
    n_hiddens = 128
    print(name_dataset.n_letters, name_dataset.n_labels)
    #rnn = RNNNaive(name_dataset.n_letters, n_hiddens, name_dataset.n_labels)
    rnn = RNN(name_dataset.n_letters, n_hiddens, name_dataset.n_labels)
    
    # define learner
    learner = NameLearner(name_dataset, rnn)
    
    learner.train()
