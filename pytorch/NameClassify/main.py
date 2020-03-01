from network import RNNNaive, RNN
from learner import NameLearner
import string

BATCH_SIZE = 1
INIT_LR = 1e-3
MOMENTUM = 0.9
L2_REG = 1e-5

if __name__ == '__main__':
    #data_path = '../data/names/'
    data_path = '/home/zhaoyu/Datasets/NLPBasics/names/'

    #rnn = RNNNaive(name_dataset.n_letters, n_hiddens, name_dataset.n_labels)
    
    # define learner
    learner = NameLearner(data_path, RNN)
    
    learner.train()
    learner.test()
