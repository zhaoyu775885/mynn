import os
import string
import unicodedata
import numpy as np
import torch
import random

def unicodeToAscii(s, all_letters):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def get_batch(data, idx_list, batch_size, head):
    # whether to add fixed length slicing
    length = len(idx_list)
    indices = idx_list[np.arange(head,head+batch_size) if head+batch_size<=length else \
                       np.concatenate((np.arange(head,length), np.arange(head+batch_size-length)))]
    batch = [item[indices] for item in data]
    return batch, indices, head+batch_size

def accuracy(probs, labels):
    _, predicts = torch.max(probs, dim=-1)
    correct = (predicts == labels).sum().item()
    return correct / labels.shape[0]

class Vocab():
    def __init__(self, words):
        self.words = words
        self.size = len(self.words)
        self.dict = {}
        for i, word in enumerate(self.words): self.dict[word]=i

    def __getitem__(self, item):
        return self.words[item]

    def item2index(self, word):
        return self.dict[word]

class NameDataset():
    def __init__(self, data_dir, path_splitter, letters, pad_char='~'):
        self.data_dir = data_dir
        self.path_splitter = path_splitter
        self.pad = pad_char
        self.letters = letters + pad_char if pad_char not in letters else letters
        self.data_abspath = os.path.abspath(self.data_dir)
        self.files = [os.path.join(self.data_abspath, file) for file in \
                      os.listdir(self.data_abspath) if '.txt' in file]
        self.lang_vocab = self._build_lang_vocab()
        self.char_vocab = self._build_char_vocab()

        self.raw_data = self._read_raw_data()
        self.n_samples = len(self.raw_data)
        self.n_labels = self.lang_vocab.size
        self.n_letters = self.char_vocab.size
        self._gen_tensor_data()

        self.idx_list = None
        self.batch_size = None
        self.batch_pointer = None

    def _build_lang_vocab(self):
        return Vocab([file.split('.')[0].split(self.path_splitter)[-1] for file in self.files])

    def _build_char_vocab(self):
        return Vocab(self.letters)

    def _read_raw_data(self):
        name_label_pairs = []
        for file in self.files:
            lang = file.split('.')[0].split(self.path_splitter)[-1]
            lang_label = self.lang_vocab.item2index(lang)
            lines = open(file, encoding='utf-8').read().strip().split('\n')
            name_label_pairs += [(unicodeToAscii(line, self.letters), lang_label) for line in lines]
        return name_label_pairs

    def _gen_tensor_data(self):
        self.data_lengths = torch.zeros([self.n_samples], dtype=torch.long)
        for i, item in enumerate(self.raw_data): self.data_lengths[i] = len(item[0])
        self.max_name_length = max(self.data_lengths)
        self.data_features = torch.zeros([self.n_samples, self.max_name_length, self.n_letters])
        self.data_labels = torch.zeros([self.n_samples], dtype=torch.long)
        for i, item in enumerate(self.raw_data):
            self.data_labels[i] = item[1]
            for j, c in enumerate(item[0]):
                self.data_features[i, j, self.char_vocab.item2index(c)] = 1
            for j in range(len(item[0]), self.max_name_length):
                self.data_features[i, j, self.char_vocab.item2index(self.pad)] = 1

    def init_batch_loader(self, batch_size=32):
        self.batch_size = batch_size
        self.batch_pointer = 0
        self.idx_list = np.arange(self.n_samples)
        random.shuffle(self.idx_list)

    def __iter__(self):
        return self

    def __next__(self):
        data_pack = [self.data_features, self.data_labels, self.data_lengths]
        indices = self.idx_list[np.arange(self.batch_pointer, self.batch_pointer+self.batch_size) if
            self.batch_pointer+self.batch_size <= self.n_samples else np.concatenate(
            (np.arange(self.batch_pointer, self.n_samples), np.arange(self.batch_pointer+self.batch_size-self.n_samples)))]
        batch = [item[indices] for item in data_pack]
        self.batch_pointer += self.batch_size
        return batch

if __name__ == '__main__':
    # data_path = './data/names/'
    data_path = '/home/zhaoyu/Datasets/NLPBasics/names/'
    all_letters = string.ascii_letters + " .,;'"

    name_dataset = NameDataset(data_path, '/', all_letters)
    print(name_dataset.lang_vocab.dict)
    print(name_dataset.char_vocab.dict)
    print(name_dataset.raw_data)

    name_dataset.init_batch_loader(batch_size=1)
    i = 0
    for i, item in enumerate(name_dataset):
        if i<2:
            print(i, item)
        else:
            break