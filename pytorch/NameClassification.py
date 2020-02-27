#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 22:19:52 2020
@author: zhaoyu
"""

import os
import unicodedata
import string
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def unicodeToAscii(s, all_letters):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def get_max_len(data):
    max_length = 0
    for item in raw_data: max_length = max(max_length, len(item[0]))
    return max_length


def load_data(file_paths, lang_dict, all_letters):
    name_type_pairs = []
    for file_path in file_paths:
        lang = file_path.split('.')[0].split('\\')[-1]
        lang_type = lang_dict[lang]
        lines = open(file_path, encoding='utf-8').read().strip().split('\n')
        name_type_pairs += [(unicodeToAscii(line, all_letters), lang_type) for line in lines]

    return name_type_pairs


def get_batch(data, idx_list, batch_size, head):
    # whether to add fixed length slicing
    length = len(idx_list)
    indices = idx_list[np.arange(head, head + batch_size) if head + batch_size <= length else \
        np.concatenate((np.arange(head, length), np.arange(head + batch_size - length)))]
    batch = [item[indices] for item in data]
    return batch, indices, head + batch_size


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        if hidden.shape[0] < input.shape[0]: hidden = hidden.expand([input.shape[0], self.hidden_size])
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def accuracy(probs, labels):
    _, predicts = torch.max(probs, dim=-1)
    correct = (predicts == labels).sum().item()
    return correct / labels.shape[0]


if __name__ == '__main__':
    data_path = './data/names/'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_abspath = os.path.abspath(data_path)
    files = [os.path.join(data_abspath, file) for file in os.listdir(data_path) if '.txt' in file]

    # extract label dict: lang_encode
    lang_encode = {}
    for i, file in enumerate(files):
        language_name = file.split('.')[0].split('\\')[-1]
        lang_encode[language_name] = i
    n_langs = len(lang_encode)

    # build vocab dict: letter2idx
    pad_char = '~'
    all_letters = string.ascii_letters + " .,;'" + pad_char
    n_letters = len(all_letters)
    letter2idx = {}
    for i, c in enumerate(all_letters): letter2idx[c] = i

    # extract raw_data
    raw_data = load_data(files, lang_encode, all_letters)
    n_samples = len(raw_data)
    max_name_length = get_max_len(raw_data)

    # data_features:3-D array; data_labels: 1-D array
    # auxiliary length for better slicing in Tensors: data_lengths
    data_features = torch.zeros([n_samples, max_name_length, n_letters])
    data_labels = torch.zeros([n_samples], dtype=torch.long)
    for i, item in enumerate(raw_data):
        data_labels[i] = item[1]
        for j, c in enumerate(item[0]):
            data_features[i, j, letter2idx[c]] = 1
        for j in range(len(item[0]), max_name_length):
            data_features[i, j, letter2idx[pad_char]] = 1
    data_lengths = torch.zeros([len(raw_data)], dtype=torch.int)
    for i, item in enumerate(raw_data): data_lengths[i] = len(raw_data[i][0])

    # define network
    n_hiddens = 128
    rnn = RNN(n_letters, n_hiddens, n_langs).to(device)

    # define loss function
    loss_fn = nn.CrossEntropyLoss()

    # define optimizer
    init_lr, momentum = 0.01, 0.9
    opt = optim.SGD(rnn.parameters(), init_lr, momentum)

    # define lr_schedule
    lr = optim.lr_scheduler.MultiStepLR(opt, [10, 15, 20], gamma=0.1)

    # define metrics
    softmax = nn.Softmax(dim=-1)

    # training with batches
    n_epoch, batch_size = 25, 8
    data_pack = (data_features, data_labels, data_lengths)

    for _ in range(n_epoch):
        print('epoch:', _)
        idx_list = np.arange(n_samples)
        random.shuffle(idx_list)

        head, cnt = 0, 0
        while head < n_samples:
            (feats, labels, lengths), indices, head = get_batch(data_pack, idx_list, batch_size, head)
            local_max_length = max(lengths)
            hidden = rnn.initHidden()
            rnn.zero_grad()
            for t in range(local_max_length):
                inputs, hiddens, labels_gpu = feats[:, t, :].to(device), hidden.to(device), labels.to(device)
                output, hidden = rnn(inputs, hiddens)

            acc = accuracy(softmax(output), labels_gpu)
            loss = loss_fn(output, labels_gpu)
            loss.backward()
            opt.step()
            if (cnt + 1) % 100 == 0:
                print(cnt + 1, ': acc=', acc, ', loss=', loss.item())
            cnt += 1
