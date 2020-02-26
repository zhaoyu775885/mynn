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

def load_data(file_paths, lang_dict, all_letters):
    name_type_pairs = []
    
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
        )    
    
    for file_path in file_paths:
        lang = file_path.split('.')[0].split('/')[-1]
        lang_type = lang_dict[lang]
        lines = open(file_path, encoding='utf-8').read().strip().split('\n')
        name_type_pairs += [(unicodeToAscii(line), lang_type) for line in lines]
            
    return name_type_pairs

def get_batch(data, idx_list, batch_size, head):
    if head + batch_size <= len(idx_list):
        batch = data[idx_list[head:head+batch_size]]
    else:
        batch = np.array(data[idx_list[head:]].tolist() + data[idx_list[:head+batch_size-length]].tolist())
    return batch, head+batch_size

if __name__ == '__main__':
    data_path = '/home/zhaoyu/Datasets/NLPBasics/names/'
    
    data_abspath = os.path.abspath(data_path)
    files = [os.path.join(data_abspath, file) for file in os.listdir(data_path) if '.txt' in file]
    
    lang_encode = {}
    for i, file in enumerate(files):
        language_name = file.split('.')[0].split('/')[-1]
        lang_encode[language_name] = i
    
    pad_char = '~'
    all_letters = string.ascii_letters + " .,;'" + pad_char
    n_letters = len(all_letters)
    raw_data = load_data(files, lang_encode, all_letters)
    raw_data = np.array(raw_data)

    n_epoch = 1
    batch_size = 32
    length = len(raw_data)
    for _ in range(n_epoch):
        idx_list = np.arange(length)
        random.shuffle(idx_list)
        head = 0
        while head < length:
            batch, head = get_batch(raw_data, idx_list, batch_size, head)
