# -*- coding: utf-8 -*-
# @Time : 2019/8/15 上午9:33
# @Author : Lihui Lin
# @FileName: utils.py

import numpy as np
import random
import math

'''
Created by Lihui Lin. School of Data and Computer Science, Sun Yat-sen University.
This module provides a set of functions for raw text processing and data generating.
Convert words to ids, generate data sets and batches for BOW and sequence modeling.
'''


def data_set(data_url):
    """process data input."""
    data = []
    word_count = []
    fin = open(data_url)
    while True:
        line = fin.readline()
        if not line:
            break
        # id_freqs = line.split()[1:]
        id_freqs = line.split()
        doc = {}
        count = 0
        for id_freq in id_freqs:
            items = id_freq.split(':')
            # python starts from 0
            doc[int(items[0])-1] = int(items[1])
            # doc[int(items[0])] = int(items[1])
            count += int(items[1])
        if count > 0:
            data.append(doc)
            word_count.append(count)
    fin.close()
    return data, word_count


def create_batches(data_size, batch_size, shuffle=True):
    """create index by batches."""
    batches = []
    ids = list(range(data_size))
    if shuffle:
        random.shuffle(ids)
    for i in range(math.floor(data_size / batch_size)):
        start = i * batch_size
        end = (i + 1) * batch_size
        batches.append(list(ids[start:end]))
    # the batch of which the length is less than batch_size
    rest = data_size % batch_size
    if rest > 0:
        batches.append(list(ids[-rest:]) + [-1] * (batch_size - rest))  # -1 as padding
    return batches


def fetch_data(data, count, idx_batch, vocab_size):
    """fetch input data by batch."""
    batch_size = len(idx_batch)
    data_batch = np.zeros((batch_size, vocab_size))
    count_batch = []
    mask = np.zeros(batch_size)
    # indices = []
    # values = []
    for i, doc_id in enumerate(idx_batch):
        if doc_id != -1:
            for word_id, freq in data[doc_id].items():
                data_batch[i, word_id] = freq
            count_batch.append(count[doc_id])
            mask[i] = 1.0
        else:
            count_batch.append(0)
    return data_batch, count_batch, mask


def variable_parser(var_list, prefix):
    """return a subset of the all_variables by prefix."""
    ret_list = []
    for var in var_list:
        varname = var.name
        varprefix = varname.split('/')[0]
        if varprefix == prefix:
            ret_list.append(var)
    return ret_list

def get_vocab(data_url):
  """process data input."""
  vocab = []
  fin = open(data_url, encoding='utf-8')
  while True:
    line = fin.readline()
    if not line:
      break
    word_freqs = line.strip().split()
    vocab.append(word_freqs[0])
  fin.close()
  return vocab
