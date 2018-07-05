# coding: utf-8

import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr
import random


#这里的filename是最原始的tracefile，该函数返回训练，验证，和测试集, cont_lab = [[a1, a0, c0, d, 0],...], contents = [[a1,a0,c0,d]...], label = [0...]
# 返回分割好的数据集列表，句子和标签分开
def read_raw_file(filename, window_length=5):
    cont_lab = [] 
    train_contents, train_labels = [], []
    dev_contents, dev_labels = [], []
    test_contents, test_labels = [], []
    #把原始数据按照window_length 转换成 句子的格式
    with open(filename) as f:
        lines = f.readlines()
        for num in range(len(lines)-window_length+1):
            sentence = []
            for items in lines[num:num+(window_length-1)]:
                words = items.strip().split(',')
                word = words[0] + words[1]
                sentence.append(word)
            sentence = sentence + lines[num+window_length-1].strip().split(',')
            cont_lab.append(sentence)
    #按照8:1:1划分训练集，验证集，测试集（事先shuffle）
    random.shuffle(cont_lab)
    num_lines = len(cont_lab)
    train_num = int(0.8 * num_lines)
    dev_num = int(0.1 * num_lines)
    test_num = num_lines -train_num - dev_num
    train_list = cont_lab[0:train_num+1]
    dev_list = cont_lab[train_num+1: train_num + dev_num + 1]
    test_list = cont_lab[train_num + dev_num + 1: num_lines + 1]
    #把label分离出来
    train_contents = [li[0:-1] for li in train_list]
    train_labels = [li[-1] for li in train_list]
    dev_contents = [li[0:-1] for li in dev_list]
    dev_labels = [li[-1] for li in dev_list]   
    test_contents = [li[0:-1] for li in test_list]
    test_labels = [li[-1] for li in test_list]
    return train_contents, dev_contents, test_contents,train_labels, dev_labels, test_labels
    
    
def read_file(filename):
    """读取文件数据, contents = [[],[]...]"""
    contents, labels = [], []
    with open(filename,'r') as f:
        for line in f:
            content, label = line.strip().split('\t')
            contents.append(content.strip().split())
            labels.append(label)
    return contents, labels
    

# def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    # """根据训练集构建词汇表，存储"""
    # data_train, _ = read_file(train_dir)
    # data = []
    # for item in data_train:
        # data.extend(item)   
    # counter = Counter(data)
    # count_pairs = counter.most_common(vocab_size - 1)
    # words, _ = list(zip(*count_pairs))
    # open(vocab_dir, 'w').write('\n'.join(words) + '\n')

def build_vocab(filename, vocab_dir):
    """根据原始trace文件构建词汇表，存储"""
    words = set()
    with open(filename,'r') as f:
        for line in f:
            trace = line.strip().split(',')
            word = trace[0] + trace[1]
            words.add(word)
    open(vocab_dir, 'w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    with open(vocab_dir, 'r') as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories = ['0', '1']
    categories = [x for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)
    
    
def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示, data_id = [[],[]...]"""
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
        
    #使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    # data_id = np.array(data_id)
    # label_id = np.array(label_id)
    # return data_id, label_id
    return x_pad, y_pad

    
def process_all_file(filename, word_to_id, cat_to_id, max_length=600, window_length=5):
    train_contents, dev_contents, test_contents,train_labels, dev_labels, test_labels = read_raw_file(filename, window_length)
    train_data_id, train_label_id = [], []
    dev_data_id, dev_label_id = [], []
    test_data_id, test_label_id = [], []
    for i in range(len(train_contents)):
        train_data_id.append([word_to_id[x] for x in train_contents[i] if x in word_to_id])
        train_label_id.append(cat_to_id[train_labels[i]])
    for i in range(len(dev_contents)):
        dev_data_id.append([word_to_id[x] for x in dev_contents[i] if x in word_to_id])
        dev_label_id.append(cat_to_id[dev_labels[i]])
    for i in range(len(test_contents)):
        test_data_id.append([word_to_id[x] for x in test_contents[i] if x in word_to_id])
        test_label_id.append(cat_to_id[test_labels[i]])
        
    train_x_pad = kr.preprocessing.sequence.pad_sequences(train_data_id, max_length)
    train_y_pad = kr.utils.to_categorical(train_label_id, num_classes=len(cat_to_id))   
    dev_x_pad = kr.preprocessing.sequence.pad_sequences(dev_data_id, max_length)
    dev_y_pad = kr.utils.to_categorical(dev_label_id, num_classes=len(cat_to_id))
    test_x_pad = kr.preprocessing.sequence.pad_sequences(test_data_id, max_length)
    test_y_pad = kr.utils.to_categorical(test_label_id, num_classes=len(cat_to_id))
    
    return train_x_pad, dev_x_pad, test_x_pad, train_y_pad, dev_y_pad, test_y_pad
    
    

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
