# coding: utf-8

import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr
import random



#这里的filename是最原始的tracefile，该函数返回训练，验证，和测试集, cont_lab = [[a1, a0, c0, d, 0],...], contents = [[a1,a0,c0,d]...], label = [0...]
# 返回分割好的数据集列表，句子和标签分开
def read_raw_file(train_file, eval_file, train_ratio = 1, window_length=5):
    train_contents, train_labels = [], []
    eval_contents, eval_labels = [], []
    dev_contents, dev_labels = [], []
    test_contents, test_labels = [], []
    #把原始数据按照window_length 转换成 句子的格式
    with open(train_file) as f:
        lines = f.readlines()
        lines = [item.strip().replace('\t','') for item in lines]
        for num in range(len(lines)-window_length+1):
            sentence = [item for item in lines[num:num+window_length]]
            train_labels.append(sentence[-1][-1])
            sentence[-1] = sentence[-1][:-1]
            train_contents.append(sentence)
            
    with open(eval_file) as f:
        lines = f.readlines()
        lines = [item.strip().replace('\t','') for item in lines]
        for num in range(len(lines)-window_length+1):
            sentence = [item for item in lines[num:num+window_length]]
            eval_labels.append(sentence[-1][-1])
            sentence[-1] = sentence[-1][:-1]
            eval_contents.append(sentence)
            
        
    train_num = int(train_ratio * len(train_contents))
    dev_num = int(0.5 * len(eval_contents))
    train_contents = train_contents[0:train_num]
    dev_contents = eval_contents[0: dev_num]
    test_contents = eval_contents[dev_num :]
    train_labels = train_labels[0:train_num]
    dev_labels = eval_labels[0: dev_num]
    test_labels = eval_labels[dev_num:]
    
    return train_contents, dev_contents, test_contents,train_labels, dev_labels, test_labels


def build_vocab(filename, vocab_dir):
    """根据原始trace文件构建词汇表，存储"""
    words = set()
    with open(filename,'r') as f:
        for line in f:
            trace = line.strip().split()
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

    
def process_all_file(train_file, eval_file, train_ratio,word_to_id, cat_to_id, max_length=600, window_length=5):
    train_contents, dev_contents, test_contents,train_labels, dev_labels, test_labels = read_raw_file(train_file, eval_file, train_ratio, window_length)
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
