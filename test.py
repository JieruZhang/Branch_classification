# coding: utf-8

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

#from model import TRNNConfig, TextRNN
from attention_model import TRNNConfig, TextRNN
from data_loader import read_vocab, read_category, batch_iter, process_all_file, build_vocab
import random

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len
    
    
def test():
    # print("Loading test data...")
    start_time = time.time()
    # x_test, y_test = process_file(test_dir, word_to_id, cat_to_id,config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    
    print("Predict results for random subset of the test set:")
    nums = list(range(len(y_test)))
    random.shuffle(nums)
    for num in nums[0:10]:
        print('label: {}'.format(y_test_cls[num]))
        print('Predict: {}'.format(y_pred_cls[num]))
    
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    # if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        # raise ValueError("""usage: python run_rnn.py [train / test]""")
    if len(sys.argv) != 6:
        raise ValueError("Need arguments: data_dir, save_dir, window_size")
    
    train_data_dir = sys.argv[1]
    eval_data_dir = sys.argv[2]
    base_dir = sys.argv[3]
    window_size = sys.argv[4]
    train_ratio = sys.argv[5]
    vocab_dir = os.path.join(base_dir, 'vocab.txt')
    save_dir = os.path.join(base_dir, train_ratio + '/checkpoints/textrnn')
    save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
    window_size = int(window_size)
    train_ratio = float(train_ratio)
    
    print('Configuring RNN model...')
    print('Building vocab if not exists.')
    start_time_vocab = time.time()
    config = TRNNConfig()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_data_dir, vocab_dir)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextRNN(config)
    time_dif_vocab = get_time_dif(start_time_vocab)
    print("Time usage:", time_dif_vocab)
    
    
    #读取原始数据并转换成三个集合
    print("Processing and loading training and validation data...")
    start_time = time.time()
    x_train, x_val, x_test, y_train, y_val, y_test = process_all_file(train_data_dir, eval_data_dir, train_ratio,word_to_id,cat_to_id, config.seq_length, window_size)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    
    # print('==========Training==========')
    # start_time_train = time.time()
    # train()
    # time_dif_train = get_time_dif(start_time_train)
    # print('Training time usage: ', time_dif_train)
    print('==========Testing==========')
    start_time_test = time.time()
    test()
    time_dif_test = get_time_dif(start_time_test)
    print('Test time usage: ', time_dif_test)
    
    # if sys.argv[1] == 'train':
        # train()
    # else:
        # test()