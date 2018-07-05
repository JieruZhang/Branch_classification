window_size = 15
trace_di = 'SHORT_SERVER-100.bt9.trace'
trace_new_di = "E:/traces/SHORT_SERVER-100/15/trace_file.txt"
dev_di = "E:/traces/SHORT_SERVER-100/15/trace_val.txt"
test_di = "E:/traces/SHORT_SERVER-100/15/trace_test.txt"

def trace2sentenceLabel(filename, window_length):
    #trace_file = open("E:/traces/SHORT_SERVER-101/20/trace_file.txt", 'w')
    trace_file = open(trace_new_di, 'w')
    with open(filename) as f:
        lines = f.readlines()
        for num in range(len(lines)-window_length+1):
            for items in lines[num:num+(window_length-1)]:
                words = items.strip().split(',')
                word = words[0] + words[1]
                trace_file.write(word + ' ')
            words = lines[num+window_length-1].strip().split(',')
            trace_file.write(words[0] + '\t' + words[1])
            trace_file.write('\n')
    trace_file.close()
    
trace2sentenceLabel(trace_di, window_size)

import random

def split_dataset2(filename, percent):
    train_file = open(train_di,'w')
    dev_file = open(dev_di, 'w')
    test_file = open(test_di, 'w')
    with open(filename) as f:
        lines = f.readlines()
        random.shuffle(lines)
        num_lines = int(percent * len(lines))
        train_num = int(0.8 * num_lines)
        dev_num = int(0.1 * num_lines)
        test_num = num_lines -train_num - dev_num
        for line in lines[0:train_num+1]:
            train_file.write(line)
        for line in lines[train_num+1: train_num + dev_num + 1]:
            dev_file.write(line)
        for line in lines[train_num + dev_num + 1: num_lines + 1]:
            test_file.write(line)
    train_file.close()
    dev_file.close()
    test_file.close()
    
split_dataset2(trace_new_di, 1)