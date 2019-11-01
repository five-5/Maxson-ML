# -*- coding:utf8 -*-
import datetime
import json
import sys
import timeit
import random


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import *

START_TAG = "<START>"
STOP_TAG = "<STOP>"

## helper function
def str2num(s):
    # print(s)
    if str(s).count('.') > 0:
        result = float(s)
    else:
        result = int(s)
    return result

def loadFeature(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip().split("^^")
            vecpath = s[0].strip('[').strip(']').split(',')
            vecpath = [str2num(i) for i in vecpath]
            count = int(s[1])
            jsonpath = s[2]
            curTime = s[3]
            data.append((vecpath, count, jsonpath, curTime))
    return data

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class TrainModel:
    def __init__(self):
        self.configs = {}

    def get_configs(self):
        return self.configs

    def load_config(self, config_path='config.json'):
        self.configs = json.load(open(config_path, 'r'))

    def read_data(self, file_path):
        # 读取配置
        data_config = self.configs['data']
        window_size = data_config['windowSize']
        windows_num = data_config['windowsNum']
        separator = data_config['separator']

        # 加载保存好的数据
        seqs = {} # {jsonPath: [(feature, label),...]}
        data = []
        d1 = datetime.datetime.now()
        print("start read file")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip().split(separator)
                vecpath = s[0].strip('[').strip(']').split(',')
                vecpath = [str2num(i) for i in vecpath]
                label = int(s[1])
                jsonpath = s[2]
                train_data_line = (vecpath, label)
                if jsonpath not in seqs.keys():
                    seqs[jsonpath] = []
                seqs[jsonpath].append(train_data_line)
        print("read file finished.")

        for name in seqs.keys():
            seq = seqs[name]
            start = 0
            end = window_size + 1
            length = windows_num if len(seq) - window_size >= windows_num else len(seq) - window_size
            if length > 0:
                while length:
                    data.append(seq[start:end])
                    start = start + 1
                    end = end + 1
                    length = length - 1
            else:
                data.append(seq[:])
        print("construct seq finished.")
        # data len = path数*窗口数  (100*30,31,2) 100条path 30个窗口 31天数据
        # data : [[(2,1),(1,0)....(3,1)],
        #               [(1,0),(0,0)....(1,0)],
        #                   .....
        #               [(2,1),(1,0)....(3,1)]]
        random.shuffle(data)
        d2 = datetime.datetime.now()
        print("read data : ", (d2 - d1).total_seconds(), " s")
        return data

    def train(self):
        # 获取配置文件信息
        data_config = self.configs['data']
        train_feature_path = data_config['trainName']
        training_config = self.configs['training']

        # 获取数据集
        print("get dataset")
        train_data = self.read_data(train_feature_path)
        train_set = SeqDataLoader(train_data,
                                  self.get_configs())

        train_loader = DataLoader(train_set,
                                  batch_size=get_config_value(training_config, 'batchSize', 100),
                                  shuffle=get_config_value(training_config, 'shuffle', True))
        print("get dataset finished.")
        # 设置训练模型
        epoch = get_config_value(training_config, 'epoch', 5)
        model = MyModel(self.get_configs())
        lr = get_config_value(training_config, 'learningRate', 0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr= lr)

        # 开始训练
        print("begin training.")
        d1 = datetime.datetime.now()
        for ep in range(epoch):
            for x, label in train_loader:
                label = label.view(-1)
                optimizer.zero_grad()
                score, preds = model.forward(x)
                total = label.view(-1, 1).size(0)
                correct = (np.array(preds) == label.numpy()).sum()
                accuracy = correct / total
                loss = model.neg_log_likelihood(x, label)
                mif1score = f1_score(label, preds, average='micro')
                maf1score = f1_score(label, preds, average='macro')

                # Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                loss.backward()
                optimizer.step()
            print("[%f] loss = %f, acc = %f, mif1score = %f, maf1score = %f" % (score, loss, accuracy, mif1score, maf1score))
        d2 = datetime.datetime.now()
        print("训练时长: ", (d2 - d1).total_seconds(), " s")

        # 保存模型
        torch.save(model.state_dict(), self.configs['model']['saveName'])  # 只保存当前面模型的参数

class MyModel(nn.Module):
    def __init__(self, configs):
        super(MyModel, self).__init__()

        self.tag_to_ix = {"non-cache": 0, "cache": 1, START_TAG: 2, STOP_TAG: 3}
        self.tagset_size = configs['data']['tagsetSize']

        lstm_crf_config = configs['model']['lstm_crf']
        self.hidden_dim = get_config_value(lstm_crf_config, 'hiddenSize', 10)
        self.lstm = nn.LSTM(input_size=get_config_value(lstm_crf_config, 'inputSize', 1),
                            hidden_size=self.hidden_dim,
                            num_layers=get_config_value(lstm_crf_config, 'numLayers', 2),
                            batch_first=get_config_value(lstm_crf_config, 'batchFirst', True))

        # Maps the output of the LSTM into tag space.
        linear1_config = configs['model']['linear1']
        self.line1 = nn.Linear(in_features=get_config_value(linear1_config, 'inFeatures', 10),
                              out_features=self.tagset_size)


        # Matrix of transition parameters.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = -10000

    def _get_lstm_features(self, input):
        lstm_out, _ = self.lstm(input)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.line1(lstm_out)
        return out

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                # view():返回一个数据相同但是大小不同搞的tensor
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])

        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def forward(self, input):
        # Get the emission scores from the LSTM
        out = self._get_lstm_features(input)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(out)
        return score, tag_seq

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, input, labels):
        feats = self._get_lstm_features(input)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, labels)
        return forward_score - gold_score

class SeqDataLoader(Dataset):
    def __init__(self, data, configs):
        self.data = data
        self.window_size = configs['data']['windowSize']
        self.nonseq_feature_size = configs['data']['nonSeqFeatureSize']

    def __len__(self):
        return len(self.data)

    # 查找数据和标签
    def __getitem__(self, idx):
        seq = self.data[idx]
        x = torch.tensor([y[0] for y in seq[0:self.window_size]])
        x = x.view(-1, self.window_size * 2 + self.nonseq_feature_size)
        label = [y[1] for y in seq[0:self.window_size]]
        labels = torch.tensor(label, dtype = torch.long)
        return x, labels

def get_config_value(config, name, default):
    return config[name] if name in config else default

if __name__ == '__main__':
    # config_path = 'config.json'
    config_path = sys.argv[1]
    trainModel = TrainModel()
    print("config_path = ", config_path)
    trainModel.load_config(config_path)
    trainModel.train()