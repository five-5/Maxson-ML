import datetime
import sys
import lstmcrf_train
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json
from sklearn.metrics import *

def utils(file_path, configs):
    # 读取配置
    data_config = configs['data']
    window_size = data_config['windowSize']
    windows_num = data_config['windowsNum']
    separator = data_config['separator']

    # 加载保存好的数据
    seqs = {}  # {jsonPath: [(feature, label),...]}
    data = []
    d1 = datetime.datetime.now()
    print("start read file")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip().split(separator)
            vecpath = s[0].strip('[').strip(']').split(',')
            vecpath = [lstmcrf_train.str2num(i) for i in vecpath]
            label = int(s[1])
            jsonpath = s[2]
            curTime = s[3]
            train_data_line = (vecpath, label, jsonpath, curTime)
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
    d2 = datetime.datetime.now()
    print("read data : ", (d2 - d1).total_seconds(), " s")
    return data

class SeqDataLoader(Dataset):
    def __init__(self, data, configs):
        self.data = data
        self.window_size = configs['data']['windowSize']
        self.nonseq_feature_size = configs['data']['nonSeqFeatureSize']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        x = torch.tensor([y[0] for y in seq[:]])
        x = x.view(-1, self.window_size * 2 + self.nonseq_feature_size)
        label = [y[1] for y in seq[:]]
        labels = torch.tensor(label, dtype=torch.long)
        jsonpath = [y[2] for y in seq[:]]
        curtime = [y[3] for y in seq[:]]
        return x, labels, jsonpath, curtime

def main(config_path):
    # load模型
    configs = json.load(open(config_path, 'r'))
    file_path = configs['data']['predictName']
    model_path = configs['model']['saveName']
    pmodel = lstmcrf_train.MyModel(configs)   # 提取参数首先需要建立和原来一模一样的神经网络框架
    pmodel.load_state_dict(torch.load(model_path)) # 然后load参数
    pmodel.eval() # 去掉dropout层

    # 准备数据
    predict = []
    print("read data from %s" % file_path)
    data = utils(file_path, configs)
    prediction_config = configs['prediction']
    data_set = SeqDataLoader(data, configs)
    data_loader = DataLoader(data_set,
                             batch_size=1,
                             shuffle=False)
    print("start prediction.")
    d1 = datetime.datetime.now()
    accuracy = 0
    # 总的测试个数
    total = 0
    labels = []
    paths = []
    times = []
    for x, label, jsonpath, curtime in data_loader:
        label = label.view(-1)
        out, preds = pmodel(x)
        paths.extend([jp for jp in jsonpath])
        times.extend([ti for ti in curtime])
        predict.extend([pred for pred in preds])
        labels.extend(label.numpy().tolist())
        correct = (np.array(preds) == label.numpy()).sum()
        accuracy += correct.item()
        total += x.size(0)
    d2 = datetime.datetime.now()
    print("预测时间：", (d2 - d1).total_seconds(), " s")
    # 用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组
    print("paths, times, predict", len(paths), len(times), len(predict))
    result = list(zip(paths, times, predict))
    print(file_path, 'accuracy = ', accuracy / total)
    precision = precision_score(labels, predict)
    print('precision = ', precision)
    recall = recall_score(labels, predict)
    print('recall = ', recall)
    mif1score = f1_score(labels, predict, average='micro')
    maf1score = f1_score(labels, predict, average='macro')
    print(file_path, 'f1score mi= ', mif1score)
    print(file_path, 'f1score ma= ', maf1score)

    out_path = prediction_config['output']
    print(out_path)
    with open(out_path, "w", encoding='utf-8') as f:
        for r in result:
            f.write(str(r[0]) + "^^")  # jsonpath
            f.write(str(r[2]) + "^^")  # 预测结果
            f.write(str(r[1]))  # 时间
            f.write('\n')
    f.close()


if __name__ == '__main__':
    # config_path = 'config.json'
    d1 = datetime.datetime.now()
    config_path = sys.argv[1]
    main(config_path)
    d2 = datetime.datetime.now()
    print((d2-d1).total_seconds())
