# -*- coding: UTF-8 -*-

import timeit, datetime
import sys
import json
import numpy as np
from gensim.models.word2vec import Word2Vec
import operator

# 传入字符串时间形如20190202
# 返回timeA - timeB 整数值
def timediff(timeA, timeB):
    date_a = datetime.datetime.strptime(timeA, "%Y%m%d")
    date_b = datetime.datetime.strptime(timeB, "%Y%m%d")
    diff = str(date_a - date_b)
    if diff.find("day") == -1:
        return int(diff.split(":")[0])
    else :
        return int(diff.split(" ")[0])

# 将word用训练好的model转换为vector
def vec(words, model):
    result = []
    for word in words:
        result.append(model.wv[word].tolist())
    return result

'''
将word用训练好的model转换为vector，然后进行简单向量相加操作构成句子
input: 
      words: 待转换词语list
      model: 已训练好的word2vec模型  
output: 
      words转换的vec list eg: []
'''
def sen2vec(words, model):
    tmp = vec(words, model)
    result = np.array(tmp[0])
    for i in range(len(tmp) - 1):
        result = np.add(result, np.array(model.wv[words[i+1]].tolist()))

    return result.tolist()


def strtime2list(strTime):
    result = []
    year = int(strTime[:4])
    month = int(strTime[4:6])
    day = int(strTime[6:8])
    result.append(year)
    result.append(month)
    result.append(day)
    return result

def load_data(file_path, configs):
    data_config = configs['data']
    column_num = data_config['columnNum']
    separator = data_config['separator']
    columns = data_config['columns']
    group = data_config['group']
    group_idx = columns.index(group)
    time_column = data_config['timeColumn']
    time_idx = columns.index(time_column)
    count_column = data_config['countColumn']
    count_idx = columns.index(count_column)
    recorders = {}
    times = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            column_data = line.strip().split(separator)
            jsonpath = column_data[group_idx]
            assert len(column_data) == column_num, "len of {} is not {}".format(column_data, column_num)
            t = column_data[time_idx]
            times.add(t)
            if jsonpath in recorders.keys():
                recorders[jsonpath].append({t: int(column_data[count_idx])})
            else:
                recorders[jsonpath] = []
                recorders[jsonpath].append({t: int(column_data[count_idx])})
        # 将时间排序
        times = list(times)
        times.sort()
    print("load data from %s finished." % file_path)
    return recorders, times

def utils(file_path, configs):
    recorders, times = load_data(file_path, configs)
    print(len(recorders))
    print(len(times))
    # print(times)
    data_config = configs['data']
    window_size = data_config['windowSize']

    data = []

    t0 = timeit.default_timer()
    print("model path: ", data_config['vecModel'])
    # 构造feature
    vecModel = Word2Vec.load(data_config['vecModel'])

    for jsonkey in recorders.keys():
        recorder = recorders[jsonkey]
        # jsonpath + path
        words = jsonkey.split("@@@@")
        # print(type(words))
        words.extend(jsonkey.split("@@@@")[:-1])
        # print(words)
        wordvec = sen2vec(words, vecModel)
        # time series
        start = 0
        end = window_size
        for i in range(len(times) - window_size):
            tmp = []
            tmp.extend(wordvec)
            currentTime = times[i + window_size]
            # print(currentTime)
            for item in recorder[start:end]:
                # print("recorder",recorder[start:end])
                for key, value in item.items():
                    tmp.append(timediff(str(key), str(currentTime)))
                    tmp.append(int(value))
            tmp.extend(strtime2list(currentTime))  # currentTime diff
            # print(tmp)
            data.append((tmp, 0 if recorder[i + window_size][currentTime] < 2 else 1, jsonkey, currentTime))
            # print(data)
            start = start + 1
            end = end + 1
    elapsed = timeit.default_timer() - t0
    print("construct features耗时：", elapsed, " s")
    print("construct features finished.")
    print(len(data))

    return data, times[-1]

def str2num(s):
    # print(s)
    if str(s).count('.') > 0:
        result = float(s)
    else:
        result = int(s)
    return result

# input: raw data filename, out put filename
# output: data->feature
def generateFeature(filein, fileout, configs):
    data, last_timestamp = utils(filein, configs)

    with open(fileout, "w", encoding='utf-8') as f:
        for i in data:
            s = str(i[0])
            for j in range(len(i) - 1):
                s = s + "^^" + str(i[j + 1])
            f.write(str(s))
            f.write('\n')
    print("write %s finished." % fileout)
    return data

# input: filename
# output: data = [(), ()]
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

def main(file_path, jsonpath):
    configs = json.load(open(jsonpath, 'r'))
    window_size = configs['data']['windowSize']
    fileout = "data/" + str(window_size) + "_feature_" + str(file_path).split('/')[-1]
    print(fileout)
    data = generateFeature(file_path, fileout, configs)
    t0 = timeit.default_timer()
    data1 = loadFeature(fileout)
    elapsed = timeit.default_timer() - t0
    print("load features耗时：", elapsed, " s")
    print(operator.eq(data,data1))

if __name__ == '__main__':
    t0 = timeit.default_timer()
    file_path = sys.argv[1]
    jsonpath = sys.argv[2]
    main(file_path, jsonpath)
    t1 = timeit.default_timer()
    print(t1 - t0, " s")