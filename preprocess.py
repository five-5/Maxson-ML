#!/usr/bin/python
# -*- coding: UTF-8 -*-

import datetime
import sys
import numpy as np
from gensim.models.word2vec import Word2Vec

sys.path.append('../data/')

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


    # return int(diff)

# 读取数据文本获取时间总天数
def getTimeCount(filename):
    times = set()
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("^^")
            times.add(line[2])
    return times

# 传入字符串时间形如20190202
# 生成timeA到timeB的时间序列
def generateData(timeA, timeB):
    times = []
    delta = timediff(timeB, timeA)
    print("delta = ", delta)
    for i in range(delta+1):
        times.append((datetime.datetime.strptime(timeA, "%Y%m%d") + datetime.timedelta(days=i)).strftime("%Y%m%d"))
    return times

# 检查数据文本中是否存在缺失，以及具体的缺失项
def checkDataFile(filename):
    days = getTimeCount(filename)
    print("days = ", len(days))
    days = list(days)
    days.sort()
    timeB = str(days[0])

    # 生成区间正确时间序列
    true_days = generateData(str(days[0]), str(days[-1]))

    for i in range(len(true_days)):
        for j in range(len(days)):
            while true_days[i] < days[j]:
                print(i, true_days[i], j, days[j])
                i = i + 1
            while true_days[i] > days[j]:
                print(i, true_days[i], j, days[j])
                j = j + 1
            i = i + 1
            j = j + 1
        break
    print("check data file finished.")

# 补全所有未出现的值
# 输入文件：jsonkey^^count^^time
# 输出文件：jsonkey^^count^^time
def complete(filename, fileout, delimeter = "^^"):
    # 获取所有的jsonpath
    jsonpath = set()
    timestamp = set()
    dict = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split(delimeter)
            jsonpath.add(line[0])
            timestamp.add(line[2])
            if line[2] in dict.keys():
                dict[line[2]][line[0]] = line[1]
            else:
                dict[line[2]] = {}
                dict[line[2]][line[0]] = line[1]
    with open(fileout, "w", encoding="utf-8") as f:
        for time in timestamp:
            content = dict[time]
            if len(content) != len(jsonpath):
                for sql in jsonpath:
                    if sql not in content.keys():
                        content[sql] = 0
            for key, value in content.items():
                f.write(key + delimeter + str(value) + delimeter + time)
                f.write('\n')

# 按jsonpath排序输出 
def groupJson(filename, fileout, delimeter = "^^"):
    # 获取所有的jsonpath
    dict = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split(delimeter)
            jsonpath = line[0]
            count = line[1]
            timestamp = line[2]
            if jsonpath in dict.keys():
                dict[jsonpath][timestamp] = count
            else:
                dict[jsonpath] = {}
                dict[jsonpath][timestamp] = count
    print("load data from %s succeed." % filename)

    with open(fileout, "w", encoding="utf-8") as f:
        for jsonpath in dict.keys():
            # 按时间排序
            content = [(k,dict[jsonpath][k]) for k in sorted(dict[jsonpath].keys())]
            for recorder in content:
                f.write(jsonpath + delimeter + str(recorder[1]) + delimeter + recorder[0])
                f.write('\n')
    print("write data to %s succeed." % fileout)

# 获取offset开始的n天数据到fileout中
# 输入文件格式为：jsonkey^^count^^time
# 输出文件格式为：jsonkey^^count^^time
def spliteData(filein, fileout, offset, n, delimter = "^^"):
    # load data save in tmpMap as {time:[jsonkey, count, jsonkey2, count2,....]}
    tmpMap = {}
    with open(filein, 'r', encoding="utf-8") as f:
        for line in f:
            ls = line.strip().split(delimter)
            if ls[0] == "":
                continue
            time = ls[2]
            key = ls[0]
            count = int(ls[1])
            if time in tmpMap.keys():
                tmpMap[time].append(key)
                tmpMap[time].append(count)
            else:
                tmpMap[time] = [key, count]
    f.close()
    print("load data succeed from ", filein)

    if offset > len(tmpMap):
        print("bad offset")
        return

    if offset + n > len(tmpMap):
        n = len(tmpMap) - offset
        print("n becomes to ", n)

    # write file
    with open(fileout, 'w', encoding="utf-8") as f:
        keys = list(tmpMap.keys())
        for k in range(n):
            j = k + offset
            key = keys[j]
            for i in range(int(len(tmpMap[key]) / 2)):
                f.write(tmpMap[key][2 * i] + delimter + str(tmpMap[key][2 * i + 1]) + delimter + key + "\n")
    f.close()
    print("write done.")

# 加载jsonpath
def load_sentences(filename):
        jsonpath = set()
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split("^^")
                if line[0] != "":
                    jsonpath.add(line[0])
        return jsonpath

# 将word用训练好的model转换为vector
def vec(words, model):
    result = []
    print(words)
    for word in words:
        result.append(model.wv[word].tolist())
    return result

if __name__ == '__main__':
    # 1. 检查数据
    checkDataFile("full_set_group_json.txt")

    # 2. 分割训练集、测试集、预测集
    spliteData("data/full_set_group_json.txt", "data/train.txt", 30, 60)
    spliteData("data/full_set_group_json.txt", "data/predict.txt", 90, 30)

    # 3. 训练word2vec模型
    raw_sentences = list(load_sentences("full_set_group_json.txt"))
    print(len(raw_sentences))
    sentences = [s.split("@@@@") for s in raw_sentences]
    for i in range(10):
        print(sentences[i])

    maxLenth = 0
    for s in raw_sentences:
        s = s.split("@@@@")
        for i in range(len(s)):
            if len(s[i]) > maxLenth:
                maxLenth = len(s[i])
    print("maxLenth = ", maxLenth)

    t0 = timeit.default_timer()
    # 构建模型
    model = Word2Vec(sentences, size=50, window=4, min_count=1, sg=1, iter=20)  # 10-25次
    elapsed = timeit.default_timer() - t0
    print("训练模型耗时：", elapsed, " s")

    # 保存模型
    model.save('wordModel_50_4.model')

    # example
    # # 获取词向量
    # print(model.wv['auction_base'])
    # print((model.wv['auction_base']).shape)


    # # 加载模型
    # model = Word2Vec.load('wordModel_50_4.model')
    # s = "auction_base@@@@s_toop@@@@payload@@@@channel"
    # s1 = "auction_base@@@@s_toop@@@@payload@@@@cp_card_swiftcode"
    # s2 = "icbuda_alg@@@@s_tt_icbu_tpp_text_log_tt4@@@@content@@@@itemInfo[0].triggerId"
    # vector1 = vec(s.split("@@@@"), model)
    # vector2 = vec(s1.split("@@@@"), model)
    # vector3 = vec(s2.split("@@@@"), model)
    # print(vector1)
    # print(len(vector1[0]))
    # print(np.array(vector1).reshape((4,50)))