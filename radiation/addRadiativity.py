# coding=UTF-8
# 把辐照度相加
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer


import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU内存不够
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
tf.Session(config=config)

# 记录程序运行时间
import time
start_time = time.time()

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# 读入数据
train_feature = pd.read_csv("train_feature.csv", low_memory=False)
train_label = pd.read_csv("train_label.csv", low_memory=False)
test_feature = pd.read_csv("test_feature.csv", low_memory=False)

# 预处理：把时刻拼接成一天
cur = 0
flag = 0
tmp = []
train = []
radiation = 0
for index, row in train_feature.iterrows():
    # 8个数据为一天
    if index // 8 != cur:
        cur = cur + 1 # 下一天
        flag = 0
        tmp.append(radiation)
        radiation = 0
        train.append(tmp)
    if flag == 0:
        flag = 1
        tmp = [cur+1]# 放入日期编号
    # 放入别的数据
    # tmp.append(row["辐照度"])
    radiation = radiation + row["辐照度"] # 辐照度相加
    tmp.append(row["风速"])
    tmp.append(row["风向"])
    tmp.append(row["温度"])
    tmp.append(row["湿度"])
    tmp.append(row["气压"])
tmp.append(radiation)
train.append(tmp)

head = 'date'
for i in range(1, 9):
    # head = head + "," + 'radioactivity' + str(i)
    head = head + "," + 'windSpeed' + str(i)
    head = head + "," + 'windDirection' + str(i)
    head = head + "," + 'temperature' + str(i)
    head = head + "," + 'humidity' + str(i)
    head = head + "," + 'pressure' + str(i)
head.encode('gb2312')
print(head)
head = head + "," + 'radioactivity'

np.savetxt('train01.csv', train, delimiter=',', header=head, comments='', fmt='%lf')

# test
# 预处理：把时刻拼接成一天
cur = 0
flag = 0
tmp = []
test = []
for index, row in test_feature.iterrows():
    # 8个数据为一天
    if index // 8 != cur:
        cur = cur + 1 # 下一天
        flag = 0
        tmp.append(radiation)
        radiation = 0
        test.append(tmp)
    if flag == 0:
        flag = 1
        tmp = [cur+1]#放入日期编号
    # 放入别的数据
    # tmp.append(row["辐照度"])
    radiation = radiation + row["辐照度"]  # 辐照度相加
    tmp.append(row["风速"])
    tmp.append(row["风向"])
    tmp.append(row["温度"])
    tmp.append(row["湿度"])
    tmp.append(row["气压"])
tmp.append(radiation)
test.append(tmp)

np.savetxt('test01.csv', test, delimiter=',', header=head, comments='', fmt='%lf')
