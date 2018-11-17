# coding=UTF-8

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

# 填补缺失值
# train = train.fillna(0)  # 用0替换缺失值
# tests = train.fillna(0)

# 填补\N
# train.replace("\N", 0, inplace=True) #前面是需要替换的值，后面是替换后的值。
# tests.replace("\N", 0, inplace=True)

# 预处理
# 类别字典
#dic = dict()
#cla = 0
#lst = []
# for i, val in enumerate(train.current_service):
#     if dic.has_key(val):
#         lst.append(dic[val])
#     else:
#         cla = cla + 1
#         lst.append(cla)
# train.drop(['current_service'], axis=1)
# train.loc[:, 'current_service'] = lst

# 预处理：把时刻拼接成一天
cur = 0
flag = 0
tmp = []
train = []
for index, row in train_feature.iterrows():
    # 8个数据为一天
    if index // 8 != cur:
        cur = cur + 1 # 下一天
        flag = 0
        train.append(tmp)
    if flag == 0:
        flag = 1
        tmp = [cur+1]#放入日期编号
    # 放入别的数据
    tmp.append(row["辐照度"])
    tmp.append(row["风速"])
    tmp.append(row["风向"])
    tmp.append(row["温度"])
    tmp.append(row["湿度"])
    tmp.append(row["气压"])
train.append(tmp)

head = 'date'
for i in range(1, 9):
    head = head + "," + 'radioactivity' + str(i)
    head = head + "," + 'windSpeed' + str(i)
    head = head + "," + 'windDirection' + str(i)
    head = head + "," + 'temperature' + str(i)
    head = head + "," + 'humidity' + str(i)
    head = head + "," + 'pressure' + str(i)
head.encode('gb2312')
print(head)

np.savetxt('train.csv', train, delimiter=',', header=head, comments='', fmt='%lf')

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
        test.append(tmp)
    if flag == 0:
        flag = 1
        tmp = [cur+1]#放入日期编号
    # 放入别的数据
    tmp.append(row["辐照度"])
    tmp.append(row["风速"])
    tmp.append(row["风向"])
    tmp.append(row["温度"])
    tmp.append(row["湿度"])
    tmp.append(row["气压"])
test.append(tmp)

np.savetxt('test.csv', test, delimiter=',', header=head, comments='', fmt='%lf')

# 用sklearn.cross_validation进行训练数据集划分，这里训练集和交叉验证集比例为7：3，可以自己根据需要设置
# train_xy, val = train_test_split(train, test_size=0.3, random_state=1)
#
# y = train_xy.current_service
# X = train_xy.drop(['user_id', 'current_service'], axis=1)
# val_y = val.current_service
# val_X = val.drop(['user_id', 'current_service'], axis=1)
#
# ID = tests.user_id
# test = tests.drop(['user_id'], axis=1)
#
# # 转换为float
# X = pd.DataFrame(X, dtype=np.float)
# val_X = pd.DataFrame(val_X, dtype=np.float)
# test = pd.DataFrame(test, dtype=np.float)
#
# #xgb矩阵赋值
# xgb_val = xgb.DMatrix(val_X, label=val_y)
# xgb_train = xgb.DMatrix(X, label=y)
# xgb_test = xgb.DMatrix(test)
#
# params = {
#     'booster': 'gbtree',
#     'objective': 'multi:softmax',  #多分类的问题
#     'num_class': cla,  # 类别数，与 multisoftmax 并用
#     'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
#     'max_depth': 12,  # 构建树的深度，越大越容易过拟合
#     'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#     'subsample': 0.7,  # 随机采样训练样本
#     'colsample_bytree': 0.7,  # 生成树时进行的列采样
#     'min_child_weight': 3,
#     # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#     #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#     #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
#     'silent': 0,  #设置成1则没有运行信息输出，最好是设置为0.
#     'eta': 0.007,  # 如同学习率
#     'seed': 1000,
#     'nthread': 10,# cpu 线程数
#     #'eval_metric': 'auc'
# }
# plst = list(params.items())
# num_rounds = 5000  # 迭代次数
# watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
#
# #训练模型并保存
# # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
# model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
# model.save_model('./model/xgb.model')  # 用于存储训练出的模型
# print "best best_ntree_limit", model.best_ntree_limit
#
# preds = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)
#
# # 转字典
# for i, val in enumerate(preds):
#     preds[i] = dic[preds[i]]
#
# np.savetxt('xgb_submission.csv', np.c_[ID, preds], delimiter=',', header='ImageId,Label', comments='', fmt='%d')
#
# #输出运行时长
# cost_time = time.time()-start_time
# print "xgboost success!", '\n', "cost time:", cost_time, "(s)......"
