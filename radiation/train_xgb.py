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
train_feature = pd.read_csv("train01.csv", low_memory=False)
train_label = pd.read_csv("train_label.csv", low_memory=False)
tests = pd.read_csv("test01.csv", low_memory=False)

# 用sklearn.cross_validation进行训练数据集划分，这里训练集和交叉验证集比例为7：3，可以自己根据需要设置
X_train, X_test, y_train, y_test = train_test_split(train_feature, train_label, test_size=0.3, random_state=1)

# 不要日期
X_train = X_train.drop(['date'], axis=1)
X_test = X_test.drop(['date'], axis=1)
y_train = y_train.drop(['日期'], axis=1)
y_test = y_test.drop(['日期'], axis=1)

date = tests.date
test = tests.drop(['date'], axis=1)
#
# # 转换为float
# X = pd.DataFrame(X, dtype=np.float)
# val_X = pd.DataFrame(val_X, dtype=np.float)
# test = pd.DataFrame(test, dtype=np.float)

# xgb矩阵赋值
xgb_val = xgb.DMatrix(X_test, label=y_test)
xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_test = xgb.DMatrix(test)

params = {
    'booster': 'gbtree',
    'objective': 'reg:linear',  #回归问题
    #'num_class': cla,  # 类别数，与 multisoftmax 并用
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 5,  # 构建树的深度，越大越容易过拟合
    'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'min_child_weight': 2,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'silent': 0,  #设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.007,  # 如同学习率
    'seed': 1000,
    'nthread': 10,# cpu 线程数
    'eval_metric': 'mae'
}
plst = list(params.items())
num_rounds = 5000  # 迭代次数
watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]

#训练模型并保存
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
model.save_model('xgb.model')  # 用于存储训练出的模型
print "best best_ntree_limit", model.best_ntree_limit

preds = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)
#
# # 转字典
# for i, val in enumerate(preds):
#     preds[i] = dic[preds[i]]
#
np.savetxt('xgb_submission.csv', np.c_[date, preds], delimiter=',', header='time,prediction', comments='', fmt='%d,%.15lf')

#输出运行时长
cost_time = time.time()-start_time
print "xgboost success!", '\n', "cost time:", cost_time, "(s)......"
