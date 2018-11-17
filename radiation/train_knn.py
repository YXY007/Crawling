# coding=UTF-8

import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB

# 读入数据
train_feature = pd.read_csv("train01.csv", low_memory=False)
train_label = pd.read_csv("train_label.csv", low_memory=False)
tests = pd.read_csv("test01.csv", low_memory=False)

# 不要日期
train_feature = train_feature.drop(['date'], axis=1)
train_label = train_label.drop(['日期'], axis=1)

date = tests.date
test = tests.drop(['date'], axis=1)

# 归一化
min_max_scaler = MinMaxScaler()
train_feature = min_max_scaler.fit_transform(train_feature)
test= min_max_scaler.transform(test)

# 用sklearn.cross_validation进行训练数据集划分，这里训练集和交叉验证集比例为7：3，可以自己根据需要设置
X_train, X_test, y_train, y_test = train_test_split(train_feature, train_label, test_size=0.3, random_state=1)


# r2
def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """
    # score = r2_score(y_true, y_predict)
    #
    # return score
    mae = np.sum(np.absolute(y_true - y_predict)) / len(y_predict)
    return mae


def fit_model_shuffle(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

    # Create a KNN regressor object
    regressor = KNeighborsRegressor()

    # Create a NB regressor object
    clf = GaussianNB()

    # Create a dictionary for the parameter 'n_neighbors' with a range from 3 to 10
    params = {'n_neighbors': range(1, 100, 10)}

    # Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(regressor, param_grid=params, scoring='neg_mean_absolute_error', cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


# k-fold
def fit_model_k_fold(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    # cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    k_fold = KFold(n_splits=10)

    # TODO: Create a decision tree regressor object
    regressor = KNeighborsRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'n_neighbors': range(1, 200, 1)}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search object
    grid = GridSearchCV(regressor, param_grid=params, scoring=scoring_fnc, cv=k_fold)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


reg = fit_model_k_fold(train_feature, train_label)
# reg = fit_model_shuffle(train_feature, train_label)

print "Parameter 'n_neighbors' is {} for the optimal model.".format(reg.get_params()['n_neighbors'])


# Show predictions
np.savetxt('knn_submission.csv', np.c_[date, reg.predict(test)], delimiter=',', header='time,prediction', comments='', fmt='%d,%.15lf')

print performance_metric(reg.predict(train_feature), train_label)