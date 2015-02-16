#!/usr/bin/env python
#-*-coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import *
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.base import RegressorMixin, BaseEstimator

class EnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressors=None):
        self.regressors = regressors

    def fit(self, X, y):
        for regressor in self.regressors:
            regressor.fit(X, y)

    def predict(self, X):
        self.predictions_ = list()
        for regressor in self.regressors:
            self.predictions_.append(regressor.predict(X))
        return np.mean(self.predictions_, axis=0)


def fit_regressors():
    dico = {}
    for Type in ["A", "B", "C"]:
        for spec in ["", "spec"]:
            print Type, " ", spec
            X = pd.read_table(Type+spec+"values.csv", sep=',',
                    warn_bad_lines=True,
                    error_bad_lines=True)
            X = X.drop("Date", 1)
            X = X.drop("Store", 1)
            X = X.drop("Dept", 1)
            X = X.drop("Type", 1)
            X = X.drop("IsHoliday", 1)
            Y = X["Weekly_Sales3"]
            X = X.drop("Weekly_Sales3", 1)

            # X1 = preprocessing.scale(X1)
            # X2 = preprocessing.scale(X2)

            # rbf_svc = svm.SVC(kernel='rbf',
            #                   cache_size=1000)

            # rbf_svc.fit(X1, Y1)
            r1 = linear_model.LassoLars(alpha = 0.1)
            r2 = RandomForestRegressor(n_estimators=200)
            #clf = AdaBoostRegressor(base_estimator=RandomForestRegressor(n_estimators=15),n_estimators=50)
            r3 = GradientBoostingRegressor(loss='lad', n_estimators=200, max_depth=7)
            reg = EnsembleRegressor([r1,r2,r3,r3])
            reg.fit(X, Y)
            dico[Type+spec] = reg
    return dico
