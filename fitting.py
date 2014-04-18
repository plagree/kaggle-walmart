#!/usr/bin/env python
#-*-coding: utf-8 -*-

from walmart import *
from sklearn import *

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
            clf = linear_model.LassoLars(alpha = 0.1)
            clf.fit(X, Y)
            dico[Type+spec] = clf
    return dico
