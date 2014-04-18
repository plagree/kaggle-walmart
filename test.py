#!/usr/bin/env python
#-*-coding: utf-8 -*-

from walmart import *
from sklearn import *
import random

def test():
    for Type in ["A", "B", "C"]:
        for spec in ["", "spec"]:
            print Type, " ", spec
            X = pd.read_table(Type+spec+"values.csv", sep=',', warn_bad_lines=True, error_bad_lines=True)
            length = len(X)
            print length
            number = length * 9 / 10
            liste = random.sample(range(length), number)
            missing = list(set(range(length)) - set(liste))
            X1 = X.iloc[liste]
            print len(X1)
            X1 = X1.drop("Date", 1)
            X1 = X1.drop("Store", 1)
            X1 = X1.drop("Dept", 1)
            X1 = X1.drop("Type", 1)
            Y1 = X1["Weekly_Sales3"]
            X1 = X1.drop("Weekly_Sales3", 1)
            X2 = X.iloc[missing]
            print len(X2)
            X2 = X2.drop("Date", 1)
            X2 = X2.drop("Dept", 1)
            X2 = X2.drop("Store", 1)
            X2 = X2.drop("Type", 1)
            Y2 = X2["Weekly_Sales3"]
            X2 = X2.drop("Weekly_Sales3", 1)
            return X1, Y1, X2, Y2
            
            # X1 = preprocessing.scale(X1)
            # X2 = preprocessing.scale(X2)
            
            # rbf_svc = svm.SVC(kernel='rbf',
            #                   cache_size=1000)
            
            # rbf_svc.fit(X1, Y1)
            clf = linear_model.Lasso(alpha = 0.1)
            clf2 = linear_model.LassoLars(alpha = 0.1)
            #clf3 = linear_model.SGDRegressor()
            clf4 = linear_model.Ridge(alpha = .5)
            clf5 = linear_model.BayesianRidge()

            clf.fit(X1, Y1)
            clf2.fit(X1, Y1)
            #clf3.fit(X1, Y1)
            clf4.fit(X1, Y1)
            clf5.fit(X1, Y1)
            Y2f = clf.predict(X2)
            Y2f2 = clf2.predict(X2)
            #Y2f3 = clf3.predict(X2)
            Y2f4 = clf4.predict(X2)
            Y2f5 = clf4.predict(X2)

            error = metrics.mean_absolute_error(Y2, Y2f)
            error2 = metrics.mean_absolute_error(Y2, Y2f2)
            #error3 = metrics.mean_absolute_error(Y2, Y2f3)
            error4 = metrics.mean_absolute_error(Y2, Y2f4)
            error5 = metrics.mean_absolute_error(Y2, Y2f4)
            print "Error Lasso: " + str(error)
            print "Error LassoLars: " + str(error2)
            #print "Error Lasso: " + str(error3)
            print "Error Ridge: " + str(error4)
            print "Error BayesianRidge: " + str(error5)
            break
    return #X1, X2, Y1, Y2#, clf
