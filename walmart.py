"""

Kaggle Store Sales Forecasting.
__author__ : Paul Lagree
__date__ : 26/03/14 - 

"""

import time
import pandas as pd
import numpy as np
import datetime as dt
import cPickle
from sklearn import preprocessing
from sklearn.svm import LinearSVC
import  scipy.stats as stats
import sklearn.linear_model as lm
from week_searcher import *

Types = ["A", "B", "C"]
Specs = [True, False]

def storesdata(filename):
    X = pd.read_table(filename, sep=',', warn_bad_lines=True,
                      error_bad_lines=True)
    # X = np.asarray(X.values, dtype
    return X

def featuresdata(filename):
    X = pd.read_table(filename, sep=',', warn_bad_lines=True,
                      error_bad_lines=True)
    return X

def traindata(filename):
    X = pd.read_table(filename, sep=',', warn_bad_lines=True,
                      error_bad_lines=True)
    return X

def testdata(filename):
    X = pd.read_table(filename, sep=',', warn_bad_lines=True,
                      error_bad_lines=True)
    return X

def join(Xtrain, Xfeatures, Xstores):
    Xmerged = pd.merge(Xtrain, Xfeatures, how='inner')
    Xfinal = pd.merge(Xmerged, Xstores, how='inner')
    # labels = Xfinal["Weekly_Sales"]
    # Drop feature to Xfinal
    return Xfinal # , labels

def createDataset(Xtypespec, Xttype, Xspec, holiday, perfect=False):
    Xtypespec = Xtypespec.sort(["Store", "Dept", "Date"])
    # X1.reset_index(drop=True)
    # Create Dataframe
    Size = []
    Store = []
    Dept= []
    Date = []
    IsHoliday = []
    Type = []
    # Week year -1
    Weekly_Sales1 = []
    Temperature1 = []
    Fuel_Price1 = []
    MarkDown11 = []
    MarkDown21 = []
    MarkDown31 = []
    MarkDown41 = []
    MarkDown51 = []
    CPI1 = []
    Unemployment1 = []
    # Week week -1 (not holidays)
    Weekly_Sales2 = []
    Temperature2 = []
    Fuel_Price2 = []
    MarkDown12 = []
    MarkDown22 = []
    MarkDown32 = []
    MarkDown42 = []
    MarkDown52 = []
    CPI2 = []
    Unemployment2 = []
    # Week studied :) 
    Weekly_Sales3 = []
    Temperature3 = []
    Fuel_Price3 = []
    MarkDown13 = []
    MarkDown23 = []
    MarkDown33 = []
    MarkDown43 = []
    MarkDown53 = []
    CPI3 = []
    Unemployment3 = []
    dic = {}

    # Population algorithm
    print str(len(Xtypespec)), " LENGTH"
    previousStore = None
    previousDept = None
    for i in range(len(Xtypespec)):
        if i%1000 == 0 and i != 0:
            print i

        row = Xtypespec.iloc[i]
        date = dt.datetime.strptime(row["Date"], "%Y-%m-%d").date()

        if previousStore != row["Store"] or previousDept != row["Dept"]:
            ok = True
            ok2 = True
            Xttype_curr = Xttype[(Xttype.Store == row["Store"]) & (Xttype.Dept == row["Dept"])]
            Xspec_curr = Xspec[(Xspec.Store == row["Store"]) & (Xspec.Dept == row["Dept"])]
            if len(Xspec_curr) == 0:
                ok = False
            if len(Xttype_curr) == 0:
                ok2 = False
            previousStore = row["Store"]
            previousDept = row["Dept"]

        # ### LAST YEAR ###
        # if holiday:
        #     datel = str(date - dt.timedelta(days=365) - dt.timedelta(days=110))
        #     dater = str(date - dt.timedelta(days=365) + dt.timedelta(days=11))
        #     if ok:
        #         liste = Xspec_curr[(Xspec_curr.Date > datel) & (Xspec_curr.Date < dater)]
        #     else:
        #         liste = []
        # else:
        #     datel = str(date - dt.timedelta(days=365) - dt.timedelta(days=100))
        #     dater = str(date - dt.timedelta(days=365) + dt.timedelta(days=4))
        #     if ok2:
        #         liste = Xttype_curr[(Xttype_curr.Date > datel) & (Xttype_curr.Date < dater)]
        #     else:
        #         liste = []

        # if len(liste) == 0:
        #     if perfect:
        #         continue
        #     last_year = row
        # else:
        #     last_year = liste.iloc[len(liste)-1]
        # 
        # ###### WEEKLY_SALES MANAGEMENT FOR TESTING ######
        # sales = last_year["Weekly_Sales"]
        # if sales is None:
        #     if holiday:
        #         datel = str(date - dt.timedelta(days=730) - dt.timedelta(days=110))
        #         dater = str(date - dt.timedelta(days=730) + dt.timedelta(days=11))
        #         if ok:
        #             liste = Xspec_curr[(Xspec_curr.Date > datel) & (Xspec_curr.Date < dater)]
        #         else:
        #             liste = []
        #     else:
        #         datel = str(date - dt.timedelta(days=730) - dt.timedelta(days=100))
        #         dater = str(date - dt.timedelta(days=730) + dt.timedelta(days=4))
        #         if ok2:
        #             liste = Xttype_curr[(Xttype_curr.Date > datel) & (Xttype_curr.Date < dater)]
        #         else:
        #             liste = []
        #     if len(liste) == 0:
        #         print "PROBLEM"
        #     else:
        #         sales = liste.iloc[len(liste)-1]["Weekly_Sales"]
        # if sales is None:
        #     print "PROBLEM"
        # ##################################################

        last_year, sales = last_year_same_week(row, date, Xttype_curr, Xspec_curr,
                                               ok, ok2, perfect, holiday)

        if last_year is None:
            continue

        # ## LAST WEEK ##
        # datel = str(date - dt.timedelta(days=28))
        # dater = str(date - dt.timedelta(days=4))
        # if ok2:
        #     liste = Xttype_curr[(Xttype_curr.Date > datel) & (Xttype_curr.Date < dater)]
        # else:
        #     liste = []
        # if len(liste) == 0:
        #     if perfect:
        #         continue
        #     if holiday:
        #         last_week = Xttype_curr.iloc[0]
        #     else:
        #         last_week = row
        # else:
        #     last_week = liste.iloc[len(liste)-1]

        # sales2 = last_week["Weekly_Sales"]

        # ####### NEW WORK ############
        # if sales2 is None:
        #     datel = str(date - dt.timedelta(days=365) - dt.timedelta(days=100))
        #     dater = str(date - dt.timedelta(days=365) - dt.timedelta(days=4))
        #     if ok2:
        #         liste = Xttype_curr[(Xttype_curr.Date > datel) & (Xttype_curr.Date < dater)]
        #     else:
        #         liste = []
        #     if len(liste) == 0:
        #         datel = str(date - dt.timedelta(days=730) - dt.timedelta(days=100))
        #         dater = str(date - dt.timedelta(days=730) - dt.timedelta(days=4))
        #         if ok2:
        #             liste = Xttype_curr[(Xttype_curr.Date > datel) & (Xttype_curr.Date < dater)]
        #         else:
        #             liste = []
        #         if (len(liste)) == 0:
        #             print "PROBLEM"
        #         else:
        #             sales2 = liste.iloc[len(liste)-1]["Weekly_Sales"]
        #     else:
        #         sales2 = liste.iloc[len(liste)-1]["Weekly_Sales"]
        #         if sales2 is None:
        #             datel = str(date - dt.timedelta(days=730) - dt.timedelta(days=100))
        #             dater = str(date - dt.timedelta(days=730) - dt.timedelta(days=4))
        #             if ok2:
        #                 liste = Xttype_curr[(Xttype_curr.Date > datel) & (Xttype_curr.Date < dater)]
        #             else:
        #                 liste = []
        #             if (len(liste)) == 0:
        #                 print "PROBLEM"
        #             else:
        #                 sales2 = liste.iloc[len(liste)-1]["Weekly_Sales"]

        # #############################
    
        last_week, sales2 = last_year_previous_week(row, date, Xttype_curr, Xspec_curr, ok, ok2, perfect, holiday)

        if last_week is None:
            continue
        
        Weekly_Sales1.append(sales)
        Temperature1.append(last_year["Temperature"])
        Fuel_Price1.append(last_year["Fuel_Price"])
        MarkDown11.append(last_year["MarkDown1"])
        MarkDown21.append(last_year["MarkDown2"])
        MarkDown31.append(last_year["MarkDown3"])
        MarkDown41.append(last_year["MarkDown4"])
        MarkDown51.append(last_year["MarkDown5"])
        CPI1.append(last_year["CPI"])
        Unemployment1.append(last_year["Unemployment"])

        Size.append(row["Size"])
        IsHoliday.append(row["IsHoliday"])
        Type.append(row["Type"])

        Weekly_Sales2.append(sales2)
        Temperature2.append(last_week["Temperature"])
        Fuel_Price2.append(last_week["Fuel_Price"])
        MarkDown12.append(last_week["MarkDown1"])
        MarkDown22.append(last_week["MarkDown2"])
        MarkDown32.append(last_week["MarkDown3"])
        MarkDown42.append(last_week["MarkDown4"])
        MarkDown52.append(last_week["MarkDown5"])
        CPI2.append(last_week["CPI"])
        Unemployment2.append(last_week["Unemployment"])
        
        Date.append(row["Date"])
        Store.append(row["Store"])
        Dept.append(row["Dept"])
        Weekly_Sales3.append(row["Weekly_Sales"])
        Temperature3.append(row["Temperature"])
        Fuel_Price3.append(row["Fuel_Price"])
        MarkDown13.append(row["MarkDown1"])
        MarkDown23.append(row["MarkDown2"])
        MarkDown33.append(row["MarkDown3"])
        MarkDown43.append(row["MarkDown4"])
        MarkDown53.append(row["MarkDown5"])
        CPI3.append(row["CPI"])
        Unemployment3.append(row["Unemployment"])


    # Creation dictionary
    dic["Size"] = Size
    dic["Store"] = Store
    dic["Dept"] = Dept
    dic["Date"] = Date
    dic["IsHoliday"] = IsHoliday
    dic["Type"] = Type
    dic["Weekly_Sales1"] = Weekly_Sales1
    dic["Temperature1"] = Temperature1
    dic["Fuel_Price1"] = Fuel_Price1
    dic["MarkDown11"] = MarkDown11
    dic["MarkDown21"] = MarkDown21
    dic["MarkDown31"] = MarkDown31
    dic["MarkDown41"] = MarkDown41
    dic["MarkDown51"] = MarkDown51
    dic["CPI1"] = CPI1
    dic["Unemployment1"] = Unemployment1
    dic["Weekly_Sales2"] = Weekly_Sales2
    dic["Temperature2"] = Temperature2
    dic["Fuel_Price2"] = Fuel_Price2
    dic["MarkDown12"] = MarkDown12
    dic["MarkDown22"] = MarkDown22
    dic["MarkDown32"] = MarkDown32
    dic["MarkDown42"] = MarkDown42
    dic["MarkDown52"] = MarkDown52
    dic["CPI2"] = CPI2
    dic["Unemployment2"] = Unemployment2
    dic["Weekly_Sales3"] = Weekly_Sales3
    dic["Temperature3"] = Temperature3
    dic["Fuel_Price3"] = Fuel_Price3
    dic["MarkDown13"] = MarkDown13
    dic["MarkDown23"] = MarkDown23
    dic["MarkDown33"] = MarkDown33
    dic["MarkDown43"] = MarkDown43
    dic["MarkDown53"] = MarkDown53
    dic["CPI3"] = CPI3
    dic["Unemployment3"] = Unemployment3

    X = pd.DataFrame(dic)
    return X

def createTestDataset():
    dicoTest = runningTests2()
    Xorigin = runningTests()
    liste_datasets = []
    
    for Type in Types:
        X = dicoTest[Type]
        XType = Xorigin[(Xorigin.Type == Type)]
        XType = cleanBigDataset2(XType)
        Xtotal = pd.concat([X, XType])
        XttypeNoSpec = Xtotal[(Xtotal.IsHoliday == False)]
        # date = dt.datetime.strptime(X.iloc[0]["Date"], "%Y-%m-%d").date()
        # date = date - dt.timedelta(days=365) - dt.timedelta(days=15)
        # Xtotal = Xtotal[Xtotal.Date > str(date)]
        Xtotal = Xtotal.sort(["Store", "Dept", "Date"])
        for spec in Specs:
            start_time = time.time()
            Xtypespec= X[(X.IsHoliday == spec)]
            Xtypespec = Xtypespec.sort(["Store", "Dept", "Date"])
            # Xttype = Xtotal[Xtotal.Type == Type]
            if spec:
                Xtotaltypespec = Xtotal[Xtotal.IsHoliday == True]
            # X2 = createDataset(X, XttypeNoSpec, X, spec, perfect)
                Xnew = createDataset(Xtypespec, XttypeNoSpec, Xtotaltypespec, spec, False)
                liste_datasets.append(Xnew)
            else:
                Xtotaltypespec = Xtotal[Xtotal.IsHoliday == False]
                Xnew = createDataset(Xtypespec, XttypeNoSpec, Xtotaltypespec, spec, False)
                liste_datasets.append(Xnew)
            print time.time() - start_time, "seconds"

    Xres = pd.concat(liste_datasets)
    Xres.to_csv("test_dataset.csv", index=False)
    return Xres

def createTrainDataset():
    Xorigin = runningTests()
    perfect = True
    for Type in Types:
        XttypeNoSpec = cleanBigDataset(Xorigin, Type)
        for spec in Specs:
            start_time = time.time()
            X = originalDatasetCleaning(Xorigin, Type, spec)
            X2 = createDataset(X, XttypeNoSpec, X, spec, perfect)
            str_spec = ""
            if spec:
                str_spec = "spec"
            X2.to_csv(Type+str_spec+"values.csv",
                      index=False)
            print time.time() - start_time, "seconds"


def originalDatasetCleaning(Xfinal, Type="A", spec=True): #, labels
    indices = np.where((Xfinal["Type"]==Type) & (Xfinal["IsHoliday"]==spec))[0]
    Xtest = Xfinal.iloc[indices, :]

    # Xtest = Xtest.drop('Type', 1)
    # Xtest = Xtest.drop('IsHoliday', 1)

    mean1 = stats.nanmean(Xtest["MarkDown1"])
    mean2 = stats.nanmean(Xtest["MarkDown2"])
    mean3 = stats.nanmean(Xtest["MarkDown3"])
    mean4 = stats.nanmean(Xtest["MarkDown4"])
    mean5 = stats.nanmean(Xtest["MarkDown5"])

    Xtest["MarkDown1"].fillna(value=mean1, inplace=True)
    Xtest["MarkDown2"].fillna(value=mean2, inplace=True)
    Xtest["MarkDown3"].fillna(value=mean3, inplace=True)
    Xtest["MarkDown4"].fillna(value=mean4, inplace=True)
    Xtest["MarkDown5"].fillna(value=mean5, inplace=True)

    # X = createDataset(Xtest)

    return Xtest

def cleanBigDataset(Xfinal, Type="A"):
    indices = np.where((Xfinal["Type"]==Type) & (Xfinal["IsHoliday"]==False))[0]
    Xtest = Xfinal.iloc[indices, :]

    # Xtest = Xtest.drop('Type', 1)
    # Xtest = Xtest.drop('IsHoliday', 1)

    mean1 = stats.nanmean(Xtest["MarkDown1"])
    mean2 = stats.nanmean(Xtest["MarkDown2"])
    mean3 = stats.nanmean(Xtest["MarkDown3"])
    mean4 = stats.nanmean(Xtest["MarkDown4"])
    mean5 = stats.nanmean(Xtest["MarkDown5"])

    Xtest["MarkDown1"].fillna(value=mean1, inplace=True)
    Xtest["MarkDown2"].fillna(value=mean2, inplace=True)
    Xtest["MarkDown3"].fillna(value=mean3, inplace=True)
    Xtest["MarkDown4"].fillna(value=mean4, inplace=True)
    Xtest["MarkDown5"].fillna(value=mean5, inplace=True)

    # X = createDataset(Xtest)

    return Xtest

def cleanBigDataset2(Xfinal):
    #indices = np.where((Xfinal["Type"]==Type))[0]
    #Xtest = Xfinal.iloc[indices, :]
    Xtest = Xfinal

    # Xtest = Xtest.drop('Type', 1)
    # Xtest = Xtest.drop('IsHoliday', 1)

    mean1 = stats.nanmean(Xtest["MarkDown1"])
    mean2 = stats.nanmean(Xtest["MarkDown2"])
    mean3 = stats.nanmean(Xtest["MarkDown3"])
    mean4 = stats.nanmean(Xtest["MarkDown4"])
    mean5 = stats.nanmean(Xtest["MarkDown5"])

    Xtest["MarkDown1"].fillna(value=mean1, inplace=True)
    Xtest["MarkDown2"].fillna(value=mean2, inplace=True)
    Xtest["MarkDown3"].fillna(value=mean3, inplace=True)
    Xtest["MarkDown4"].fillna(value=mean4, inplace=True)
    Xtest["MarkDown5"].fillna(value=mean5, inplace=True)

    # X = createDataset(Xtest)

    return Xtest

def runningTests():
    X_stores = storesdata('stores.csv')
    X_features = featuresdata('features.csv')
    X_train = traindata('train.csv')
    # X_test = testdata('test.csv')
    Xfinal = join(X_train, X_features, X_stores)
    return Xfinal
    # Xtest = testnormalweekA(Xfinal)
    # return Xtest

def runningTests2(dataset="test.csv"):
    X_stores = storesdata('stores.csv')
    X_features = featuresdata('features.csv')
    X_test = testdata(dataset)
    dico_tests = {}
    # X_test = testdata('test.csv')
    Xfinal = join(X_test, X_features, X_stores)
    for Type in Types:
        listeXs = []
        for spec in Specs:
            indices = np.where((Xfinal["Type"]==Type) & (Xfinal["IsHoliday"]==spec))[0]
            Xtest = Xfinal.iloc[indices, :]

            # Xtest = Xtest.drop('Type', 1)
            # Xtest = Xtest.drop('IsHoliday', 1)

            mean1 = stats.nanmean(Xtest["MarkDown1"])
            mean2 = stats.nanmean(Xtest["MarkDown2"])
            mean3 = stats.nanmean(Xtest["MarkDown3"])
            mean4 = stats.nanmean(Xtest["MarkDown4"])
            mean5 = stats.nanmean(Xtest["MarkDown5"])
            meanCPI = stats.nanmean(Xtest["CPI"])
            meanUnem = stats.nanmean(Xtest["Unemployment"])

            Xtest["MarkDown1"].fillna(value=mean1, inplace=True)
            Xtest["MarkDown2"].fillna(value=mean2, inplace=True)
            Xtest["MarkDown3"].fillna(value=mean3, inplace=True)
            Xtest["MarkDown4"].fillna(value=mean4, inplace=True)
            Xtest["MarkDown5"].fillna(value=mean5, inplace=True)
            Xtest["CPI"].fillna(value=meanCPI, inplace=True)
            Xtest["Unemployment"].fillna(value=meanUnem, inplace=True)
            Xtest["Weekly_Sales"] = None
            listeXs.append(Xtest)
        newXfinal = pd.concat(listeXs)
        dico_tests[Type] = newXfinal
    return dico_tests

def loadTestDataset():
    print "Loading dataset..."
    X = testdata("test_dataset.csv")
    X = X.drop("Weekly_Sales3", 1)
    print "Cleaning dataset..."
    X = fillNanTestDataset(X)
    return X


def fillNanTestDataset(Xfinal):
    listeXs = []
    Xfinal = Xfinal.sort(["Store", "Dept", "Date"])
    Xfinal = Xfinal.reset_index(drop=True)

    paires = []
    for i in range(len(Xfinal)):
        row = Xfinal.iloc[i]
        Store = row.get("Store")
        Dept = row.get("Dept")
        new_item = [Store, Dept]
        if new_item not in paires:
            paires.append(new_item)
    print "Paires created..."
    for item in paires:
        for spec in Specs:
            indices = np.where((Xfinal["Store"]==item[0]) & (Xfinal["Dept"]==item[1]) & (Xfinal["IsHoliday"]==spec))[0]
            Xtest = Xfinal.iloc[indices, :]

            mean1 = stats.nanmean(Xtest["Weekly_Sales1"])
            mean2 = stats.nanmean(Xtest["Weekly_Sales2"])

            Xtest["Weekly_Sales1"].fillna(value=mean1, inplace=True)
            Xtest["Weekly_Sales2"].fillna(value=mean2, inplace=True)
            listeXs.append(Xtest)
    newXfinal = pd.concat(listeXs)
    listeXs = []
    for Type in Types:
        for spec in Specs:
            indices = np.where((newXfinal["Type"]==Type) & (newXfinal["IsHoliday"]==spec))[0]
            Xtest = newXfinal.iloc[indices, :]

            mean1 = stats.nanmean(Xtest["Weekly_Sales1"])
            mean2 = stats.nanmean(Xtest["Weekly_Sales2"])

            Xtest["Weekly_Sales1"].fillna(value=mean1, inplace=True)
            Xtest["Weekly_Sales2"].fillna(value=mean2, inplace=True)
            listeXs.append(Xtest)
    newXfinal2 = pd.concat(listeXs)
    return newXfinal2

def dumb(Xfinal):
    listeXs = []
    for Type in Types:
        for spec in Specs:
            indices = np.where((Xfinal["Type"]==Type) & (Xfinal["IsHoliday"]==spec))[0]
            Xtest = Xfinal.iloc[indices, :]

            mean1 = stats.nanmean(Xtest["Weekly_Sales1"])
            mean2 = stats.nanmean(Xtest["Weekly_Sales2"])

            Xtest["Weekly_Sales1"].fillna(value=mean1, inplace=True)
            Xtest["Weekly_Sales2"].fillna(value=mean2, inplace=True)
            listeXs.append(Xtest)
    newXfinal2 = pd.concat(listeXs)
    return newXfinal2

def predic(X, dico=None):
    #X = X.sort(["Type", "Dept", "Date"])
    with open("predictions.csv", "w") as myFile:
        myFile.write("Id,Weekly_Sales\n")
        for Type in Types:
            for spec in Specs:
                str_spec = ""
                if spec:
                    str_spec = "spec"
                print "Predicting ", Type, str_spec
                clf = dico[Type+str_spec]
                Xts = X[(X.Type == Type) & (X.IsHoliday == spec)]
                Xts = Xts.reset_index(drop=True)
                identifiers = []
                for i in range(len(Xts)):
                    row = Xts.iloc[i]
                    identifier = str(row.get("Store"))+"_"+str(row.get("Dept"))+"_"+row.get("Date")
                    identifiers.append(identifier)
                Xts = Xts.drop(["Store", "Dept", "Date", "Type", "IsHoliday"], 1)
                Res = clf.predict(Xts)
                for i in range(len(Xts)):
                    myFile.write(identifiers[i]+','+str(Res[i])+"\n")
    return

def predic2(X, dico=None):
    X = X.sort(["Store", "Dept", "Date"])
    with open("predictions.csv", "w") as myFile:
        myFile.write("Id,Weekly_Sales\n")
        for item in X.iterrows():
            # print str(item)
            item = item[1]
            if item.get("IsHoliday") == True:
                clf = dico[item.get("Type")+"spec"]
            else:
                clf = dico[item.get("Type")]
            identifier = str(item.get("Store"))+"_"+str(item.get("Dept"))+"_"+item.get("Date")
            item = item.drop("Store")
            item = item.drop("Dept")
            item = item.drop("Date")
            item = item.drop("Type")
            item = item.drop("IsHoliday")
            # print str(item)
            res = clf.predict(item)
            myFile.write(identifier+','+str(res)+"\n")
    return

if __name__ == '__main__':
    X_stores = storesdata('stores.csv')
    X_features = featuresdata('features.csv')
    X_train, labels = traindata('train.csv')
    X_test = testdata('test.csv')
    
    X, labels = data(filename)
    
    clf = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                             C=1.0, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=None)
    
    X = preprocessing.scale(X)	
    X_test = preprocessing.scale(X_test)
    
    createSub(clf, X, labels, X_test)
