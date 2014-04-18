"""

Kaggle Store Sales Forecasting.
__author__ : Paul Lagree
__date__ : 26/03/14 - 

"""

import sys
import time
import pandas as pd
import numpy as np
import datetime as dt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
import  scipy.stats as stats
import sklearn.linear_model as lm
from week_searcher import *
from fitting import fit_regressors
from test import test

Types = ["A", "B", "C"]
Specs = [True, False]


def load_data(filename):
    X = pd.read_table(filename, sep=',', warn_bad_lines=True,
                      error_bad_lines=True)
    return X

def join(Xtrain, Xfeatures, Xstores):
    Xmerged = pd.merge(Xtrain, Xfeatures, how='inner')
    Xfinal = pd.merge(Xmerged, Xstores, how='inner')
    return Xfinal

def createDataset(Xtypespec, Xttype, Xspec, holiday, perfect=False):
    Xtypespec = Xtypespec.sort(["Store", "Dept", "Date"])
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
    # Week year -1 week -1 (not holidays)
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

        last_year, sales = last_year_same_week(row, date, Xttype_curr, Xspec_curr,
                                               ok, ok2, perfect, holiday)

        if last_year is None:
            continue

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

def createMyTestingDataset():
    dicoTest = loadAndCleanTestingDataset()
    Xorigin = loadTrainingDataset()
    liste_datasets = []
    
    for Type in Types:
        X = dicoTest[Type]
        XType = Xorigin[(Xorigin.Type == Type)]
        XType = cleanMarkDowns(XType)
        Xtotal = pd.concat([X, XType])
        XttypeNoSpec = Xtotal[(Xtotal.IsHoliday == False)]
        Xtotal = Xtotal.sort(["Store", "Dept", "Date"])
        for spec in Specs:
            start_time = time.time()
            Xtypespec= X[(X.IsHoliday == spec)]
            Xtypespec = Xtypespec.sort(["Store", "Dept", "Date"])
            if spec:
                Xtotaltypespec = Xtotal[Xtotal.IsHoliday == True]
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

def createMyTrainingDataset():
    Xorigin = loadTrainingDataset()
    perfect = True
    for Type in Types:
        XttypeNoSpec = getAndCleanTypeNotHoliday(Xorigin, Type)
        for spec in Specs:
            start_time = time.time()
            X = getAndCleanGivenTypeSpec(Xorigin, Type, spec)
            X2 = createDataset(X, XttypeNoSpec, X, spec, perfect)
            str_spec = ""
            if spec:
                str_spec = "spec"
            X2.to_csv(Type+str_spec+"values.csv",
                      index=False)
            print time.time() - start_time, "seconds"


def getAndCleanGivenTypeSpec(Xfinal, Type="A", spec=True):
    indices = np.where((Xfinal["Type"]==Type) & (Xfinal["IsHoliday"]==spec))[0]
    Xtest = Xfinal.iloc[indices, :]

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

    return Xtest

def getAndCleanTypeNotHoliday(Xfinal, Type="A"):
    indices = np.where((Xfinal["Type"]==Type) & (Xfinal["IsHoliday"]==False))[0]
    Xtest = Xfinal.iloc[indices, :]

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

    return Xtest

def cleanMarkDowns(Xfinal):
    Xtest = Xfinal

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

    return Xtest

def loadTrainingDataset():  # Supposed to be already cleaned
    X_stores = load_data('stores.csv')
    X_features = load_data('features.csv')
    X_train = load_data('train.csv')
    Xfinal = join(X_train, X_features, X_stores)
    return Xfinal

def loadAndCleanTestingDataset(dataset="test.csv"):
    X_stores = load_data('stores.csv')
    X_features = load_data('features.csv')
    X_test = load_data(dataset)
    dico_tests = {}
    Xfinal = join(X_test, X_features, X_stores)
    for Type in Types:
        listeXs = []
        for spec in Specs:
            indices = np.where((Xfinal["Type"]==Type) & (Xfinal["IsHoliday"]==spec))[0]
            Xtest = Xfinal.iloc[indices, :]

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

def loadMyTestingDataset():
    print "Loading dataset..."
    X = load_data("test_dataset.csv")
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

def predic(X, dico=None):
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

if __name__ == '__main__':
    """ Two parameters to this script:
        --create : create every dataset (for each type, spec, and test)
        --predict : load created datasets, predict results of the
                    testing dataset and write results on predictions.csv
    """
    args = sys.argv
    if len(args) not in [2, 3]:
        print "Usage: python walmart.py\nParameters available (at least 1): "+ \
            "--create --predict --test"

    for arg in args:
        if arg == "--create":
            # Training and Testing dataset creation
            createMyTrainingDataset()
            createMyTestingDataset()
        elif arg == "--predict":
            # Prediction and wrting in a CSV file of
            # results on testing dataset
            dico = fit_regressors()
            X = loadMyTestingDataset()
            predic(X, dico)
        elif arg == "--test":
            # Tests on training datasets (90% for training, 10% for measure)
            test()
