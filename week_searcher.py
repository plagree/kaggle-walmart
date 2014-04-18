#!/usr/bin/env python
#-*-coding: utf-8 -*-

import datetime as dt


def last_year_same_week(row, date, Xttype_curr, Xspec_curr, ok, ok2, perfect, holiday):
    ### LAST YEAR ###
    if holiday:
        datel = str(date - dt.timedelta(days=365) - dt.timedelta(days=110))
        dater = str(date - dt.timedelta(days=365) + dt.timedelta(days=11))
        if ok:
            liste = Xspec_curr[(Xspec_curr.Date > datel) & (Xspec_curr.Date < dater)]
        else:
            liste = []
    else:
        datel = str(date - dt.timedelta(days=365) - dt.timedelta(days=100))
        dater = str(date - dt.timedelta(days=365) + dt.timedelta(days=5))
        if ok2:
            liste = Xttype_curr[(Xttype_curr.Date > datel) & (Xttype_curr.Date < dater)]
        else:
            liste = []
    
    if len(liste) == 0:
        if perfect:
            return None, None
        last_year = row
    else:
        last_year = liste.iloc[len(liste)-1]
    
    ###### WEEKLY_SALES MANAGEMENT FOR TESTING ######
    sales = last_year["Weekly_Sales"]
    if sales is None:
        if holiday:
            datel = str(date - dt.timedelta(days=730) - dt.timedelta(days=110))
            dater = str(date - dt.timedelta(days=730) + dt.timedelta(days=11))
            if ok:
                liste = Xspec_curr[(Xspec_curr.Date > datel) & (Xspec_curr.Date < dater)]
            else:
                liste = []
        else:
            datel = str(date - dt.timedelta(days=730) - dt.timedelta(days=100))
            dater = str(date - dt.timedelta(days=730) + dt.timedelta(days=4))
            if ok2:
                liste = Xttype_curr[(Xttype_curr.Date > datel) & (Xttype_curr.Date < dater)]
            else:
                liste = []
        if len(liste) == 0:
            print "PROBLEM"
        else:
            sales = liste.iloc[len(liste)-1]["Weekly_Sales"]
    if sales is None:
        print "PROBLEM"
    ##################################################
    return last_year, sales


def last_year_previous_week(row, date, Xttype_curr, Xspec_curr, ok, ok2, perfect, holiday):
    ## LAST YEAR LAST WEEK ##
    datel = str(date - dt.timedelta(days=365) - dt.timedelta(days=28))
    dater = str(date - dt.timedelta(days=365) - dt.timedelta(days=4))
    if ok2:
        liste = Xttype_curr[(Xttype_curr.Date > datel) & (Xttype_curr.Date < dater)]
    else:
        liste = []
    if len(liste) == 0:
        if perfect:
            return None, None
        if holiday:
            last_week = Xttype_curr.iloc[0]
        else:
            last_week = row
    else:
        last_week = liste.iloc[len(liste)-1]

    sales2 = last_week["Weekly_Sales"]

    ####### NEW WORK ############
    if sales2 is None:
        datel = str(date - dt.timedelta(days=730) - dt.timedelta(days=100))
        dater = str(date - dt.timedelta(days=730) - dt.timedelta(days=4))
        if ok2:
            liste = Xttype_curr[(Xttype_curr.Date > datel) & (Xttype_curr.Date < dater)]
        else:
            liste = []
        if len(liste) != 0:
            sales2 = liste.iloc[len(liste)-1]["Weekly_Sales"]
        else:
            print "PROBLEM"
    return last_week, sales2
