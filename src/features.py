################################################
# Scripts to turn raw data into features for exploration and cluster analysis
################################################
import os
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

def checkDF(df):
    """function for checking of any missing, dupliction
    and abnormal distribution of dataset and return the
    the total number of missing and duplication 
    ======================================================
    ARGUMENTS: the input raw dataframe 
    ======================================================
    RETURN: Total number of missing, Total number of duplication
            and wether there is an abnormality or normality
    """
    cols_missing_no = []
    for col in df.columns.tolist():
        if df[col].isna().sum() != 0:
            print(f"The total missing values of our dataframe is: n\ {df[col].isna().sum()}")
            cols_missing_no.append(col)
        else:
            print("The dataframe column have no any missing values")
        if df.duplicated().sum() !=0:
            print(f"The total missing values of our dataframe is: n\ {df.duplicated().sum()}")
        else:
            print("There are no duplication values observed")

def split_data(df,path, train_size = 0.8, test_size= 0.2):
    """_summary_

    Args:
        data (_type_): _description_
        train_size (_type_): _description_
        test_size (_type_): _description_
        stratify (_type_): _description_
    """
    
    X = df.drop(['Districts'], axis = 1)
    y = df['Districts']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= train_size, test_size = test_size,
                                                        stratify = y, random_state = 42)
    dtrain = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis = 1, join = 'inner')
    dtest = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis = 1, join = 'inner')
    train = dtrain.to_csv(path + 'train.csv', index = False)
    test = dtest.to_csv(path+ 'test.csv', index = False)

def check_index(x):
    dups = ["water_prc", "elect_prc", "tel_prc"]
    for dups in (x[x.index.duplicated()]):
        print('The duplications are found')
        x = x[~x.index.duplicated()]
        return x
    else:
        print('No duplication is found')
    return x

def clean_data(df):
    """this function removes the null and duplicated data points 
    if there is any checked by our preceded function
    =============================================================
    ARGUMENTS:
    =============================================================
    RETURN: cleaned and preprocessed data called interim_data
    """
    dups = df.index.duplicated()
    if dups.sum() > 0:
        df = df[~df.index.duplicated()]
    if df.val.isnull().values.any():
        df.val.fillna(0, inplace=True)
