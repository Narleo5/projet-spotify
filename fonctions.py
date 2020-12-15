#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:04:09 2020

@author: Nadir
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import scipy.stats as st
import warnings
warnings.filterwarnings(action='ignore')
from numpy import array
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import svm

def select_2_features(df, cat1, cat2):
    #df contient toutes les features et la variable catégorielle 
    features = ["loudness","instrumentalness","tempo","acousticness" ,"energy","valence","liveness","danceability", "speechiness"]
    bf1, bf2 = '', ''
    best_score = 0
    Y = df["Catégorie"].apply(lambda x: int(x==cat1))
    for feat1 in features:
        for feat2 in features:
            if feat1 != feat2: 
                X = df[[feat1, feat2]]
                X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
                reg = svm.LinearSVC(C=1).fit(X_train, Y_train)
                score = reg.score(X_test, Y_test)
                if score > best_score:
                    bf1, bf2 = feat1, feat2
                    best_score = score
    return([bf1, bf2, best_score])



def plot_best_hypp_lin(df,cat1,cat2):
    
    [feat1, feat2, score] = select_2_features(df,cat1,cat2)
    
    X = df[[feat1, feat2]]
    Y = df["Catégorie"].apply(lambda x: int(x==cat1))
    
    fig, ax = plt.subplots()
    clf2 = svm.LinearSVC(C=1).fit(X, Y)

  
    w = clf2.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf2.intercept_[0]) / w[1]

    x_min, x_max = X.iloc[:, 0].min() - 0.2 , X.iloc[:, 0].max() + 0.1
    y_min, y_max = X.iloc[:, 1].min()-0.1, X.iloc[:, 1].max() + 0.1
    xx2, yy2 = np.meshgrid(np.arange(x_min, x_max, .2),
                         np.arange(y_min, y_max, .2))
    Z = clf2.predict(np.c_[xx2.ravel(), yy2.ravel()])

    Z = Z.reshape(xx2.shape)
    ax.contourf(xx2, yy2, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=Y, cmap=plt.cm.coolwarm, s=25)
    ax.plot(xx,yy)

    ax.axis([x_min, x_max,y_min, y_max])
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    return(f'score sur test set: {score}')
    
grid = {'bootstrap': [True, False],
    'max_depth': [2, 5, 10, 20, 50, 80, 100, None],
    'max_features': ['auto', 'sqrt'],
    'n_estimators': [5, 20, 50, 100, 200, 500]}
    
random_grid = {'bootstrap': [True, False],
    'max_depth': [2,5, 10, 20, 30, 50, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [5, 20, 50, 100, 200, 500, 800, 1000, 1200, 1500, 1800, 2000]}
    
    
def rf_grid(X_train,y_train, X_test, y_test):
    grid_search_rf = GridSearchCV(RandomForestClassifier(), grid, cv=4, n_jobs=-1, verbose = 3)
    grid_search_rf.fit(X_train, y_train)
    print ("Score final : ", round(grid_search_rf.score(X_test, y_test) *100,4), " %")
    print ("Meilleurs parametres: ", grid_search_rf.best_params_)
        
        
def rf_rdgrid(X_train,y_train, X_test, y_test, n_iter):
    rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = random_grid, n_iter=n_iter, cv = 4, verbose=2, n_jobs = -1)
    rf_random.fit(X_train, y_train)
    print ("Score final : ", round(rf_random.score(X_test, y_test) *100,4), " %")
    print ("Meilleurs parametres: ", rf_random.best_params_)
    

def plot_SVM(df,model):
    #df contient toutes les features et la variable catégorielle 
        
    label_encoder = preprocessing.LabelEncoder()
    features = ["loudness","instrumentalness","tempo","acousticness" ,"energy","valence","liveness","danceability", "speechiness"]
    bf1, bf2 = '', ''
    best_score = 0
    values = array(df["Catégorie"])
    Y = label_encoder.fit_transform(values)
    for feat1 in features:
        for feat2 in features:
            if feat1 != feat2: 
                X = df[[feat1, feat2]]
                X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
                reg = model.fit(X_train, Y_train)
                score = reg.score(X_test, Y_test)
                if score > best_score:
                    bf1, bf2 = feat1, feat2
                    best_score = score
   
    X = df[[bf1, bf2]]
    X = X.to_numpy()
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    
    h = .01
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    
    plt.subplots_adjust(wspace=0.4, hspace=0.4) 
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.PuBuGn, edgecolors='grey')
    plt.xlabel(bf1)
    plt.ylabel(bf2)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.show()
    
    return(f'score sur test set: {score}')


def knn_grid(X_train,y_train, X_test, y_test):
    leaf_size = list(range(1,50,2))
    n_neighbors = list(range(1,12))
    p=[1,2]
    
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    grid_knn=GridSearchCV(KNeighborsClassifier(), hyperparameters, cv=4, verbose = 2, n_jobs=-1)
    grid_knn.fit(X_train,y_train)
    
    print ("Score final : ", round(grid_knn.score(X_test, y_test) *100,4), " %")
    print ("Meilleurs parametres: ", grid_knn.best_params_)


def rl_grid(X_train,y_train, X_test, y_test):
    param_log={'penalty':['l2','elasticnet','none'], 'C':np.arange(0.5,1.5,0.1),'solver':['newton-cg', 'lbfgs'],'max_iter':[200,250,300]}
    grid_search_logi = GridSearchCV(LogisticRegression(), param_log, cv=4, verbose = 2, n_jobs=-1)
    grid_search_logi.fit(X_train, y_train)
    print ("Score final : ", round(grid_search_logi.score(X_test, y_test) *100,4), " %")
    print ("Meilleurs parametres: ", grid_search_logi.best_params_)
    

    
    