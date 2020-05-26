#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:07:24 2020

@author: carlosdelafuente
"""

import os
import numpy as np
#import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
#from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn_crfsuite import CRF
#from sklearn_crfsuite.metrics import flat_classification_report


def word2features(sent, i):
    word = sent[i][0]
    lema = sent[i][1]
    postag = sent[i][2]
#    bio_tag = sent[i][3]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'lemma' : lema,
        'postag': postag,
    }
    if i > 0:
        lemma1 = sent[i-1][1]
        lemma2 = sent[i-2][1]
        lemma3 = sent[i-3][1]
        postag1 = sent[i-1][2]
        postag2 = sent[i-2][2]
        postag3 = sent[i-3][2]
        features.update({
            '-1:lemma' : lemma1,
            '-2:lemma' : lemma2,
            '-3:lemma' : lemma3,
            '-1:postag' : postag1,
            '-2:postag' : postag2,
            '-3:postag' : postag3,
        })
    else:
        features['BOS'] = True

    if i < len(sent)-3:
        lemma1 = sent[i+1][1]
        lemma2 = sent[i+2][1]
        lemma3 = sent[i+3][1]
        postag1 = sent[i+1][2]
        postag2 = sent[i+2][2]
        postag3 = sent[i+3][2]        
        features.update({
            '+1:lemma' : lemma1,
            '+2:lemma' : lemma2,
            '+3:lemma' : lemma3,
            '+1:postag' : postag1,
            '+2:postag' : postag2,
            '+3:postag' : postag3,
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [bio_label for token, lemma, PoS, bio_label in sent]

# Clasificadores: MLP, NB y SVM y CRF.

# MLP

def training_per(X_train, y_train, X_test, y_test, classes, dataset):
    per = Perceptron(verbose=10, n_jobs=-1, max_iter=5)
    per.partial_fit(X_train, y_train.values.ravel(), classes)
    y_pred = per.predict(X_test)
    
    model_filename = os.getcwd() + '/models/' + dataset + '/per_model.pkl'
    with open(model_filename, 'wb') as file_model:  
        pickle.dump(per, file_model)
    return(y_pred)

# Naive Bayes

def training_nb(X_train, y_train, X_test, y_test, classes, dataset):
    nb = MultinomialNB(alpha=0.01)
    nb.partial_fit(X_train, y_train, classes)
    y_pred = nb.predict(X_test)
      
    model_filename = os.getcwd() + '/models/' + dataset + '/nb_model.pkl'
    with open(model_filename, 'wb') as file_model:  
        pickle.dump(nb, file_model)
    return(y_pred)

# SVM

def training_svc(X_train, y_train, X_test, y_test, classes, dataset):
    svc = svm.LinearSVC(C=10, max_iter = 2000)
    svc.fit(X_train, y_train.values.ravel())
    y_pred = svc.predict(X_test)
    
    model_filename = os.getcwd() + '/models/' + dataset + '/svc_model.pkl'
    with open(model_filename, 'wb') as file_model:  
        pickle.dump(svc, file_model)
    return(y_pred)


# CRF

def training_crf(data, agg_func, dataset):
      
    grouped_df = data.groupby('id_sentence').apply(agg_func)
    frases = [f for f in grouped_df]
    
    X = np.array([sent2features(f) for f in frases])
    y = np.array([sent2labels(f) for f in frases])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, 
                                                        random_state=123)
    
    crf = CRF(algorithm='lbfgs',
              c1=0.1,
              c2=0.1,
              max_iterations=100,
              all_possible_transitions=True,
              verbose=True)
    crf.fit(X_train, y_train)
    
    y_pred = crf.predict(X_test)
    
    model_filename = os.getcwd() + '/models/' + dataset + '/crf_model.pkl'
    with open(model_filename, 'wb') as file_model:  
        pickle.dump(crf, file_model)
    return(y_test, y_pred)


