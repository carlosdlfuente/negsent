#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:03:28 2020

@author: carlosdelafuente
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix

import training


# Módulo de test

# Test modelo SVM

def testing_svm(data_test, v, dataset):
    model_filename = os.getcwd() + '/models/' + dataset + '/svc_model.pkl'
        
    with open(model_filename, 'rb') as file_model:  
        svc_model = pickle.load(file_model)
     
#    data_test_gold = data_test.drop(columns = 'bio_tag')
    X_test_gold = v.transform(data_test.to_dict('records'))
    
    y_test_pred_svc = svc_model.predict(X_test_gold)
    
    np.save(os.getcwd() + '/outputs/' + 'output_svc_test', y_test_pred_svc.tolist())
    
    return(y_test_pred_svc)


# Test modelo CRF

def testing_crf(data_test, agg_func, dataset):
    model_filename = os.getcwd() + '/models/' + dataset + '/crf_model.pkl'
        
    with open(model_filename, 'rb') as file_model:  
        crf_model = pickle.load(file_model)
    
    grouped_data_test = data_test.groupby('id_sentence').apply(agg_func)
    frases_test = [f for f in grouped_data_test]
    
    X_test_gold = np.array([training.sent2features(f) for f in frases_test])
    y_test_gold = np.array([training.sent2labels(f) for f in frases_test])
    
    y_test_pred_crf = crf_model.predict(X_test_gold)
    
    return(y_test_gold, y_test_pred_crf)

#labels = list(crf.classes_)
#labels.remove('O')

