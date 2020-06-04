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

import setup
import training

# Módulo de test

# Test modelo SVM con dev file. Sólo para AJUSTES

def testing_dev_svm(data_dev, v, dataset):
    
    model_filename = os.getcwd() + '/models/' + dataset + '/svc_model.pkl'
        
    with open(model_filename, 'rb') as file_model:  
        svc_model = pickle.load(file_model)
     
    X_dev = v.transform(data_dev.to_dict('records'))
    
    y_dev_pred_svc = svc_model.predict(X_dev)
        
    return(y_dev_pred_svc, svc_model)


# Test modelo CRF con dev para AJUSTES. Uso el modelo CRF optimizado
    
def testing_dev_crf(data_dev, dataset):
    
    data_dev.insert(0,'inx', [i for i in range(len(data_dev))])
    
    agg_func = lambda s: [(w, l, p, i) for w, l, p, i in zip(
        s["word"].values.tolist(),
        s["lemma"].values.tolist(),
        s["PoS"].values.tolist(),
        s["inx"].values.tolist())]
    
    grouped_data_dev = data_dev.groupby(['domain','sentence']).apply(agg_func)
    frases_dev = [f for f in grouped_data_dev]
               
    X_dev = [training.sent2features(f) for f in frases_dev]
    
    model_filename = os.getcwd() + '/models/' + dataset + '/crf_opt_model.pkl'
        
    with open(model_filename, 'rb') as file_model:  
        crf_model = pickle.load(file_model)
    
    y_dev_pred_crf = crf_model.predict(X_dev)
    
    frases_dev = [item for elem in frases_dev for item in elem]
    y_dev_pred_crf_list = [item for elem in y_dev_pred_crf for item in elem]
            
    p = 0 
    for t in frases_dev:
        data_dev.loc[data_dev['inx'] == t[3], 'bio_tag'] = y_dev_pred_crf_list[p]
        p+=1
    
    data_dev = data_dev.drop('inx', 1)

    return(data_dev['bio_tag'], crf_model)
    

# Test sobre los conjuntos de test que generan Output files
    
def testing_model_svm(v, dataset):
    
    path = os.getcwd() + '/data/test/'

    test_files = setup.get_test(path)

    for test_file in test_files:
        with open(os.path.join(path, test_file), 'r', encoding='utf-8') as infile:

            data_test = pd.read_csv(infile, delimiter = "\t", header = None,
                                  encoding = 'UTF-8', index_col=[0])
            
#            data_test = data_test.dropna()
            
            model_filename = os.getcwd() + '/models/' + dataset + '/svc_model.pkl'
            with open(model_filename, 'rb') as file_model:  
                svc_model = pickle.load(file_model)
    
            X_test = v.transform(data_test.to_dict('records'))
            y_test_pred_svc = svc_model.predict(X_test)   
                    
            data_test['cue_tag'] = y_test_pred_svc.tolist()
            outfile = os.getcwd() + '/outputs/' + 'output_svc_' + test_file
    
            data_test.to_csv(outfile, sep = '\t', header = None, encoding = 'UTF-8')

    return


def testing_model_crf(dataset):
    
    path = os.getcwd() + '/data/test/'
    
    test_files = setup.get_test(path)
 
    for test_file in test_files:
        with open(os.path.join(path, test_file), 'r', encoding='utf-8') as infile:
            
            col_names = ['domain', 'sentence', 'token', 'word', 'lemma', 
                        'PoS', 'PoS_type']
        
            data_test = pd.read_csv(infile, delimiter = "\t", header = None,
                                  encoding = 'UTF-8', names = col_names, 
                                  skip_blank_lines=False)
          
            data_test.insert(0,'inx', [i for i in range(len(data_test))])
            
            agg_func = lambda s: [(w, l, p, i) for w, l, p, i in zip(
                s["word"].values.tolist(),
                s["lemma"].values.tolist(),
                s["PoS"].values.tolist(),
                s["inx"].values.tolist())]
            
            model_filename = os.getcwd() + '/models/' + dataset + '/crf_model.pkl'
            with open(model_filename, 'rb') as file_model:  
                crf_model = pickle.load(file_model)
            
            grouped_data_test = data_test.groupby(['domain','sentence']).apply(agg_func)
            frases_test = [f for f in grouped_data_test]
            
            X_test = [training.sent2features(f) for f in frases_test]
            
            y_test_pred_crf = crf_model.predict(X_test)
            
            frases_test = [item for elem in frases_test for item in elem]
            y_test_pred_crf_list = [item for elem in y_test_pred_crf for item in elem]
            
            p = 0 
            for t in frases_test:
                data_test.loc[data_test['inx'] == t[3], 'bio_tag'] = y_test_pred_crf_list[p]
                p+=1
            
            data_test = data_test.drop('inx', 1)
              
            outfile = os.getcwd() + '/outputs/' + 'output_crf_' + test_file
            
            data_test.to_csv(outfile, sep = '\t', header = None, encoding = 'UTF-8')
    return
