#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:03:28 2020

@author: carlosdelafuente
"""

import os
import numpy as np
import pandas as pd
import csv
import pickle
import spacy
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
    
def testing_model_svm(filenames_in, path_files_in, v, dataset):
    
    # path = os.getcwd() + '/data/test/test_dev/'

    # test_files = setup.get_test(path)
    
    test_files = [os.path.join(path_files_in, 'dev_' + f_name + '.txt') for f_name in filenames_in]
    
    model_filename = os.getcwd() + '/models/' + dataset + '/svc_model.pkl'
    with open(model_filename, 'rb') as file_model:  
        svc_model = pickle.load(file_model)    

    for test_file in test_files:
        with open(test_file, 'r', encoding='utf-8') as infile:
    
            data_test = pd.read_csv(infile, delimiter = "\t", header = None, dtype=object,
                                  index_col = False, encoding = 'UTF-8', quoting=csv.QUOTE_NONE)
            
    #            data_test = data_test.dropna()
                
            X_test = v.transform(data_test.to_dict('records'))
            y_test_pred_svc = svc_model.predict(X_test)   
                    
            data_test['cue_tag'] = y_test_pred_svc.tolist()
            outfile = os.getcwd() + '/outputs/' + 'output_svc_' + test_file.split('/')[-1]
    
            data_test.to_csv(outfile, sep = '\t', header = None, index = False,
                             quoting=csv.QUOTE_NONE, encoding = 'UTF-8')
            
    return


def testing_model_crf(filenames_in, path_files_in, dataset):
    
    # path = os.getcwd() + '/data/test/test_dev/'    
    # test_files = setup.get_test(path)

    test_files = [os.path.join(path_files_in, 'test_' + f_name + '.txt') for f_name in filenames_in]
    
    # Test Cue
    
    model_filename = os.getcwd() + '/models/' + dataset + '/crf_cue_model.pkl'
    with open(model_filename, 'rb') as file_model:  
        crf_model = pickle.load(file_model)
    
    for test_file in test_files:       
        with open(test_file, 'r', encoding='utf-8') as infile:
            
            col_names = ['domain', 'sentence', 'token', 'word', 'lemma', 
                        'PoS', 'PoS_type']
            
            data_test = pd.read_csv(infile, delimiter = "\t", header = None,
                      names = col_names, encoding = 'UTF-8', dtype=object,
                      skip_blank_lines=False, quoting=csv.QUOTE_NONE)
                    
            data_test.insert(0,'inx', [i for i in range(len(data_test))])
            
            agg_func = lambda s: [(w, l, p, i) for w, l, p, i in zip(
                s["word"].values.tolist(),
                s["lemma"].values.tolist(),
                s["PoS"].values.tolist(),
                s["inx"].values.tolist())]
                        
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
              
            outfile = os.getcwd() + '/outputs/' + 'output_crf_cue_' + test_file.split('/')[-1]
            
            data_test.to_csv(outfile, sep = '\t', header = None, index = False,
                             quoting=csv.QUOTE_NONE, encoding = 'UTF-8')
            
    # Test Scopes
    
    model_filename = os.getcwd() + '/models/' + dataset + '/crf_sco_model.pkl'
    with open(model_filename, 'rb') as file_model:  
        crf_model = pickle.load(file_model)
    
    for test_file in test_files:       
        with open(test_file, 'r', encoding='utf-8') as infile:
            
            col_names = ['domain', 'sentence', 'token', 'word', 'lemma', 
                        'PoS', 'PoS_type']
            
            data_test = pd.read_csv(infile, delimiter = "\t", header = None,
                      names = col_names, encoding = 'UTF-8', dtype=object,
                      skip_blank_lines=False, quoting=csv.QUOTE_NONE)
                    
            data_test.insert(0,'inx', [i for i in range(len(data_test))])
            
            agg_func = lambda s: [(w, l, p, i) for w, l, p, i in zip(
                s["word"].values.tolist(),
                s["lemma"].values.tolist(),
                s["PoS"].values.tolist(),
                s["inx"].values.tolist())]
                        
            grouped_data_test = data_test.groupby(['domain','sentence']).apply(agg_func)
            frases_test = [f for f in grouped_data_test]
            
            X_test = [training.sent2features(f) for f in frases_test]
            
            y_test_pred_crf = crf_model.predict(X_test)
            
            frases_test = [item for elem in frases_test for item in elem]
            y_test_pred_crf_list = [item for elem in y_test_pred_crf for item in elem]
            
            p = 0 
            for t in frases_test:
                data_test.loc[data_test['inx'] == t[3], 'sco_tag'] = y_test_pred_crf_list[p]
                p+=1
            
            data_test = data_test.drop('inx', 1)
              
            outfile = os.getcwd() + '/outputs/' + 'output_crf_scope_' + test_file.split('/')[-1]
            
            data_test.to_csv(outfile, sep = '\t', header = None, index = False,
                             quoting=csv.QUOTE_NONE, encoding = 'UTF-8')            
                                    
    return


def testing_model_NER(filenames_in, path_files_in, dataset):
    
    # Esta función hace el testing de CUE y SCOPE secuencialmente

    test_files = [os.path.join(path_files_in, 'test_' + f_name + '.txt') for f_name in filenames_in]
    
    # Test Cue
    
    model_spc_output = os.getcwd() + '/models/' + dataset + '/spacy_model'
        
    nlp2 = spacy.load(model_spc_output)
                 
    column_names = ['domain', 'sentence', 'token', 'word', 'lemma', 
                    'PoS', 'PoS_type']
    
    for test_file in test_files:         
        
        # leo el archivo para generar un dataframe de trabajo con el # de cols ajustado    
        d_test = pd.read_csv(test_file, delimiter = "\t", encoding = 'UTF-8', dtype=object,
                       header = None, names = column_names, quoting=csv.QUOTE_NONE,
                       skip_blank_lines=False)    
        
        agg_func = lambda s: [(inx, word) for inx, word in zip(s.index.tolist(), 
                               s['word'].values.tolist())]
        
        grouped_d_test = d_test.groupby(['domain','sentence']).apply(agg_func)
                                
        data_test_tag = pd.DataFrame()
        for f in grouped_d_test:
            frase = ' '.join(word[1] for word in f)
            doc = nlp2(frase)        
            fragmento_d_test_tag = pd.DataFrame([(e.text, e.ent_iob_, word[0], e.ent_type_) for e, word in zip(doc, f)])
            fragmento_d_test_tag[3].replace(r'^\s*$', 'O-Cue', regex=True, inplace = True)
            fragmento_d_test_tag = fragmento_d_test_tag.append(pd.Series(), ignore_index = True)
            data_test_tag = pd.concat([data_test_tag, fragmento_d_test_tag], axis=0)
               
        # Append blank lines
            
        new_rows = []
        for i in range(len(data_test_tag)):
            if pd.isna(data_test_tag.iloc[i][2]):
                new_rows.append(data_test_tag.iloc[i-1][2]+1)
            else:
                new_rows.append(0)
                
        data_test_tag[2].fillna(0, inplace= True)
        data_test_tag[2] = data_test_tag[2] + new_rows
        data_test_tag = data_test_tag.sort_values(by=[2])
        data_test_tag.fillna('', inplace = True)
        
        
        d_test['bio_tag'] = data_test_tag[3].values
        
        f_test = os.getcwd() + '/outputs/output_ner_cue_' + test_file.split('/')[-1]
        
        d_test.to_csv(f_test, sep = '\t', index=False, quoting=csv.QUOTE_NONE, header= None, encoding = 'UTF-8')


    # Test Scopes
    
    model_spc_output = os.getcwd() + '/models/' + dataset + '/spacy_scope_model'
        
    nlp2 = spacy.load(model_spc_output)
                 
    column_names = ['domain', 'sentence', 'token', 'word', 'lemma', 
                    'PoS', 'PoS_type']
    
    for test_file in test_files:         
    
        # leo el archivo para generar un dataframe de trabajo con el # de cols ajustado    
        d_test = pd.read_csv(test_file, delimiter = "\t", encoding = 'UTF-8', dtype=object,
                       header = None, names = column_names, quoting=csv.QUOTE_NONE,
                       skip_blank_lines=False)    
             
        agg_func = lambda s: [(inx, word) for inx, word in zip(s.index.tolist(), 
                               s['word'].values.tolist())]
        
        grouped_d_test = d_test.groupby(['domain','sentence']).apply(agg_func)
                                
        data_test_tag = pd.DataFrame()
        for f in grouped_d_test:
            frase = ' '.join(word[1] for word in f)
            doc = nlp2(frase)        
            fragmento_d_test_tag = pd.DataFrame([(e.text, e.ent_iob_, word[0], e.ent_type_) for e, word in zip(doc, f)])
            fragmento_d_test_tag[3].replace(r'^\s*$', 'O-Sco', regex=True, inplace = True)
            fragmento_d_test_tag = fragmento_d_test_tag.append(pd.Series(), ignore_index = True)
            data_test_tag = pd.concat([data_test_tag, fragmento_d_test_tag], axis=0)
               
        # Append blank lines
            
        new_rows = []
        for i in range(len(data_test_tag)):
            if pd.isna(data_test_tag.iloc[i][2]):
                new_rows.append(data_test_tag.iloc[i-1][2]+1)
            else:
                new_rows.append(0)
                
        data_test_tag[2].fillna(0, inplace= True)
        data_test_tag[2] = data_test_tag[2] + new_rows
        data_test_tag = data_test_tag.sort_values(by=[2])
        data_test_tag.fillna('', inplace = True)
        
        
        d_test['bio_tag'] = data_test_tag[3].values
        
        f_test = os.getcwd() + '/outputs/output_ner_scope_' + test_file.split('/')[-1]
        
        d_test.to_csv(f_test, sep = '\t', index=False, quoting=csv.QUOTE_NONE, header= None, encoding = 'UTF-8')
    
    return
