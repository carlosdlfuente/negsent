#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:09:00 2020

@author: carlosdelafuente
"""

import os
#import numpy as np
import pandas as pd
#import pickle
#from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction import DictVectorizer
#from sklearn.linear_model import Perceptron
#from sklearn.naive_bayes import MultinomialNB
#from sklearn import svm
#from sklearn.metrics import classification_report, confusion_matrix
##from sklearn.model_selection import cross_val_score, cross_val_predict
#from sklearn_crfsuite import CRF, scorers, metrics
#from sklearn_crfsuite.metrics import flat_classification_report


# Setup

def join_data(dset):    # une todos los ficheros de entrada

    files = []
    path = os.getcwd() + '/data/' + dset + '/'
    join_file = os.path.join(path, (dset + '_dataset.join'))

    for f_name in os.listdir(path):
        if f_name.endswith('.txt'):
            files.append(f_name)
    with open(join_file, 'w') as outfile:
        for f in files:
            with open(os.path.join(path, f), 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
    outfile.close()
    return      


def get_test(path):   # lee todos los archivos de test

    files = []    
    for f_name in os.listdir(path):
        if f_name.endswith('.txt'):
            files.append(f_name)
    return(files) 
    

def get_data(dset):     # carga el dataset para el entrenamiento en un df

    bio_dataset = os.getcwd() + '/data/' + dset + '/' + dset + '_dataset.join'
    
    # leo el archivo para calcular el # de negaciones y asi ajustar el # de columnas   
    with open(bio_dataset, 'r') as temp_f:
        col_count = [len(l.split("\t"))-1 for l in temp_f.readlines()]    
    negaciones = int((max(col_count)-7) / 3)
    column_name_neg = []
    for i in range(negaciones):
        column_name_neg+=['cue_'+ str(i), 
                          'sco_'+ str(i), 
                          'eve_'+ str(i)]          
    column_names = ['domain', 'sentence', 'token', 'word', 'lemma', 
                    'PoS', 'PoS_type'] + column_name_neg

    # leo el archivo para generar un dataframe de trabajo con el # de cols ajustado    
    df = pd.read_csv(bio_dataset, delimiter = "\t", encoding = 'UTF-8', 
                   header = None, quotechar = "'", names = column_names, 
                   skip_blank_lines=False)
#    df.fillna('-')
#    df['bio_tag'] = '-'
#    df['sco_tag'] = '-'
    df['id_sentence'] = df['domain'] + '_' + df['sentence'].map(str)
    column_names = df.columns.tolist()
    column_names = column_names[-1:] + column_names[:-1]
    df = df[column_names]
    
    return(df, negaciones)
    
    
def set_bio_frase(bio_frase, negs):     # anota cada tolen con BIO format
    negs_frase = []
    negs_sco_frase = []
    cue_B = []
    sco_B = []
    for i in range(negs):
        negs_frase.append((bio_frase['cue_' + str(i)].nunique()) - 1)
        negs_sco_frase.append((bio_frase['sco_' + str(i)].nunique()) - 1)
        cue_B.append(False)
        sco_B.append(False)
    bio_tag = ['O-Cue'] * len(bio_frase)
    sco_tag = ['O-Sco'] * len(bio_frase)
    for t in range(len(bio_frase)):
        for n in range(negs):
            cue = 'cue_' + str(n)
            if (bio_frase.iloc[t][cue] == '***'):
                continue
            elif (bio_frase.iloc[t][cue] != '-'):
                if negs_frase[n] > 1:
                    if cue_B[n] == False:
                        bio_tag[t] ='B-Cue'
                        negs_frase[n] -= 1
                        cue_B[n] = True
                    else:
                        bio_tag[t] ='I-Cue'
                        negs_frase[n] -= 1
                else:
                    if cue_B[n] == False:
                        bio_tag[t] ='B-Cue'
                    else:
                        bio_tag[t] ='I-Cue'
        for n in range(negs):
            sco = 'sco_' + str(n)  
            if (bio_frase.iloc[t]['cue_0'] == '***'):
                continue           
            elif (bio_frase.iloc[t][sco] != '-'):
                if negs_sco_frase[n] > 1:
                    if sco_B[n] == False:
                        sco_tag[t] ='B-Sco'
                        negs_sco_frase[n] -= 1
                        sco_B[n] = True
                    else:
                        sco_tag[t] ='I-Sco'
                        negs_sco_frase[n] -= 1
                else:
                    if sco_B[n] == False:
                        sco_tag[t] ='B-Sco'
                    else:
                        sco_tag[t] ='I-Sco'                
    return(bio_tag, sco_tag)
                    
    
# anota todos los arcihvos dev para usarlos como test durante la fase de ajuste

def tag_dev():
    path_dev_test = os.getcwd() + '/data/dev/dev_test/'
    path_dev_tagged_test = os.getcwd() + '/data/dev/dev_test/tagged/'
    
    for f_dev in os.listdir(path_dev_test):
        if f_dev.endswith('.txt'):
            # leo el archivo para calcular el # de negaciones y asi ajustar el # de columnas   
            with open(os.path.join(path_dev_test, f_dev), 'r') as temp_f:
                col_count = [len(l.split("\t"))-1 for l in temp_f.readlines()]    
            negaciones = int((max(col_count)-7) / 3)
            column_name_neg = []
            for i in range(negaciones):
                column_name_neg+=['cue_'+ str(i), 
                                  'sco_'+ str(i), 
                                  'eve_'+ str(i)]          
            column_names = ['domain', 'sentence', 'token', 'word', 'lemma', 
                            'PoS', 'PoS_type'] + column_name_neg
        
            # leo el archivo para generar un dataframe de trabajo con el # de cols ajustado    
            df = pd.read_csv(os.path.join(path_dev_test, f_dev), delimiter = "\t", 
                    encoding = 'UTF-8', header = None, quotechar = "'", 
                    names = column_names, skip_blank_lines=False)
#            df = df.fillna('-')
#            df['bio_tag'] = '-'
#            df['sco_tag'] = '-'
            df['id_sentence'] = df['domain'] + '_' + df['sentence'].map(str)
            column_names = df.columns.tolist()
            column_names = column_names[-1:] + column_names[:-1]
            df = df[column_names]
            
            # Ajusta bio_tag frase a frase
            tag_data = pd.DataFrame()
            frases = df['id_sentence'].unique()
            for f in frases:
                if pd.notna(f):
                    frase = df.loc[df['id_sentence'] == f].copy()
                    frase.fillna('-', inplace = True)
                    frase['bio_tag'], frase['sco_tag'] = set_bio_frase(frase, negaciones)
                    frase = frase.append(pd.Series(), ignore_index = True)
                    tag_data = tag_data.append(frase)
            # salvo archivo bio_taggeado a disco
            tag_data.to_csv(os.path.join(path_dev_tagged_test, f_dev), sep = '\t',                         
                            encoding = 'UTF-8')
    return

def genera_dev_test():
    path_dev_test = os.getcwd() + '/data/dev/dev_test/test'
    path_dev_tagged_test = os.getcwd() + '/data/dev/dev_test/tagged/'
     
    for f_dev in os.listdir(path_dev_tagged_test):
        if f_dev.endswith('.txt'):
            with open(os.path.join(path_dev_tagged_test, f_dev), 'r') as temp_f:
                dev_test = pd.read_csv(temp_f, delimiter = "\t", 
                      encoding = 'UTF-8', index_col=[0], skip_blank_lines=False)
                dev_test = dev_test[['domain', 'sentence', 'token', 'word', 'lemma', 
                    'PoS', 'PoS_type']]
                dev_test.to_csv(os.path.join(path_dev_test, f_dev), sep = '\t', encoding = 'UTF-8')
    return

def genera_dev_gold():
    path_dev_test = os.getcwd() + '/data/dev/dev_test/test'
    path_dev_gold_test = os.getcwd() + '/data/dev/dev_test/gold/'
    path_dev_tagged_test = os.getcwd() + '/data/dev/dev_test/tagged/'
    
    for f_dev, f_test in zip(os.listdir(path_dev_tagged_test), os.listdir(path_dev_test)) :
        if f_dev.endswith('.txt'):
            with open(os.path.join(path_dev_tagged_test, f_dev), 'r') as temp_f_tagged:
                    tagged_dev = pd.read_csv(temp_f_tagged, delimiter = "\t", 
                        encoding = 'UTF-8', index_col=[0], skip_blank_lines=False)
                    
            with open(os.path.join(path_dev_test, f_test), 'r') as temp_f_test:
                    test_dev = pd.read_csv(temp_f_test, delimiter = "\t",   
                        encoding = 'UTF-8', index_col=[0], skip_blank_lines=False)
                    
                    test_gold = test_dev
                    test_gold['bio_tag'] = tagged_dev['bio_tag']
            test_gold.to_csv(os.path.join(path_dev_gold_test, f_test), sep = '\t', 
                    header = None, encoding = 'UTF-8')
    return

def quita_header_test():
    path_dev_test = os.getcwd() + '/data/dev/dev_test/test'
    
    for f_test in os.listdir(path_dev_test):
        if f_test.endswith('.txt'):                   
            with open(os.path.join(path_dev_test, f_test), 'r') as temp_f_test:
                    test_dev = pd.read_csv(temp_f_test, delimiter = "\t",   
                        encoding = 'UTF-8', index_col=[0], skip_blank_lines=False)
            test_dev.to_csv(os.path.join(path_dev_test, f_test), sep = '\t', 
                    header = None, encoding = 'UTF-8')
    return
