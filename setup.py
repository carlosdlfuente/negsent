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
    join_file = os.path.join(path, (dset + '_dataset.txt'))

    for f_name in os.listdir(path):
        if f_name.endswith('subtaskA.txt'):
            files.append(f_name)
    with open(join_file, 'w') as outfile:
        for f in files:
            with open(os.path.join(path, f), 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
    outfile.close()
    return      


def get_data(dset):     # carga el dataset para el entrenamiento en un df
    
    bio_dataset = os.getcwd() + '/data/' + dset + '/' + dset + '_dataset.txt'
    
    # leo el archivo para calcular el # de negaciones y asi ajustar el # de columnas
    
    with open(bio_dataset, 'r') as temp_f:
        col_count = [len(l.split("\t"))-1 for l in temp_f.readlines()]    
    negaciones = int((max(col_count)-7) / 3)
    column_name_neg = []
    for i in range(negaciones):
        column_name_neg+=['cue_'+ str(i), 
                          'cue_'+ str(i) + '1-', 
                          'cue_'+ str(i) + '2-']          
    column_names = ['domain', 'sentence', 'token', 'word', 'lemma', 
                    'PoS', 'PoS_type'] + column_name_neg

    # leo el archivo para generar un dataframe de trabajo con el # de cols ajustado
    
    df = pd.read_csv(bio_dataset, delimiter = "\t", encoding = 'UTF-8', 
                   header = None, dtype='unicode', names = column_names)
    df = df.fillna('-')
    df['bio_tag'] = '-'
    
    df['id_sentence'] = df['domain'] + '_' + df['sentence'].map(str)
    column_names = df.columns.tolist()
    column_names = column_names[-1:] + column_names[:-1]
    df = df[column_names]

    return(df, negaciones)
    
    
def set_bio_frase(bio_frase, negs):     # anota cada tolen con BIO format
    bio_tag = ['O'] * len(bio_frase)
    for t in range(len(bio_frase)):
        for n in range(negs):
            cue = 'cue_' + str(n)
            if (bio_frase.iloc[t][cue] == '***'):
                continue
            elif (bio_frase.iloc[t][cue] != '-'):        
                bio_tag[t] ='B'
    indx_bes = [i for i, x in enumerate(bio_tag) if x == 'B']
    indx_sign = [i for i, x in enumerate(list(bio_frase['word'])) if x in '?¿¡!,.;()[]']
    indx = indx_bes + indx_sign
    indx.sort()
    for j in range(len(indx)-1):
        if bio_tag[indx[j]] == 'B':
            for k in range(indx[j],indx[j+1]):
                if bio_tag[k] == 'O':
                    bio_tag[k] = 'I'                  
    return(bio_tag)