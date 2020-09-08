#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 19:05:25 2020

@author: carlosdelafuente
"""

# import sys
# import codecs
import os
import pickle
import pandas as pd
import numpy as np
import string
import csv

tecnica = 'baseline'

def comentarios_neg(grouped_data):  
    
    # Comentarios con negación 'NOT_'; elimina además signos de puntuación
    
    comentarios_negs = []
    for comentario_neg in grouped_data:
        frases = []
        for index, token in enumerate(comentario_neg):
            if token[3] in ['B-Sco', 'I-Sco']:
                w = list(token)
                w[1] = 'NOT_' + w[1]
                token = tuple(w)
            comentario_neg[index] = token
            if token[1] in string.punctuation: 
                continue
            else: 
                frases.append(token[1])
        dict_frase = {}
        dict_frase.update({
            'nombre_comentario' : comentario_neg[0][0],
            'comentario': frases,
            'sentiment': 1 if 'yes' in comentario_neg[0][0] else 0
            })
        comentarios_negs.append(dict_frase)        
    data_comentarios_neg = pd.DataFrame(comentarios_negs, columns = ['nombre_comentario', 'comentario', 'sentiment']) 
    return(data_comentarios_neg) 


def comentarios_raw(grouped_data):
    
    # Comentarios raw; elimina además signos de puntuación.
    
    comentarios = []
    for comentario in grouped_data:
        frases = []
        for token in comentario:
            if token[1] in string.punctuation: 
                continue
            else: 
                frases.append(token[1])
        dict_frase = {}
        dict_frase.update({
            'nombre_comentario' : comentario[0][0],
            'comentario': frases,
            'sentiment': 1 if 'yes' in comentario[0][0] else 0
            })
        comentarios.append(dict_frase)        
    data_comentarios = pd.DataFrame(comentarios, columns = ['nombre_comentario', 'comentario', 'sentiment'])  
    return(data_comentarios)  

def dummy(doc):
    return doc


dataset_test = 'test'
dataset_train = 'train_dev'
    
filenames = ['coches', 'lavadoras', 'hoteles', 'moviles', 'libros', 'musica', 
             'ordenadores', 'peliculas']

if tecnica == 'crf':
    path_test_files = os.getcwd() + '/outputs/'
    test_files = [os.path.join(path_test_files, 'output_crf_cue_scope_test_' + f_name + '.txt') for f_name in filenames]
    model_filename = os.getcwd() + '/models/opinion/' + dataset_train + '/' + 'modelo_neg_opinion.pkl'
elif tecnica == 'ner':
    path_test_files = os.getcwd() + '/outputs/'
    test_files = [os.path.join(path_test_files, 'output_ner_cue_scope_test_' + f_name + '.txt') for f_name in filenames]
    model_filename = os.getcwd() + '/models/opinion/' + dataset_train + '/' + 'modelo_neg_opinion.pkl'
else:
    path_test_files = os.getcwd() + '/data/test/'
    test_files = [os.path.join(path_test_files, 'test_' + f_name + '.txt') for f_name in filenames]
    model_filename = os.getcwd() + '/models/opinion/' + dataset_train + '/' + 'modelo_raw_opinion.pkl'
    

with open(model_filename, 'rb') as file_model:  
    modelo = pickle.load(file_model)
    
output  = pd.DataFrame()
opinion = pd.DataFrame()

for datafile_test, nombre_dominio in zip(test_files, filenames):  

    dominio = []
    nombre_comentario = []
    
    # datafile_test = os.getcwd() + '/data/test/test_join.txt'

    if tecnica == 'baseline':
        data_test = pd.read_csv(datafile_test, delimiter = '\t', encoding = 'UTF-8', 
                                header = None, index_col = False, dtype=object,
                                usecols = [0, 4], skip_blank_lines=False, quoting=csv.QUOTE_NONE)
        
        data_test.columns = ['domain', 'lemma']
        
        # data_test['lemma'] = data_test['lemma'].str.lower()
            
        agg_func = lambda s: [(domain, lemma) for domain, lemma
                              in zip(s['domain'].values.tolist(),                            
                                     s['lemma'].values.tolist())]
        
        grouped_data_test = data_test.groupby(['domain']).apply(agg_func) 
        data_comentarios_test = comentarios_raw(grouped_data_test)
    else:
        data_test = pd.read_csv(datafile_test, delimiter = '\t', encoding = 'UTF-8', 
                                header = None, index_col = False, dtype=object,
                                usecols = [0, 4, 7, 8], skip_blank_lines=False, quoting=csv.QUOTE_NONE)
        
        data_test.columns = ['domain', 'lemma', 'bio_tag', 'sco_tag']
        
        # data_test['lemma'] = data_test['lemma'].str.lower()
            
        agg_func = lambda s: [(domain, lemma, bio_tag, sco_tag) for domain, lemma, bio_tag, sco_tag
                              in zip(s['domain'].values.tolist(),                            
                                     s['lemma'].values.tolist(),
                                     s['bio_tag'].values.tolist(),
                                     s['sco_tag'].values.tolist())]
        
        grouped_data_test = data_test.groupby(['domain']).apply(agg_func)        
        data_comentarios_test = comentarios_neg(grouped_data_test)
    
    data_comentarios_test['sentiment'].value_counts()
    
    y_test = modelo.predict(data_comentarios_test['comentario'])
    
    for i in range(len(data_comentarios_test)): dominio.append(data_comentarios_test['nombre_comentario'][i].split('_')[0])
    for i in range(len(data_comentarios_test)): nombre_comentario.append('_'.join(data_comentarios_test['nombre_comentario'][i].rsplit('_')[1:]))
    y_test = np.where(y_test == 1, 'positive', 'negative')
   
    output = pd.concat([pd.DataFrame(nombre_comentario), pd.DataFrame(dominio), pd.DataFrame(y_test)], axis = 1)
    
    # for i in range(len(output)): output.loc[i, 'nombre_comentario'] = '_'.join(output.loc[i, 'nombre_comentario'].rsplit('_')[1:])
    
    opinion = pd.concat([opinion, output], ignore_index=True)
    
# para guardar todos los test uno por uno
# output_data = os.getcwd() + '/outputs/opinion/' + dataset_test + '/opinion_' + datafile_test.split('/')[-1]

output_data = os.getcwd() + '/outputs/opinion/' + dataset_test + '/' + tecnica + '_opinion.txt'
opinion.to_csv(output_data, sep = '\t', index=False, quoting=csv.QUOTE_NONE, header= None, encoding = 'UTF-8')
        