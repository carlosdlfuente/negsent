#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 19:34:21 2020

@author: carlosdelafuente
"""

import os

import es_core_news_md
import random
import spacy
import collections

import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt

import csv
import string

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer, 
from sklearn.metrics import roc_curve, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score, classification_report
from sklearn.model_selection import learning_curve

# import nltk
from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer
# from nltk.tokenize import TweetTokenizer


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


def comentarios_neg(grouped_data):  
    
    # Comentarios con negación 'NOT_'; elimina además signos de puntuación
    
    comentarios_negs = []
    for comentario_neg in grouped_data:
        frases = []
        for index, token in enumerate(comentario_neg):
            if token[2] in ['B-Sco', 'I-Sco']:
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

def get_negaciones(data):
    negaciones = []
    for i in range(len(data)):
        if (data.iloc[i]['bio_tag'] in ['B-Cue', 'I-Cue']):
            negaciones.append(data.iloc[i]['word'])
    counts = collections.Counter(negaciones)
    top_negaciones = sorted(counts, key=lambda x: -counts[x])    
    negaciones_list = list(top_negaciones)[:15]
    return(negaciones_list)


def join_data(dset):    
    
    # une todos los ficheros de entrada 'train' o 'dev'

    files = []
    path = os.getcwd() + '/outputs/opinion/'
    join_file = os.path.join(path, ('output_ner_cue_scope_' + dset + '_join.txt'))

    for f_name in os.listdir(path):
        if f_name.endswith('.txt'):
            files.append(f_name)
    with open(join_file, 'w') as outfile:
        for f in files:
            with open(os.path.join(path, f), 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
    outfile.close()
    return    


def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)        

    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    result = {'auc': auc, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
    return result


def get_roc_curve(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_proba)
    return fpr, tpr


dataset = 'train_dev'

# join_data(dataset)
# datafile = os.getcwd() + '/outputs/opinion/output_ner_cue_scope_' + dataset + '_join.txt'
# data = pd.read_csv(datafile, delimiter = '\t', encoding = 'UTF-8', header = None,
#                     index_col=None, skip_blank_lines=False)
# data = data[[0, 1, 3, 7, 8]]
# data.columns = ['domain', 'sentence', 'word', 'bio_tag', 'sco_tag']

datafile = os.getcwd() + '/data/' + dataset + '/bio/' + dataset + '_dataset_bio.txt'
data = pd.read_csv(datafile, delimiter = '\t', encoding = 'UTF-8', dtype=object,
                    index_col=None, skip_blank_lines=False, quoting=csv.QUOTE_NONE)
data = data[['domain', 'sentence', 'word', 'bio_tag', 'sco_tag']]


data['word'] = data['word'].str.lower()

# agg_func = lambda s: [(domain, sentence, lemma, bio_tag, sco_tag) for domain, sentence, lemma, bio_tag, sco_tag
#                       in zip(s['domain'].values.tolist(),
#                              s['sentence'].values.tolist(),
#                              s['lemma'].values.tolist(), 
#                              s['bio_tag'].values.tolist(),
#                              s['sco_tag'].values.tolist(),)]

agg_func = lambda s: [(domain, word, sco_tag) for domain, word, sco_tag
                      in zip(s['domain'].values.tolist(),
                             s['word'].values.tolist(), 
                             s['sco_tag'].values.tolist())]

grouped_data = data.groupby(['domain']).apply(agg_func)

data_comentarios = comentarios_raw(grouped_data)
data_comentarios['sentiment'].value_counts()

data_comentarios_neg = comentarios_neg(grouped_data)
data_comentarios_neg['sentiment'].value_counts()


train, test = train_test_split(data_comentarios, test_size=0.2, random_state=123)
X_train = train['comentario'].values
X_test = test['comentario'].values
y_train = train['sentiment']
y_test = test['sentiment']

# Preprocesamiento de StopWords

# Opción 1.- Uso na lista de stopwords personalizada donde elimino las Top 15 negacioones
    
negaciones_habituales = get_negaciones(data)
es_stopwords = set(stopwords.words("spanish")) 
new_es_stopwords = [x for x in es_stopwords if x not in negaciones_habituales]

# Opción 2.- Usar  max_df=1.0, min_df=0.6

def dummy(doc):
    return doc

# vectorizer = CountVectorizer(
#     analyzer = 'word',
#     tokenizer = dummy,
#     preprocessor = dummy,
#     # lowercase = True,
#     ngram_range=(1, 5),     #trigramas
#     max_df=1.0, min_df=0.4, max_features=None,
#     # stop_words = new_es_stopwords,
#     token_pattern=None)

vectorizer = TfidfVectorizer(
    stop_words = new_es_stopwords,
    tokenizer = dummy,
    preprocessor = dummy,
    # sublinear_tf = True,
    # strip_accents = 'unicode',
    analyzer = 'word',
    # token_pattern = r'\w{2,}',  #vectorize 2-character words or more
    ngram_range = (1, 3))
    # max_df=1.0, min_df=0.6, max_features=None)

kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

# Pipeline RAW

np.random.seed(123)

pipeline_svm = make_pipeline(vectorizer, 
                            SVC(probability=True, 
                                kernel="linear",
                                class_weight="balanced"))

grid_svm = GridSearchCV(pipeline_svm,
                    param_grid = {'svc__C': [0.01, 0.1, 1, 100]}, 
                    cv = kfolds,
                    scoring="roc_auc",
                    verbose=1,   
                    n_jobs=-1) 

grid_svm.fit(X_train, y_train)
grid_svm.score(X_test, y_test)

grid_svm.best_params_
grid_svm.best_score_

results_raw = report_results(grid_svm.best_estimator_, X_test, y_test)
results_raw

roc_svm = get_roc_curve(grid_svm.best_estimator_, X_test, y_test)

# Pipeline NEGS

train, test = train_test_split(data_comentarios_neg, test_size=0.2, random_state=123)
X_train = train['comentario'].values
X_test = test['comentario'].values
y_train = train['sentiment']
y_test = test['sentiment']

np.random.seed(123)

grid_svm_neg = GridSearchCV(pipeline_svm,
                    param_grid = {'svc__C': [0.01, 0.1, 1, 100]}, 
                    cv = kfolds,
                    scoring="roc_auc",
                    verbose=1,   
                    n_jobs=-1) 

grid_svm_neg.fit(X_train, y_train)
grid_svm_neg.score(X_test, y_test)

grid_svm_neg.best_params_
grid_svm_neg.best_score_

grid_svm_neg.fit(X_train, y_train)
grid_svm_neg.score(X_test, y_test)

grid_svm_neg.best_params_
grid_svm_neg.best_score_

results_neg = report_results(grid_svm_neg.best_estimator_, X_test, y_test)
results_neg

# Agrupa resultados en una tabla 

resultados_validacion = pd.concat([pd.DataFrame(results_raw.items()), 
                                   pd.DataFrame(results_neg.values())], axis = 1)

resultados_validacion.columns = ['metrica', 'raw_data', 'NEG_data']

roc_svm_neg = get_roc_curve(grid_svm_neg.best_estimator_, X_test, y_test)

# Plot clasificadores

fpr, tpr = roc_svm
fpr_neg, tpr_neg = roc_svm_neg

plt.figure(figsize=(14,8))

plt.plot(fpr, tpr, color="red", label = 'AUC Baseline')
plt.plot(fpr_neg, tpr_neg, color="blue", label = 'AUC Neg')

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc curve')
plt.legend(prop={'size':13}, loc='lower right')
plt.show()


''' Learning curves ayudan a visualizar el efecto del numero de observaciones sobre
las métricas de rendimiento. No hay problema de sesgo o varianza. El rensimiento 
del modelo aumenta cuando tenemos más datos.'''

'''OJO:  Usar grid_svm o grid_svm_neg y cambiar X_train, y_train'''

# scoring="roc_auc"
# scoring="accuracy"

# Ref: https://chrisalbon.com/machine_learning/model_evaluation/plot_the_learning_curve/

train_sizes, train_scores, test_scores = \
    learning_curve(grid_svm_neg.best_estimator_, X_train, y_train, cv=10, n_jobs=-1, 
                   scoring="roc_auc", train_sizes=np.linspace(.1, 1.0, 10), random_state=1)

def plot_learning_curve(X, y, train_sizes, train_scores, test_scores, title='', ylim=None, figsize=(14,8)):

    plt.figure(figsize=figsize)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="lower right")
    return plt


plot_learning_curve(X_train, y_train, train_sizes, 
                    train_scores, test_scores, ylim=(0.7, 1.01), figsize=(14,6))
plt.show()


# Test
    
dataset_test = 'test'

path_test_files = os.getcwd() + '/data/test/'
    
filenames = ['coches', 'lavadoras', 'hoteles', 'moviles', 'libros', 'musica', 
             'ordenadores', 'peliculas']

test_files = [os.path.join(path_test_files, 'test_' + f_name + '.txt') for f_name in filenames]

opinion = pd.DataFrame()

for datafile_test, nombre_dominio in zip(test_files, filenames):  
    
    data_test = pd.read_csv(datafile_test, delimiter = '\t', encoding = 'UTF-8', 
                            header = None, index_col = False,
                            usecols = [0, 4], skip_blank_lines=False)
    
    data_test.columns = ['domain', 'lemma']
        
    agg_func = lambda s: [(domain, lemma) for domain, lemma
                          in zip(s['domain'].values.tolist(),                            
                                 s['lemma'].values.tolist())]
    
    grouped_data_test = data_test.groupby(['domain']).apply(agg_func)
    
    # Comentarios raw
    
    data_comentarios_test = comentarios_raw(grouped_data_test)
    data_comentarios_test['sentiment'].value_counts()
    
    # y_test = grid_svm.predict(data_comentarios_test['comentario'])
    y_test = grid_svm_neg.predict(data_comentarios_test['comentario'])
    
    dominio = np.array([nombre_dominio] * len(data_comentarios_test))
    
    y_test = np.where(y_test == 1, 'positive', 'negative')
   
    output = pd.concat([data_comentarios_test['nombre_comentario'], pd.DataFrame(dominio), pd.DataFrame(y_test)], axis = 1)
    
    for i in range(len(output)): output.loc[i, 'nombre_comentario'] = '_'.join(output.loc[i, 'nombre_comentario'].rsplit('_')[1:])
    
    opinion = pd.concat([opinion, output], ignore_index=True)
    
# para guardar todos los test uno por uno
# output_data = os.getcwd() + '/outputs/opinion/' + dataset_test + '/opinion_' + datafile_test.split('/')[-1]

output_data = os.getcwd() + '/outputs/opinion/' + dataset_test + '/opinion.txt'
opinion.to_csv(output_data, sep = '\t', index=False, quoting=csv.QUOTE_NONE, header= None, encoding = 'UTF-8')
    





