#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 19:34:21 2020

@author: carlosdelafuente
"""
# PENDIENTE

''' Aplicado clasificador LinearSVC a RAW y a NEG.
'''

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
from sklearn.calibration import CalibratedClassifierCV

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
            negaciones.append(data.iloc[i]['lemma'])
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
    return    

def join_data_test():    
    
    # une todos los ficheros de entrada 'train' o 'dev'

    files = []
    path = os.getcwd() + '/data/test/'
    join_file = os.path.join(path, ('test_join.txt'))

    for f_name in os.listdir(path):
        if f_name.endswith('.txt'):
            files.append(f_name)
    with open(join_file, 'w') as outfile:
        for f in files:
            with open(os.path.join(path, f), 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
    return 

def report_results(model, decision_scores, X, y):
    # pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)        
    auc = roc_auc_score(y, decision_scores)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    result = {'auc': auc, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
    return result

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


def dummy(doc):
    return doc



def training_svc(training_data, modelo, dataset):
    train, test = train_test_split(training_data, test_size=0.2, random_state=0)
    X_train = train['comentario'].values
    X_test = test['comentario'].values
    y_train = train['sentiment']
    y_test = test['sentiment']
    
    # Preprocesamiento de StopWords
    
    # Opción 1.- Uso na lista de stopwords personalizada donde elimino las Top 15 negacioones
        
    # negaciones_habituales = get_negaciones(data)
    # es_stopwords = set(stopwords.words("spanish")) 
    # new_es_stopwords = [x for x in es_stopwords if x not in negaciones_habituales]
    
    # Opción 2.- Usar  max_df=1.0, min_df=0.6
    
    # vectorizer = CountVectorizer(
    #     analyzer = 'word',
    #     tokenizer = dummy,
    #     preprocessor = dummy,
    #     # lowercase = True,
    #     ngram_range=(1, 5),     #trigramas
    #     max_df=1.0, min_df=0.4, max_features=None,
    #     # stop_words = new_es_stopwords,
    #     token_pattern=None)
    
    # vectorizer = TfidfVectorizer(
    #     # stop_words = new_es_stopwords,
    #     tokenizer = dummy,
    #     preprocessor = dummy,
    #     # sublinear_tf = True,
    #     strip_accents = 'ascii',
    #     analyzer = 'word'
    #     # token_pattern = r'\w{2,}',  #vectorize 2-character words or more
    #     # ngram_range = (1, 2),
    #     # max_df=1.0, min_df=0.6, max_features=None
    #     )
    
    parameters = {
        'vectorizer__min_df': (0.2, 0.4, 0.6, 0.8),
        'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
        "clf__penalty": ["l2"],
        "clf__dual":[False,True]
        # 'SVM__C': [0.01, 0.1, 1, 100],
        # 'SVM__gamma':[0.1,0.01]
        }
    
    pipeline_svm = Pipeline([
        ('vectorizer', TfidfVectorizer(
            tokenizer = dummy,
            preprocessor = dummy,
            strip_accents = 'ascii',
            analyzer = 'word',
            max_df = 1.0,
            max_features = None
        )), 
        # ('SVM', SVC(
        #     probability=True, 
        #     kernel = "linear", 
        #     class_weight = "balanced"
        ('clf', LinearSVC())
        ])
    
    kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    
    grid_svm = GridSearchCV(pipeline_svm,
                        param_grid = parameters, 
                        cv = kfolds,
                        scoring="roc_auc",
                        verbose=1,   
                        n_jobs=-1) 
    
    np.random.seed(0)
    
    # pipeline_svm = make_pipeline(vectorizer, 
    #                             SVC(probability=True, 
    #                                 kernel="linear",
    #                                 class_weight="balanced"))
    
    
    # grid_svm = GridSearchCV(pipeline_svm,
    #                     param_grid = {'svc__C': [0.01, 0.1, 1, 100]}, 
    #                     cv = kfolds,
    #                     scoring="roc_auc",
    #                     verbose=1,   
    #                     n_jobs=-1) 
    
    grid_svm.fit(X_train, y_train)
    
    # salva modelo
    
    model_filename = os.getcwd() + '/models/opinion/' + dataset + '/' + modelo + 'opinion.pkl'
    with open(model_filename, 'wb') as file_model:  
        pickle.dump(grid_svm, file_model)
        
    return grid_svm, X_train, y_train, X_test, y_test


dataset = 'train_dev'

# join_data(dataset)
# datafile = os.getcwd() + '/outputs/opinion/output_ner_cue_scope_' + dataset + '_join.txt'
# data = pd.read_csv(datafile, delimiter = '\t', encoding = 'UTF-8', header = None,
#                     index_col=None, skip_blank_lines=False)
# data = data[[0, 1, 3, 7, 8]]
# data.columns = ['domain', 'sentence', 'word', 'bio_tag', 'sco_tag']

# Datafile etiquetas originales
datafile = os.getcwd() + '/data/' + dataset + '/bio/' + dataset + '_dataset_bio_o.txt'

# # Datafile etiquetas predichas
# datafile = os.getcwd() + '/outputs/' + dataset + '/bio/' + dataset + '_dataset_bio_o.txt'

data = pd.read_csv(datafile, delimiter = '\t', encoding = 'UTF-8', dtype=object,
                    index_col=None, skip_blank_lines=False, quoting=csv.QUOTE_NONE)
data = data[['domain', 'lemma', 'bio_tag', 'sco_tag']]

# data['lemma'] = data['lemma'].str.lower()

# agg_func = lambda s: [(domain, sentence, lemma, bio_tag, sco_tag) for domain, sentence, lemma, bio_tag, sco_tag
#                       in zip(s['domain'].values.tolist(),
#                              s['sentence'].values.tolist(),
#                              s['lemma'].values.tolist(), 
#                              s['bio_tag'].values.tolist(),
#                              s['sco_tag'].values.tolist(),)]

agg_func = lambda s: [(domain, lemma, sco_tag) for domain, lemma, sco_tag
                      in zip(s['domain'].values.tolist(),
                             s['lemma'].values.tolist(), 
                             s['sco_tag'].values.tolist())]

grouped_data = data.groupby(['domain']).apply(agg_func)

data_comentarios_raw = comentarios_raw(grouped_data)
data_comentarios_raw['sentiment'].value_counts()

data_comentarios_neg = comentarios_neg(grouped_data)
data_comentarios_neg['sentiment'].value_counts()

training_data = data_comentarios_raw
modelo_svc_raw, X_train_raw, y_train_raw, X_test_raw, y_test_raw = training_svc(training_data, 'modelo_raw_', dataset)

training_data = data_comentarios_neg
modelo_svc_neg, X_train_neg, y_train_neg, X_test_neg, y_test_neg = training_svc(training_data, 'modelo_neg_', dataset)
    
# Modelo Raw

modelo_svc_raw.score(X_test_raw, y_test_raw)

print("Best: %f using %s" % (modelo_svc_raw.best_score_, modelo_svc_raw.best_params_))
print(classification_report(y_test_raw, modelo_svc_raw.predict(X_test_raw)))


clases = modelo_svc_raw.classes_
conf_mtx = pd.DataFrame(
    confusion_matrix(y_test_raw, modelo_svc_raw.predict(X_test_raw), labels = clases), 
    index=['true:{:}'.format(x) for x in clases], 
    columns=['pred:{:}'.format(x) for x in clases]
)
print(conf_mtx)

decision_scores_raw = modelo_svc_raw.decision_function(X_test_raw)
fpr_raw, tpr_raw, thers_raw = roc_curve(y_test_raw, decision_scores_raw)
auc_raw = roc_auc_score(y_test_raw, decision_scores_raw)

results_raw = report_results(modelo_svc_raw.best_estimator_, decision_scores_raw, X_test_raw, y_test_raw)
results_raw

# Modelo Neg

modelo_svc_neg.score(X_test_neg, y_test_neg)

print("Best: %f using %s" % (modelo_svc_neg.best_score_, modelo_svc_neg.best_params_))
print(classification_report(y_test_neg, modelo_svc_neg.predict(X_test_neg)))


clases = modelo_svc_neg.classes_
conf_mtx = pd.DataFrame(
    confusion_matrix(y_test_neg, modelo_svc_neg.predict(X_test_neg), labels = clases), 
    index=['true:{:}'.format(x) for x in clases], 
    columns=['pred:{:}'.format(x) for x in clases]
)
print(conf_mtx)

decision_scores_neg = modelo_svc_neg.decision_function(X_test_neg)
fpr_neg, tpr_neg, thers_neg = roc_curve(y_test_neg, decision_scores_neg)
auc_neg = roc_auc_score(y_test_neg, decision_scores_neg)

results_neg = report_results(modelo_svc_neg.best_estimator_, decision_scores_neg, X_test_neg, y_test_neg)
results_neg

# Tabla de resultados de validación

resultados_validacion = pd.concat([pd.DataFrame(results_raw.items()), 
                                   pd.DataFrame(results_neg.values())], axis = 1)

resultados_validacion.columns = ['metrica', 'raw_data', 'NEG_data']
resultados_validacion

# Plot ROC clasificadores

plt.figure(figsize=(14,8))

plt.plot(fpr_raw, tpr_raw, color="red", label = 'AUC Baseline: %0.3f' % auc_raw)
plt.plot(fpr_neg, tpr_neg, color="blue", label = 'AUC Negación: %0.3f' % auc_neg)

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc curve')
plt.legend(prop={'size':13}, loc='lower right')
plt.show()


''' Learning curves ayudan a visualizar el efecto del numero de observaciones sobre
las métricas de rendimiento. No hay problema de sesgo o varianza. El rendimiento 
del modelo aumenta cuando tenemos más datos.'''

'''OJO:  Usar grid_svm o grid_svm_neg y cambiar X_train, y_train'''

# scoring="roc_auc"
# scoring="accuracy"

# Ref: https://chrisalbon.com/machine_learning/model_evaluation/plot_the_learning_curve/

train_sizes_raw, train_scores_raw, test_scores_raw = \
    learning_curve(modelo_svc_raw.best_estimator_, X_train_raw, y_train_raw, cv=10, n_jobs=-1, 
                   scoring="roc_auc", train_sizes=np.linspace(.1, 1.0, 10), random_state=1)

plot_learning_curve(X_train_raw, y_train_raw, train_sizes_raw, 
                    train_scores_raw, test_scores_raw, 'Learning curve RAW Data', ylim=(0.7, 1.01), figsize=(14,6))
plt.show()


train_sizes_neg, train_scores_neg, test_scores_neg = \
    learning_curve(modelo_svc_neg.best_estimator_, X_train_neg, y_train_neg, cv=10, n_jobs=-1, 
                   scoring="roc_auc", train_sizes=np.linspace(.1, 1.0, 10), random_state=1)

plot_learning_curve(X_train_neg, y_train_neg, train_sizes_neg, 
                    train_scores_neg, test_scores_neg, 'Learning curve NEG Data', ylim=(0.7, 1.01), figsize=(14,6))
plt.show()


# # Test
    
# # join_data_test()
# dataset_test = 'test'

# path_test_files = os.getcwd() + '/data/test/'
    
# filenames = ['coches', 'lavadoras', 'hoteles', 'moviles', 'libros', 'musica', 
#              'ordenadores', 'peliculas']

# test_files = [os.path.join(path_test_files, 'test_' + f_name + '.txt') for f_name in filenames]
# model_filename = os.getcwd() + '/models/opinion/' + dataset + '/' + 'modelo_raw_opinion.pkl'

# with open(model_filename, 'rb') as file_model:  
#     modelo = pickle.load(file_model)
    
# output  = pd.DataFrame()
# opinion = pd.DataFrame()

# for datafile_test, nombre_dominio in zip(test_files, filenames):  
    
#     dominio = []
#     nombre_comentario = []
    
#     # datafile_test = os.getcwd() + '/data/test/test_join.txt'
    
#     data_test = pd.read_csv(datafile_test, delimiter = '\t', encoding = 'UTF-8', 
#                             header = None, index_col = False, dtype=object,
#                             usecols = [0, 4], skip_blank_lines=False, quoting=csv.QUOTE_NONE)
    
#     data_test.columns = ['domain', 'lemma']
    
#     # data_test['lemma'] = data_test['lemma'].str.lower()
        
#     agg_func = lambda s: [(domain, lemma) for domain, lemma
#                           in zip(s['domain'].values.tolist(),                            
#                                  s['lemma'].values.tolist())]
    
#     grouped_data_test = data_test.groupby(['domain']).apply(agg_func)
    
#     data_comentarios_test = comentarios_raw(grouped_data_test)
#     data_comentarios_test['sentiment'].value_counts()
    
#     y_test = modelo.predict(data_comentarios_test['comentario'])
#     # y_test = grid_svm_neg.predict(data_comentarios_test['comentario'])
    
#     # dominio = np.array([nombre_dominio] * len(data_comentarios_test))
    
#     for i in range(len(data_comentarios_test)): dominio.append(data_comentarios_test['nombre_comentario'][i].split('_')[0])
#     for i in range(len(data_comentarios_test)): nombre_comentario.append('_'.join(data_comentarios_test['nombre_comentario'][i].rsplit('_')[1:]))
#     y_test = np.where(y_test == 1, 'positive', 'negative')
   
#     output = pd.concat([pd.DataFrame(nombre_comentario), pd.DataFrame(dominio), pd.DataFrame(y_test)], axis = 1)
    
#     # for i in range(len(output)): output.loc[i, 'nombre_comentario'] = '_'.join(output.loc[i, 'nombre_comentario'].rsplit('_')[1:])
    
#     opinion = pd.concat([opinion, output], ignore_index=True)
    
# # para guardar todos los test uno por uno
# # output_data = os.getcwd() + '/outputs/opinion/' + dataset_test + '/opinion_' + datafile_test.split('/')[-1]

# output_data = os.getcwd() + '/outputs/opinion/' + dataset_test + '/opinion.txt'
# opinion.to_csv(output_data, sep = '\t', index=False, quoting=csv.QUOTE_NONE, header= None, encoding = 'UTF-8')
    





