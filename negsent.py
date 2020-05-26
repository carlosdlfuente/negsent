#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:11:39 2020

@author: carlosdelafuente
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
#from sklearn_crfsuite.metrics import flat_classification_report
from sklearn_crfsuite import metrics

import setup
import training
import testing

### Main   

# dataset de trabajo: 'dev' o 'train'
dataset = 'train'

# une todos los archivos de entrada en uno
setup.join_data(dataset)

# calcula el número máximo de negaciones y crea un df de trabajo
data, negaciones = setup.get_data(dataset)


# Ajusta bio_tag frase a frase

frases = data['id_sentence'].unique()
for f in frases:
    frase = data.loc[data['id_sentence'] == f].copy()
    frase['bio_tag'] = setup.set_bio_frase(frase, negaciones)
    data.loc[data['id_sentence'] == f] = frase


# salvo archivo bio_taggeado a disco
data.to_csv(os.getcwd() + '/data/' + dataset + '/' + dataset + '_bio_dataset.txt',
            sep = '\t')

# entrena usando: lemma, PoS

X = data[['lemma','PoS']]
y = data[['bio_tag']]

v = DictVectorizer(sparse=False)
X = v.fit_transform(X.to_dict('records'))

classes = np.unique(y)
classes = classes.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                test_size=0.3,random_state=123)

# Clasificador MLP

y_pred = training.training_per(X_train, y_train, X_test, y_test, classes, dataset)

conf_mtx_per = pd.DataFrame(
    confusion_matrix(y_test, y_pred, labels=classes), 
    index=['true:{:}'.format(x) for x in classes], 
    columns=['pred:{:}'.format(x) for x in classes]
)

print('Clasificador MLP\n')
print(conf_mtx_per)
print(classification_report(y_test,y_pred))

tabla_resultados = 'Clasificador MLP:\n\n'
tabla_resultados = tabla_resultados + classification_report(y_test,y_pred)

# Opción: Dejar fuera la clase 'O' par medir el rendimiento sobre clases relevantes
#labels = list(crf.classes_)
#labels.remove('O')

# Clasificador NAive Bayes

y_pred = training.training_nb(X_train, y_train, X_test, y_test, classes, dataset)

conf_mtx_nb = pd.DataFrame(
    confusion_matrix(y_test, y_pred, labels=classes), 
    index=['true:{:}'.format(x) for x in classes], 
    columns=['pred:{:}'.format(x) for x in classes]
)

print('Clasificador NB\n')
print(conf_mtx_nb)
print(classification_report(y_test,y_pred))

tabla_resultados = 'Clasificador NB:\n\n'
tabla_resultados = tabla_resultados + classification_report(y_test,y_pred)

# Clasificador SVM

y_pred = training.training_svc(X_train, y_train, X_test, y_test, classes, dataset)

conf_mtx_svc = pd.DataFrame(
    confusion_matrix(y_test, y_pred, labels=classes), 
    index=['true:{:}'.format(x) for x in classes], 
    columns=['pred:{:}'.format(x) for x in classes]
)

print('Clasificador SVM\n')
print(conf_mtx_svc)
print(classification_report(y_test,y_pred))

tabla_resultados = tabla_resultados + '\nClasificador SVM:\n\n'
tabla_resultados = tabla_resultados + classification_report(y_test,y_pred)

# Clasificador CRF

agg_func = lambda s: [(w, l, p, t) for w, l, p, t in zip(
                s["word"].values.tolist(),
                s["lemma"].values.tolist(),
                s["PoS"].values.tolist(),
                s["bio_tag"].values.tolist())]
    
y_test, y_pred = training.training_crf(data, agg_func, dataset)

print('Clasificador CRF\n')
print(metrics.flat_classification_report(y_test, y_pred, labels = classes))

tabla_resultados = tabla_resultados + '\nClasificador CRF:\n\n'
tabla_resultados = tabla_resultados + metrics.flat_classification_report(y_test, 
                                                    y_pred, labels = classes)

# Testing

''' **** Cargo el archivo de test ('dev', ahora) para tomarlo como 'gold' al 
aplicar el modelo. Hay que vectorizar usando el mismo vocabulario usado con el 
training set!'''

test_dataset = os.getcwd() + '/data/dev/dev_bio_dataset.txt'
data_test = pd.read_csv(test_dataset, delimiter = "\t", 
                      encoding = 'UTF-8', index_col=[0])
y_test_gold = data_test[['bio_tag']]

# Test SVM

y_test_pred_svc = testing.testing_svm(data_test, v, dataset)

conf_mtx_god_svc = pd.DataFrame(
    confusion_matrix(y_test_gold, y_test_pred_svc, labels=classes), 
    index=['true:{:}'.format(x) for x in classes], 
    columns=['pred:{:}'.format(x) for x in classes]
)

print('Test Clasificador SVM\n')
print(conf_mtx_god_svc)
print(classification_report(y_test_gold, y_test_pred_svc))

tabla_resultados = tabla_resultados + '\nPredicción sobre Test usando modelo SVM:\n\n'
tabla_resultados = tabla_resultados + classification_report(y_test_gold, y_test_pred_svc)

# Test CRF

y_test_gold, y_test_pred_crf = testing.testing_crf(data_test, agg_func, dataset)

print('Test Clasificador CRF\n')
print(metrics.flat_classification_report(y_test_gold, y_test_pred_crf, labels = classes))

tabla_resultados = tabla_resultados + '\nPredicción sobre Test usando modelo CRF:\n\n'
tabla_resultados = tabla_resultados + metrics.flat_classification_report(y_test_gold, 
                                                                         y_test_pred_crf, 
                                                                         labels = classes)

y_test_pred_crf_list = [ item for elem in y_test_pred_crf for item in elem]
np.save(os.getcwd() + '/outputs/' + 'output_crf_test', y_test_pred_crf_list)