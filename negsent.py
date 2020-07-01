#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:11:39 2020

@author: carlosdelafuente

Versión: 0.2, usando datasets con cue scope.
"""

import os
import numpy as np
import pandas as pd
import pickle
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn_crfsuite import metrics
from sklearn_crfsuite import CRF
from collections import Counter

import setup
import training
import testing

from sklearn_crfsuite import CRF
import scipy.stats

from sklearn.model_selection import RandomizedSearchCV


## Main   

# dataset de trabajo: 'dev', 'train' o 'train_dev'
dataset = 'train'

##### -----> Si NER usa '_dataset_bio_o'; ELSE usa '_dataset_bio.txt'

bio_dataset = os.getcwd() + '/data/' + dataset + '/bio/' + dataset + '_dataset_bio_o.txt'

##### -----

if os.path.isfile(bio_dataset) == False: 
    # une todos los archivos de entrada en uno
    if dataset == 'train_dev':
        setup.join_data_train_dev()
    else:
        setup.join_data(dataset)

    # calcula el número máximo de negaciones y crea un df de trabajo
    data, negaciones = setup.get_data(dataset)
    
    # Ajusta bio_tag frase a frase
    bio_data = pd.DataFrame()
    frases = data['id_sentence'].unique()
    for f in frases:
        if pd.notna(f):
            frase = data.loc[data['id_sentence'] == f].copy()
            frase.fillna('-', inplace = True)
            frase['bio_tag'], frase['sco_tag'] = setup.set_bio_frase(frase, negaciones)
            frase = frase.append(pd.Series(), ignore_index = True)
            bio_data = bio_data.append(frase)
            
    # salvo archivo bio_taggeado a disco
    bio_data.to_csv(bio_dataset, encoding = 'UTF-8', sep ='\t')
            
else:
    data = pd.read_csv(bio_dataset, delimiter = '\t', encoding = 'UTF-8', 
                       index_col=[0], skip_blank_lines=False)

# Obtiene el listado de negaciones más frecuentes   
training.get_negaciones(data)

# Sentencias para generar test a partir de dev files. SOLO PARA AJUSTES !!
   
#setup.tag_dev()
#setup.get_dev_4_training()

#setup.genera_dev_test()
#setup.quita_header_test() 



# Training usando MLP, NB y SVC; entrena usando: lemma, PoS
    
X = data[['lemma','PoS']]
X = X.dropna()
y = data[['bio_tag']]
y = y.dropna()

v = DictVectorizer(sparse=False)
X = v.fit_transform(X.to_dict('records'))

classes = np.unique(y)
classes = classes.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                test_size=0.3,random_state=123)

# Clasificador MLP

y_pred = training.training_per(X_train, y_train, X_test, classes, dataset)

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

y_pred = training.training_nb(X_train, y_train, X_test, classes, dataset)

conf_mtx_nb = pd.DataFrame(
    confusion_matrix(y_test, y_pred, labels=classes), 
    index=['true:{:}'.format(x) for x in classes], 
    columns=['pred:{:}'.format(x) for x in classes]
)

print('Clasificador NB\n')
print(conf_mtx_nb)
print(classification_report(y_test,y_pred))

tabla_resultados = tabla_resultados + '\nClasificador NB:\n\n'
tabla_resultados = tabla_resultados + classification_report(y_test,y_pred)

# Clasificador SVM

y_pred = training.training_svc(X_train, y_train, X_test, dataset)

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

# Clasificador CRF. Secuencial: Cue y Scope

training_cue = 'cue'
y, pred, modelo = training.training_crf(training_cue, data, dataset)

classes = sorted(list(modelo.classes_))

print('Clasificador CRF (Cue):\n')
print(flat_classification_report(y_pred=pred, y_true=y))

tabla_resultados = tabla_resultados + '\nClasificador CRF (Cue):\n\n'
tabla_resultados = tabla_resultados + metrics.flat_classification_report(y_pred=pred, 
                                                    y_true=y, labels = classes)

# Compruebo aprendizaje del modelo

print("Top likely transitions:")
training.print_transitions(Counter(modelo.transition_features_).most_common())

print("Top positive:")
training.print_state_features(Counter(modelo.state_features_).most_common(30))
print("\nTop negative:")
training.print_state_features(Counter(modelo.state_features_).most_common()[-30:])


training_cue = 'sco'
y, pred, modelo = training.training_crf(training_cue, data, dataset)

classes = sorted(list(modelo.classes_))

print('Clasificador CRF (Scope):\n')
print(flat_classification_report(y_pred=pred, y_true=y))

tabla_resultados = tabla_resultados + '\nClasificador CRF (Scope):\n\n'
tabla_resultados = tabla_resultados + metrics.flat_classification_report(y_pred=pred, 
                                                    y_true=y, labels = classes)

# Compruebo aprendizaje del modelo

print("Top likely transitions:")
training.print_transitions(Counter(modelo.transition_features_).most_common())

print("Top positive:")
training.print_state_features(Counter(modelo.state_features_).most_common(30))
print("\nTop negative:")
training.print_state_features(Counter(modelo.state_features_).most_common()[-30:])

# ----------
# Ajuste hiperparámetros del modelo CRF

#Now we will create the Randomized CV search model wherein we will use a modified F1 scorer model considering only the relevant labels
# define fixed parameters and parameters to search

#agg_func = lambda s: [(w, l, p, t) for w, l, p, t in zip(
#                s["word"].values.tolist(),
#                s["lemma"].values.tolist(),
#                s["PoS"].values.tolist(),
#                s["bio_tag"].values.tolist())]
#    
#grouped_df = data.groupby('id_sentence').apply(agg_func)
#frases = [f for f in grouped_df]
#    
#X = np.array([training.sent2features(f) for f in frases])
#y = np.array([training.sent2labels(f) for f in frases])
#    
#crf3 = CRF(
#    algorithm='lbfgs',
#    max_iterations=100,
#    all_possible_transitions=True)
#
#params_space = {
#    'c1': scipy.stats.expon(scale=0.5),
#    'c2': scipy.stats.expon(scale=0.05),}
#
## use the same metric for evaluation
#f1_scorer = make_scorer(metrics.flat_f1_score,
#                        average='weighted', labels=classes)
#
## search
#rs = RandomizedSearchCV(crf3, params_space,
#                        cv=3,
#                        verbose=1,
##                        n_jobs=-1,
#                        n_iter=50,
#                        scoring=f1_scorer)
#rs.fit(X, y)
#
##Lets check the best estimated parameters and CV score
#print('Best parameters:', rs.best_params_)
#print('Best CV score:', rs.best_score_)
#print('Model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
#
#crf3 = rs.best_estimator_
#y_pred = crf3.predict(X)
#print(metrics.flat_classification_report(y, y_pred, labels=classes, digits=3))
#
#crf3.fit(X,y)
#
#model_filename = os.getcwd() + '/models/' + dataset + '/crf_opt_model.pkl'
#with open(model_filename, 'wb') as file_model:  
#    pickle.dump(crf3, file_model)

#-----------

# Training NER BIO. Comprueba iteraciones!

setup.tsv_spacy_cue_format()
setup.tsv_spacy_scope_format()  # Scope
  
# Training Cue

bio_spc_output = os.getcwd() + '/data/' + dataset + '/bio/' + dataset + '_dataset_bio_o.spc'
training_cue = 'cue'
training.training_scope_NER(training_cue, bio_spc_output, dataset)

# Training Scope

bio_spc_output = os.getcwd() + '/data/' + dataset + '/bio/' + dataset + '_dataset_bio_scope_o.spc'
training_cue = 'sco'
training.training_scope_NER(training_cue, bio_spc_output, dataset)

# ----------   TEST ---------

# Testing solo para ajustar el sistema

''' **** Uso el devfile con anotaciones bio para hacer test: versión etiquetada como 
'gold' y otra sin etiquetar para predecir las etiquetas aplicando el modelo 
entrenado. Para SVM hay que vectorizar usando el mismo vocabulario usado con el 
training set!'''

dev_test_file = os.getcwd() + '/data/dev/bio/dev_dataset_bio.txt'

data_dev_test = pd.read_csv(dev_test_file, delimiter = "\t", 
                  encoding = 'UTF-8', index_col=[0])

data_dev_test = data_dev_test.dropna()

y_dev_gold = data_dev_test[['bio_tag']]

data_dev_test = data_dev_test.drop(columns = 'bio_tag')

# Test SVM

y_dev_test_pred_svc, modelo = testing.testing_dev_svm(data_dev_test, v, dataset)

classes = sorted(list(modelo.classes_))

conf_mtx_test_svc = pd.DataFrame(
    confusion_matrix(y_dev_gold, y_dev_test_pred_svc, labels=classes), 
    index=['true:{:}'.format(x) for x in classes], 
    columns=['pred:{:}'.format(x) for x in classes]
)

print('Test Clasificador SVM\n')
print(conf_mtx_test_svc)
print(classification_report(y_dev_gold, y_dev_test_pred_svc))

tabla_resultados = tabla_resultados + '\nPredicción sobre Test (Dev) usando modelo SVM:\n\n'
tabla_resultados = tabla_resultados + classification_report(y_dev_gold, y_dev_test_pred_svc)

#y_dev_test_pred_svc_list = y_dev_test_pred_svc.tolist()
#np.save(os.getcwd() + '/outputs/' + 'output_svc_test', y_dev_test_pred_svc_list)
  

# Test CRF

data_dev_test = pd.read_csv(dev_test_file, delimiter = "\t", encoding = 'UTF-8', 
                            index_col=[0], skip_blank_lines=False)

y_dev_gold = data_dev_test['bio_tag'].tolist()

data_dev_test = data_dev_test[['domain', 'sentence', 'token', 'word', 'lemma', 
                        'PoS', 'PoS_type']]

y_dev_test_pred_crf, modelo = testing.testing_dev_crf(data_dev_test, dataset)

classes = sorted(list(modelo.classes_))
y_dev_test_pred_crf_list = y_dev_test_pred_crf.tolist()

conf_mtx_test_crf = pd.DataFrame(
    confusion_matrix(y_dev_gold, y_dev_test_pred_crf_list, 
                     labels=classes), 
                     index=['true:{:}'.format(x) for x in classes], 
                     columns=['pred:{:}'.format(x) for x in classes]
)

print('Test Clasificador CRF\n')
print(conf_mtx_test_crf)
print(classification_report(y_dev_gold, y_dev_test_pred_crf_list))

tabla_resultados = tabla_resultados + '\nPredicción sobre Test (Dev) usando modelo CRF optimizado:\n\n'
tabla_resultados = tabla_resultados + classification_report(y_dev_gold, 
                                                y_dev_test_pred_crf_list)
   

#----------- TEST PARA SCORER ---------------


''' **** Se usan los devfile ! Cambiar el path_test_files para hacer testing 
sobre otros ficheros (test). Hay que cambiar tambien 'dev_' por 'test_' en los nombres
de los ficheros dentro de las funciones del módulo testing '''

# Test de modelos con los conjuntos de test reales. La salida siempre es a
# un directorio '/outputs/' cuando es 'BIO' o '/scorer/' cuando es 'Conll'.

# Ficheros BIO de test entrenados con SVC y/o CRF

# path_test_files = os.getcwd() + '/data/test/test_dev/'
path_test_files = os.getcwd() + '/data/test/'

filenames = ['coches', 'lavadoras', 'hoteles', 'moviles', 'libros', 'musica', 
             'ordenadores', 'peliculas']

# testing.testing_model_svm(filenames, path_test_files_in, v, dataset)   # Revisar
testing.testing_model_crf(filenames, path_test_files, dataset)

# Genera ficheros CONLL a partir de los BIO Cue / Scope anteriores para llevar al Scorer

for f_name in filenames:    
    tecnica = 'crf'
    setup.genera_cue_scope(tecnica)
    setup.bio_conll(tecnica, f_name)
    

# Ficheros BIO de test entrenados con NER

# Testing Cue / Scope secuencialente en dis archivos de salida para hacer scores
# independientes. Para hacer el score final se concatenan.

testing.testing_model_NER(filenames, path_test_files, dataset)

# Genera ficheros CONLL a partir de los BIO Cue / Scope anteriores para llevar al Scorer

for f_name in filenames:    
    tecnica = 'ner'    
    setup.genera_cue_scope(tecnica)
    setup.bio_conll(tecnica, f_name)
    
# ------------------ END