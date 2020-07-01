#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:07:24 2020

@author: carlosdelafuente
"""

import os
import numpy as np
import pickle
import collections
import es_core_news_md
import random
import spacy
from spacy.util import minibatch, compounding
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from tqdm import tqdm # loading bar


def get_negaciones(data):
    negaciones = []
    for i in range(len(data)):
        if (data.iloc[i]['bio_tag'] in ['B-Cue', 'I-Cue']):
            negaciones.append(data.iloc[i]['lemma'])
    counts = collections.Counter(negaciones)
    top_negaciones = sorted(counts, key=lambda x: -counts[x])    
    get_negaciones.negaciones_list = list(top_negaciones)[:15]
    return


def check_negacion(lema):    
    if lema in get_negaciones.negaciones_list:
        return(True)
    else:
        return(False)
        

def word2features(sent, i):

    features = {
        'bias': 1.0,
        'known_cue' : True if check_negacion(sent[i][1]) else False,
    }
    minimum_index = max(i - 7, 0)
    maximum_index = min(i + 7, len(sent) - 1)

    for j in range(minimum_index, maximum_index + 1):
        if j == i - 6:
            features.update({
                'lemma-6': sent[j][1],
                'postag-6': sent[j][2]
            })
        if j == i - 5:
            features.update({
                'lemma-5': sent[j][1],
                'postag-5': sent[j][2]
            })
        if j == i - 4:
            features.update({
                'lemma-4': sent[j][1],
                'postag-4': sent[j][2]
            })
        if j == i - 3:
            features.update({
                'lemma-3': sent[j][1],
                'postag-3': sent[j][2]
            })
        if j == i - 2:
            features.update({
                'lemma-2': sent[j][1],
                'postag-2': sent[j][2]
            })
        elif j == i - 1:
            features.update({
                'lemma-1': sent[j][1],
                'postag-1': sent[j][2]
            })
        elif j == i:
            features.update({
                'lemma': sent[i][1],
                'postag': sent[i][2]
            })
        elif j == i + 1:
            features.update({
                'lemma+1': sent[j][1],
                'postag+1': sent[j][2]
            })
        elif j == i + 2:
            features.update({
                'lemma+2': sent[j][1],
                'postag+2': sent[j][2]
            })
        elif j == i + 3:
            features.update({
                'lemma+3': sent[j][1],
                'postag+3': sent[j][2]
            })
        elif j == i + 4:
            features.update({
                'lemma+4': sent[j][1],
                'postag+4': sent[j][2]
            })
        elif j == i + 5:
            features.update({
                'lemma+5': sent[j][1],
                'postag+5': sent[j][2]
            })
        elif j == i + 6:
            features.update({
                'lemma+6': sent[j][1],
                'postag+6': sent[j][2]
            })
    return features


def sent2features(sent):
    features = [word2features(sent, i) for i in range(len(sent))]
    return features

def sent2labels(sent, training_cue):
    if training_cue == 'cue':
        return [bio_tag for token, lemma, PoS, bio_tag, sco_tag in sent]
    else:
        return [sco_tag for token, lemma, PoS, bio_tag, sco_tag in sent]
    
# Clasificadores: MLP, NB y SVM y CRF.

# MLP

def training_per(X_train, y_train, X_test, classes, dataset):
    per = Perceptron(verbose=10, n_jobs=-1, max_iter=5)
    per.partial_fit(X_train, y_train.values.ravel(), classes)
    y_pred = per.predict(X_test)
    
    model_filename = os.getcwd() + '/models/' + dataset + '/per_model.pkl'
    with open(model_filename, 'wb') as file_model:  
        pickle.dump(per, file_model)
    return(y_pred)

# Naive Bayes

def training_nb(X_train, y_train, X_test, classes, dataset):
    nb = MultinomialNB(alpha=0.01)
    nb.partial_fit(X_train, y_train, classes)
    y_pred = nb.predict(X_test)
      
    model_filename = os.getcwd() + '/models/' + dataset + '/nb_model.pkl'
    with open(model_filename, 'wb') as file_model:  
        pickle.dump(nb, file_model)
    return(y_pred)

# SVM

def training_svc(X_train, y_train, X_test, dataset):
    svc = svm.LinearSVC(C=10, max_iter = 2000)
    svc.fit(X_train, y_train.values.ravel())
    y_pred = svc.predict(X_test)
    
    model_filename = os.getcwd() + '/models/' + dataset + '/svc_model.pkl'
    with open(model_filename, 'wb') as file_model:  
        pickle.dump(svc, file_model)
    return(y_pred)

# CRF

class get_frase(object):
    
    def __init__(self, data):
        self.n_sent = 1.0
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, l, p, t, s) for w, l, p, t, s in zip(s["word"].values.tolist(),
                                                           s["lemma"].values.tolist(),                               
                                                           s["PoS"].values.tolist(),
                                                           s["bio_tag"].values.tolist(),
                                                           s["sco_tag"].values.tolist())]
        self.grouped = self.data.groupby(['domain','sentence']).apply(agg_func)
        self.get_frase = [s for s in self.grouped]
        

def training_crf(training_cue, data, dataset):
       
    getter = get_frase(data)
    frases = getter.get_frase
   
    get_negaciones(data)
        
    X = [sent2features(f) for f in frases]
    y = [sent2labels(f, training_cue) for f in frases]
       
    crf = CRF(algorithm='lbfgs',
              c1=0.1,
              c2=0.1,
              max_iterations=100,
              all_possible_transitions=True,
              verbose=True)
    
    pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)
        
    crf.fit(X, y)
    
    if training_cue == 'cue':
        model_filename = os.getcwd() + '/models/' + dataset + '/crf_cue_model.pkl'
    else:
        model_filename = os.getcwd() + '/models/' + dataset + '/crf_sco_model.pkl'
        
    with open(model_filename, 'wb') as file_model:  
        pickle.dump(crf, file_model)
        
    return(y, pred, crf)

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))


def training_scope_NER(training_cue, bio_spc_output, dataset):
    
    # Training spaCy NER with Custom Entities
    
    if training_cue == 'cue':
        nlp = es_core_news_md.load()
        model_spc_output = os.getcwd() + '/models/' + dataset + '/spacy_model'
        LABEL = ['B-Cue', 'I-Cue']
    else:
        model_spc_output = os.getcwd() + '/models/' + dataset + '/spacy_model'
        nlp = spacy.load(model_spc_output)
        model_spc_output = os.getcwd() + '/models/' + dataset + '/spacy_scope_model'
        LABEL = ['B-Cue', 'I-Cue', 'B-Sco', 'I-Sco']
      
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else:
        ner =  nlp.get_pipe('ner')
        
    n_iter = 10                     # Iteraciones
    
    with open (bio_spc_output, 'rb') as fp:
        TRAIN_DATA = pickle.load(fp)
        
    for i in LABEL:
        ner.add_label(i)
    
    optimizer = nlp.resume_training()
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, drop=0.2, sgd=optimizer,
                            losses=losses)
            print("Losses", losses)

    nlp.to_disk(model_spc_output)
    
    return