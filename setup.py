#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:09:00 2020

@author: carlosdelafuente
"""

import os
import numpy as np
import pandas as pd
import csv
import logging
import pickle
import json
import csv
import more_itertools as mit
from operator import length_hint


# Setup: este módulo contiene varias funciones que principalemente se usan en
# operaciones de entrada, salida y transformación de datos

def join_data(dset):    
    
    # une todos los ficheros de entrada 'train' o 'dev'

    files = []
    path = os.getcwd() + '/data/' + dset + '/'
    bio_path = os.getcwd() + '/data/' + dset + '/bio/'
    join_file = os.path.join(bio_path, (dset + '_dataset.join'))

    for f_name in os.listdir(path):
        if f_name.endswith('.txt'):
            files.append(f_name)
    with open(join_file, 'w') as outfile:
        for f in files:
            with open(os.path.join(path, f), 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
    outfile.close()
    return      

def join_data_train_dev():    
    
    # une todos los ficheros de entrada 'train' y 'dev' para entrenamiento y reporte final

    files = []
    path_train = os.getcwd() + '/data/train/'
    path_dev = os.getcwd() + '/data/dev/'    
    join_file = os.getcwd() + '/data/train_dev/bio/train_dev_dataset.join'

    for f_name in os.listdir(path_train):
        if f_name.endswith('.txt'):
            files.append(os.path.join(path_train, f_name))
    for f_name in os.listdir(path_dev):
        if f_name.endswith('.txt'):
            files.append(os.path.join(path_dev, f_name))
    with open(join_file, 'w') as outfile:
        for f in files:
            with open(f, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
    return  

def get_test(path):   
    
    # lee todos los archivos de test

    files = []    
    for f_name in os.listdir(path):
        if f_name.endswith('.txt'):
            files.append(f_name)
    return(files) 
    

def get_data(dset):     
    
    # Esta función carga el dataset para el entrenamiento en un df

    bio_dataset = os.getcwd() + '/data/' + dset + '/bio/' + dset + '_dataset.join'
    
    # leo el archivo para calcular el # de negaciones y asi ajustar el # de columnas   
    with open(bio_dataset, 'r') as temp_f:
        col_count = [len(l.split("\t"))+1 for l in temp_f.readlines()]    
    negaciones = int((max(col_count)-8) / 3)
    column_name_neg = []
    for i in range(negaciones):
        column_name_neg+=['cue_'+ str(i), 
                          'sco_'+ str(i), 
                          'eve_'+ str(i)]          
    column_names = ['domain', 'sentence', 'token', 'word', 'lemma', 
                    'PoS', 'PoS_type'] + column_name_neg

    # leo el archivo para generar un dataframe de trabajo con el # de cols ajustado    
    df = pd.read_csv(bio_dataset, delimiter = "\t", encoding = 'UTF-8', dtype=object,
                   header = None, names = column_names, quoting=csv.QUOTE_NONE,
                   skip_blank_lines=False)    
    
    df['id_sentence'] = df['domain'] + '_' + df['sentence'].map(str)
    column_names = df.columns.tolist()
    column_names = column_names[-1:] + column_names[:-1]
    df = df[column_names]
    
    return(df, negaciones)
    
    
def set_bio_frase(bio_frase, negs):     
    
    # Esta función anota cada token con BIO format (claves de negación y scope)
    
    negs_frase = []
    negs_sco_frase = []
    cue_B = []
    sco_B = []
    for i in range(negs):
        negs_frase.append((bio_frase['cue_' + str(i)].nunique()) - 1)
        negs_sco_frase.append((bio_frase['sco_' + str(i)].nunique()) - 1)
        cue_B.append(False)
        sco_B.append(False)
    bio_tag = ['O'] * len(bio_frase)
    sco_tag = ['O'] * len(bio_frase)
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
                    

def tag_dev():
    
    # Esta función anota todos los arcihvos dev para usarlos como test durante la fase de ajuste
    
    path_dev_test = os.getcwd() + '/data/dev/dev_test/'
    path_dev_tagged_test = os.getcwd() + '/data/dev/dev_test/tagged/'
    
    for f_dev in os.listdir(path_dev_test):
        if f_dev.endswith('.txt'):
            # leo el archivo para calcular el # de negaciones y asi ajustar el # de columnas
            
            with open(os.path.join(path_dev_test, f_dev), 'r') as temp_f:
                col_count = [len(l.split("\t"))+1 for l in temp_f.readlines()]    
            negaciones = int((max(col_count)-8) / 3)
            column_name_neg = []
            for i in range(negaciones):
                column_name_neg+=['cue_'+ str(i), 
                                  'sco_'+ str(i), 
                                  'eve_'+ str(i)]          
            column_names = ['domain', 'sentence', 'token', 'word', 'lemma', 
                            'PoS', 'PoS_type'] + column_name_neg
        
            # leo el archivo para generar un dataframe de trabajo con el # de cols ajustado    
            df = pd.read_csv(os.path.join(path_dev_test, f_dev), delimiter = "\t", 
                    encoding = 'UTF-8', header = None, quoting=csv.QUOTE_NONE,
                    names = column_names, skip_blank_lines=False)

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
                    index=False, quoting=csv.QUOTE_NONE, encoding = 'UTF-8')
    return


def get_dev_4_training():
    
    # Esta función genera archivos para el entrenamento a partir de los dev originales
    
    path_dev = os.getcwd() + '/data/dev/'
    path_dev_gold_test = os.getcwd() + '/data/test/test_dev/'
    
    for f_dev in os.listdir(path_dev):
        if f_dev.endswith('.txt'):
            with open(os.path.join(path_dev, f_dev), 'r') as temp_f_dev:          
                df_temp_f_dev = pd.read_csv(temp_f_dev, delimiter = "\t", header = None, 
                            usecols=[0, 1, 2, 3, 4, 5, 6], encoding = 'UTF-8', index_col = False,
                            dtype=object, quoting=csv.QUOTE_NONE, skip_blank_lines=False)

                df_temp_f_dev.to_csv(os.path.join(path_dev_gold_test, f_dev), 
                            sep = '\t', header = None, 
                            encoding = 'UTF-8', index = False,
                            quoting=csv.QUOTE_NONE)    
    return

def get_max_negs(frases_d_test):
    
    # Esta función la usa bio_conll para calcular el número máx de negaciones de un archivo
    
    b=[]
    max_bs = 0
    for f in frases_d_test:
        b = []
        for t in f:
            b.append(t[2])
        bs = b.count('B-Cue')
        if bs > max_bs: max_bs = bs
    return(max_bs)


def get_columnas_b_i_cue(cue_b_i, cue_b_i_pos):
    columnas = []
    grupo = [cue_b_i[0]]
    grupo_pos = [cue_b_i_pos[0]]
    grupo_final = []
    grupo_pos_final = []
    for i , j in zip(cue_b_i[1:], cue_b_i_pos[1:]):
        if i == 'I-Cue':
            grupo.append(i)
            grupo_pos.append(j)
        else:
            grupo_final.append(grupo)
            grupo = []
            grupo.append(i)
            grupo_pos_final.append(grupo_pos)
            grupo_pos = []
            grupo_pos.append(j)
    grupo_final.append(grupo)
    grupo_pos_final.append(grupo_pos)
            
    for i in range(len(grupo_final)):
        columnas.append('cue_' + str(i)) 
        
    return(columnas, grupo_pos_final)
  

def get_columnas_b_i_scope(cue_b_i, cue_b_i_pos):   
    columnas = []
    grupo = [cue_b_i[0]]
    grupo_pos = [cue_b_i_pos[0]]
    grupo_final = []
    grupo_pos_final = []
    for i , j, k in zip(cue_b_i[1:], cue_b_i_pos[1:], range(len(cue_b_i_pos[1:]))):
        if i == 'I-Sco' and (cue_b_i_pos[k+1] - cue_b_i_pos[k] == 1):
            grupo.append(i)
            grupo_pos.append(j)
        else:
            grupo_final.append(grupo)
            grupo = []
            grupo.append(i)
            grupo_pos_final.append(grupo_pos)
            grupo_pos = []
            grupo_pos.append(j)
    grupo_final.append(grupo)
    grupo_pos_final.append(grupo_pos)
            
    for i in range(len(grupo_final)):
        columnas.append('sco_' + str(i)) 
        
    return(columnas, grupo_pos_final)

    
def bio_conll(tecnica, f_name):
    
    # Esta función transforma un archivo con anotación BIO en Cue y Scope a Conll
    
    f_mame='moviles'
    
    if tecnica == 'crf':
        input_file = os.getcwd() + '/outputs/output_crf_cue_scope_test_' + f_name + '.txt'
        output_conll = os.getcwd() + '/scorer/output_crf_test_cue_scope_' + f_name + '_cll.txt'
    else:
        input_file = os.getcwd() + '/outputs/output_ner_cue_scope_test_' + f_name + '.txt'
        output_conll = os.getcwd() + '/scorer/output_ner_test_cue_scope_' + f_name + '_cll.txt'
    
    col_names = ['domain', 'sentence', 'token', 'word', 'lemma', 
                            'PoS', 'PoS_type', 'bio_tag', 'sco_tag']
    
    d_test = pd.read_csv(input_file, delimiter = "\t", encoding = 'UTF-8', 
                         index_col=False, quoting=csv.QUOTE_NONE,
                         names = col_names, dtype=object, 
                         skip_blank_lines=False)
    
    d_test.insert(0,'inx', [i for i in range(len(d_test))])
    
    agg_func = lambda s: [(inx, word, bio_tag, sco_tag) for inx, word, bio_tag, sco_tag
                          in zip(s['inx'].values.tolist(),
                                 s['word'].values.tolist(), 
                                 s['bio_tag'].values.tolist(),
                                 s['sco_tag'].values.tolist(),)]
    
    grouped_d_test = d_test.groupby(['domain', 'sentence']).apply(agg_func)
    frases_d_test = [f for f in grouped_d_test]
    
    max_ngs = get_max_negs(frases_d_test)
    
    column_neg = ['token']
    for i in range(max_ngs):
        column_neg+=['cue_'+ str(i), 
                          'sco_'+ str(i), 
                          'eve_'+ str(i)]      
    
    d_conll = pd.DataFrame(columns = column_neg)
    
    for f in frases_d_test:
        fragmento_d_conll = pd.DataFrame(columns = column_neg)
        
        token = []
        b = []      # almacena cue-tags
        bs = []     # almacena sco-tags
        w = []
        for t in f:
            token.append(t[0])
            w.append(t[1])
            b.append(t[2])
            bs.append(t[3])
        fragmento_d_conll['token'] = token
        
            
        posiciones_b_cue = [pos_b for pos_b, x in enumerate(b) if x == 'B-Cue']
        posiciones_i_cue = [pos_b for pos_b, x in enumerate(b) if x == 'I-Cue']
        # posiciones_o_cue = [pos_b for pos_b, x in enumerate(b) if x == 'O-Cue']
        posiciones_b_sco = [pos_b for pos_b, x in enumerate(bs) if x == 'B-Sco']
        posiciones_i_sco = [pos_b for pos_b, x in enumerate(bs) if x == 'I-Sco']
        # posiciones_o_sco = [pos_b for pos_b, x in enumerate(bs) if x == 'O-Sco']
            
        max_ngs = max(len(posiciones_b_cue), len(posiciones_i_cue), 
                      len(posiciones_b_sco), len(posiciones_i_sco))
        
        if max_ngs == 0:
            fragmento_d_conll['cue_0'] = '***'    
        else:
            
            cue_b_i = []
            for item in b:
                if item != 'O-Cue':
                    cue_b_i.append(item)
                
            sco_b_i = []
            for item in bs:
                if item != 'O-Sco':
                    sco_b_i.append(item)
                    
            if cue_b_i: 
                posiciones_b_i_cue = sorted(posiciones_b_cue + posiciones_i_cue)
                cols_cue, posiciones_cue = get_columnas_b_i_cue(cue_b_i, posiciones_b_i_cue)
                for col_cue, pos_cue in zip(cols_cue, posiciones_cue):
                    for item in pos_cue:
                        fragmento_d_conll.loc[item, col_cue] = w[item]  
            
            if sco_b_i:
                posiciones_b_i_sco = sorted(posiciones_b_sco + posiciones_i_sco)
                cols_scope, posiciones_sco = get_columnas_b_i_scope(sco_b_i, posiciones_b_i_sco)        
                for col_sco, pos_sco in zip(cols_scope, posiciones_sco):
                    for item in pos_sco:
                        fragmento_d_conll.loc[item, col_sco] = w[item]           
            for n_col in range (fragmento_d_conll.shape[1]):
                for n_row in range (fragmento_d_conll.shape[0]):
                    if pd.isna(fragmento_d_conll.iloc[n_row, n_col]):
                        fragmento_d_conll.iloc[n_row, n_col] = '-'
    
        fragmento_d_conll = fragmento_d_conll.append(pd.Series(), ignore_index=True)   
        d_conll = pd.concat([d_conll, fragmento_d_conll], axis=0)   
    
    # Append blank lines
        
    new_rows = []
    for i in range(len(d_conll)):
        if pd.isna(d_conll.iloc[i]['token']):
            new_rows.append(d_conll.iloc[i-1]['token']+1)
        else:
            new_rows.append(0)
    
    d_conll['token'].fillna(0, inplace= True)
    d_conll['token'] = d_conll['token'] + new_rows
    d_conll = d_conll.sort_values(by=['token'])
    d_conll.fillna('', inplace = True)
    
    d_conll.rename(columns={'token':'inx'}, inplace=True)
    d_conll = pd.merge(d_conll, d_test, on=['inx'])
    
    d_conll = d_conll.loc[:, (col_names[:-2] + column_neg[1:])]
    
    # Archivo conll a disco
     
    with open(output_conll,'w',newline='') as output_file:     
        writer = csv.writer(output_file,delimiter='\t',quotechar='', lineterminator='\n', quoting=csv.QUOTE_NONE)
        for line in d_conll.values:
            if pd.isna(line[0]):
                line = ''
            writer.writerow(line)

    return


def tsv_to_json_format(input_path,output_path,unknown_label):
    
    # Conveirte TSV a JSON
    
    try:
        f=open(input_path,'r') # input file
        fp=open(output_path, 'w') # output file
        data_dict={}
        annotations =[]
        label_dict={}
        s=''
        start=0
        for line in f:
            if line[0]!='\t':
                word,entity=line.split('\t')
                s+=word+" "
                entity=entity[:len(entity)-1]
                if entity!=unknown_label:
                    if len(entity) != 1:
                        d={}
                        d['text']=word
                        d['start']=start
                        d['end']=start+len(word)-1  
                        try:
                            label_dict[entity].append(d)
                        except:
                            label_dict[entity]=[]
                            label_dict[entity].append(d) 
                start+=len(word)+1
            else:
                data_dict['content']=s
                s=''
                label_list=[]
                for ents in list(label_dict.keys()):
                    for i in range(len(label_dict[ents])):
                        if(label_dict[ents][i]['text']!=''):
                            l=[ents,label_dict[ents][i]]
                            label_list.append(l)                          
                            
                for entities in label_list:
                    label={}
                    label['label']=[entities[0]]
                    label['points']=entities[1:]
                    annotations.append(label)
                data_dict['annotation']=annotations
                annotations=[]
                json.dump(data_dict, fp, ensure_ascii=False)
                fp.write('\n')
                data_dict={}
                start=0
                label_dict={}
    except Exception as e:
        logging.exception("Unable to process file" + "\n" + "error = " + str(e))
        return None


def convert_json_spacy_format(input_file=None, output_file=None):
    
    # Convierte JSON a SPACY Format
    
    try:
        training_data = []
        lines=[]
        with open(input_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                labels = annotation['label']
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    entities.append((point['start'], point['end'] + 1 ,label))

            training_data.append((text, {"entities" : entities}))

        with open(output_file, 'wb') as fp:
            pickle.dump(training_data, fp)

    except Exception as e:
        logging.exception("Unable to process " + input_file + "\n" + "error = " + str(e))
        return None


def tsv_spacy_cue_format():
    
    # Esta funcion transforma TSV (BIO) -> TSV (BIO) 2 COLS -> JSON -> SPACY FORMAT
    
    datasets = ['train', 'dev', 'train_dev']
       
    for dset in datasets:
        bio_dataset_input = os.getcwd() + '/data/' + dset + '/bio/' + dset + '_dataset_bio_o.txt'
        bio_two_cols_input = os.getcwd() + '/data/' + dset + '/bio/' + dset + '_two_cols_bio_o.txt'
        bio_json_output = os.getcwd() + '/data/' + dset + '/bio/' + dset + '_dataset_bio_o.json'
        bio_spc_output = os.getcwd() + '/data/' + dset + '/bio/' + dset + '_dataset_bio_o.spc'
    
        df = pd.read_csv(bio_dataset_input, delimiter = "\t", encoding = 'UTF-8', 
                          quoting=csv.QUOTE_NONE, skip_blank_lines=False)
        
        df = df[['word','bio_tag']]
        df.to_csv(bio_two_cols_input, sep = '\t',  index = None, header = None, encoding = 'UTF-8')
            
        tsv_to_json_format(bio_two_cols_input, bio_json_output, 'abc')        
        convert_json_spacy_format(bio_json_output, bio_spc_output)
        
    return


def tsv_spacy_scope_format():
    
    # Esta funcion transforma TSV (BIO) -> TSV (BIO) 2 COLS -> JSON -> SPACY FORMAT
    # para Scopes
    
    datasets = ['train', 'dev', 'train_dev']
       
    for dset in datasets:
        bio_dataset_input = os.getcwd() + '/data/' + dset + '/bio/' + dset + '_dataset_bio_o.txt'
        bio_two_cols_input = os.getcwd() + '/data/' + dset + '/bio/' + dset + '_two_cols_bio_scope_o.txt'
        bio_json_output = os.getcwd() + '/data/' + dset + '/bio/' + dset + '_dataset_bio_scope_o.json'
        bio_spc_output = os.getcwd() + '/data/' + dset + '/bio/' + dset + '_dataset_bio_scope_o.spc'
    
        df = pd.read_csv(bio_dataset_input, delimiter = "\t", encoding = 'UTF-8', 
                          quoting=csv.QUOTE_NONE, skip_blank_lines=False)
        
        df = df[['word', 'sco_tag']]
        df.to_csv(bio_two_cols_input, sep = '\t',  index = None, header = None, encoding = 'UTF-8')
            
        tsv_to_json_format(bio_two_cols_input, bio_json_output, 'abc')        
        convert_json_spacy_format(bio_json_output, bio_spc_output)
        
    return


def genera_cue_scope(tecnica):
    
    # Esta función concatena tags de clave y scope en un nuevo archivo para transformar a Conll
    
    if tecnica == 'ner':
        path_cue_test = os.getcwd() + '/outputs/output_ner_cue_test_'
        path_scope_test = os.getcwd() + '/outputs/output_ner_scope_test_'
        path_cue_scope_test = os.getcwd() + '/outputs/output_ner_cue_scope_test_'
    else:
        path_cue_test = os.getcwd() + '/outputs/output_crf_cue_test_'
        path_scope_test = os.getcwd() + '/outputs/output_crf_scope_test_'
        path_cue_scope_test = os.getcwd() + '/outputs/output_crf_cue_scope_test_'        
    
    filenames = ['coches', 'lavadoras', 'hoteles', 'moviles', 'libros', 'musica', 
             'ordenadores', 'peliculas']
    
    for f_name in filenames:    
        test_cue = path_cue_test + f_name + '.txt'
        test_scope = path_scope_test + f_name + '.txt'
        test_cue_scope = path_cue_scope_test + f_name + '.txt'
        with open(test_cue, 'r') as cue_f, open(test_scope, 'r') as scope_f, open(test_cue_scope, 'w') as cue_scope_f:
            
            cue_test = pd.read_csv(cue_f, delimiter = "\t", 
                  encoding = 'UTF-8', dtype=object, header = None, quoting=csv.QUOTE_NONE, skip_blank_lines=False)

            scope_test = pd.read_csv(scope_f, delimiter = "\t", 
                  encoding = 'UTF-8', dtype=object, header = None, quoting=csv.QUOTE_NONE, skip_blank_lines=False)
            
            cue_scope_f = pd.concat([cue_test, scope_test[7]], axis=1)
            
            cue_scope_f.to_csv(test_cue_scope, sep = '\t', index = None, quoting=csv.QUOTE_NONE,
                               header = None, encoding = 'UTF-8')
    return


