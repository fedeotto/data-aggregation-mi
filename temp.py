#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:36:48 2023

@author: federico
"""

import pandas as pd
import numpy as np
from chem import _element_composition
from collections import Counter



    
    # for elem in all_symbols:
        
    #     temp = df[df['formula'].str.contains(elem)]
    #     # score = score_evaluation(targets, preds, metric)
    #     test_elems_dict[elem] = score
        
    #     el_train_freq = formulae_train.str.contains(elem).sum()
    #     train_elems_frequency.append(el_train_freq)
    
    # freq_df = pd.DataFrame(None)
    # freq_df['test_elems'] = all_symbols
    # freq_df[f'{metric}'] = list(test_elems_dict.values())

all_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
               'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
               'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
               'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
               'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
               'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
               'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
               'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
               'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
               'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
               'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
               'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

targets = np.random.randn(100)
preds = np.random.randn(100)
formulae_train = pd.read_csv('./datasets/bulkmodulus_aflow.csv')['formula'].iloc[:100]
formulae_test = pd.read_csv('./datasets/bulkmodulus_aflow.csv')['formula'].iloc[100:200]

elem_class_score(targets, 
                 preds, 
                 formulae_train, 
                 formulae_test)
       
       
       
# def calc_diversity(df: pd.DataFrame):
    
#     train_dicts = df['formula'].apply(_element_composition)
#     train_list = [item for row in train_dicts for item in row.keys()]
#     train_counter = Counter(train_list)
#     trainc_df = pd.DataFrame.from_dict(train_counter, orient='index', columns=['count'])
#     count_col = trainc_df['count']
    
#     diversity = Entropy(count_col, base=2) / (np.log2(len(count_col)))
    
#     return diversity







