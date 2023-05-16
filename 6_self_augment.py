import pandas as pd
import numpy as np
from collections import Counter
import plotly.graph_objects as go
import plotly.io as pio
from preprocessing import preprocess_dataset, add_column
import settings
import tasks
import matplotlib.pyplot as plt
from models.baseline import concat
from settings import ascending_setting
import utils
import warnings
import plots
import torch
from models.discover_augmentation import DiscoAugment
from models.random_augmentation import RandomAugment

warnings.filterwarnings('ignore')

pio.renderers.default="svg"    # 'svg' or 'browser'
pio.templates.default="simple_white"

device = torch.device('cpu')

props_list = [ 
                'bulkmodulus',
                # 'bandgap',
                # 'seebeck',
                # 'rho',
                # 'sigma',
                # 'shearmodulus'       
                
            ]

split='random'
n_repetitions = 5
med_sigma_multiplier = 0.5  # in 'median' merging values with duplicates with std > 0.5*median are discarted
epsilon_T = 5               # controls the window size around ambient temperature
mult_outliers = 3           # values above mean + 3*sigma are discarted
n_top = 5
reg_method = 'random_forest_regression'
merging='median'              # 'median'/'best' (drop duplicates and save best value) 
shuffle_after_split = True
elem_prop='magpie'
self_augment = 0.1
model = 'disco' 
by_least_novel = True

crabnet_kwargs = {'epochs':50, 'verbose':False, 'discard_n':10}
discover_kwargs = {'thresh' : 0.0, 
                   'n_iter':100000, 
                   'batch_size':30, 
                   'proxy_weight':1.0,
                   'pred_weight':1.0,
                   'clusters' : False}

rnd_kwargs      = {'n_iter' : 100000,
                   'batch_size' : 30}

# main loop
for prop in props_list:
    freq_df_complete_before = pd.DataFrame()        
    freq_df_complete_after = pd.DataFrame()        

    """LOADING"""
    # load datsets
    data_raw = utils.load_dataset(prop)  
    keys_list = list(data_raw.keys())
    key_star = keys_list[0]; assert key_star != 'mpds'
    utils.print_info(data_raw, prop); print('')
    
    """PREPROCESSING"""
    # preprocessing
    data_clean = preprocess_dataset(data_raw, 
                                    prop, 
                                    merging,
                                    epsilon_T, 
                                    med_sigma_multiplier,
                                    mult_outliers,
                                    ascending_setting[prop])
    
    print(''); utils.print_info(data_clean, prop)
    
    # dictionary of clean datasets
    data_clean = add_column(data_clean, 
                            settings.extraord_size, 
                            settings.ascending_setting[prop])
    
    dict_results_disco = {k : [] for k in range(len(data_clean[key_star]))}
    dict_results_random = {k : [] for k in range(len(data_clean[key_star]))}
    
    '''Training and computing class-MAE before augmentation'''
    for n in range(n_repetitions):
        print(f'{n+1}-',end='')
        random_state = n
        train, _, test = tasks.apply_split(split_type = split,
                                           df = data_clean[key_star],
                                           test_size=settings.test_size,
                                           random_state=random_state,
                                           shuffle=shuffle_after_split,
                                           verbose=False)
        
        # test remains fixed
        test_feat = utils.featurize(test, elem_prop=elem_prop)
    
        #Running disco
        disco = DiscoAugment({key_star : train, 'mpds':data_clean['mpds']},
                 a_key=list(data_clean)[0], 
                 d_key=list(data_clean)[1],
                 self_augment=self_augment)
        
        augmentations = disco.apply_augmentation(
                                **discover_kwargs,
                                by_least_novel=by_least_novel,
                                random_state = random_state,
                                crabnet_kwargs = crabnet_kwargs)
        
        scores = []

        for i,augment in enumerate(augmentations):
            train_feat = utils.featurize(augment, elem_prop=elem_prop)
            print(reg_method)
            out, _ = tasks.apply_all_tasks(train_feat, test_feat, 
                                           key_star, [reg_method], 
                                           crabnet_kwargs,
                                           random_state=random_state,
                                           verbose=False)
            
            score = out[reg_method][key_star]['mae']
            dict_results_disco[i].append(score)
            
        #Running rnd_augment
        rnd_augment = RandomAugment({key_star : train, 'mpds':data_clean['mpds']}, 
                                    a_key=list(data_clean)[0], 
                                    d_key=list(data_clean)[1],
                                    self_augment = self_augment)
        
        augmentations = rnd_augment.apply_augmentation(**rnd_kwargs,
                                                       random_state=random_state)
        
        for i,augment in enumerate(augmentations):
            train_feat = utils.featurize(augment, elem_prop=elem_prop)
            out, _ = tasks.apply_all_tasks(train_feat, test_feat, 
                                           key_star, [reg_method], 
                                           crabnet_kwargs,
                                           random_state=random_state,
                                           verbose=False)
            
            score = out[reg_method][key_star]['mae']
            dict_results_random[i].append(score)
            
    reduced_dict_disco = {k:v for (k,v) in dict_results_disco.items() if v}
    reduced_dict_random = {k:v for (k,v) in dict_results_random.items() if v}
    
    means_disco = np.array([np.array(v).mean() for v in reduced_dict_disco.values()])
    stds_disco = np.array([np.array(v).std() for v in reduced_dict_disco.values()])
    
    means_random = np.array([np.array(v).mean() for v in reduced_dict_random.values()])
    stds_random = np.array([np.array(v).std() for v in reduced_dict_random.values()])
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(list(reduced_dict_disco.keys()),
            means_disco,
            linestyle='--',
            marker='o',
            markersize=3,
            label='Disco')
    
    ax.fill_between(list(reduced_dict_disco.keys()),
                    means_disco-stds_disco, 
                    means_disco+stds_disco,
                    alpha=0.3)
    
    ax.plot(list(reduced_dict_random.keys()),
            means_random,
            linestyle='--',
            marker='o',
            markersize=3,
            label='Random')
    
    ax.fill_between(list(reduced_dict_random.keys()),
                    means_random-stds_random, 
                    means_random+stds_random,
                    alpha=0.3)
    
    ax.set_title(f'{prop} self_augment')
    ax.set_xlabel('n_iter')
    ax.set_ylabel('MAE')
    
    plt.legend()
            