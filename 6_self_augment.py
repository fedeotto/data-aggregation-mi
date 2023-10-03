"""this script reproduce Fig. 3 of the paper:
Not as simple as we thought: a rigorous examination of data aggregation in materials informatics
09 August 2023, Version 1
Federico Ottomano, Giovanni De Felice, Vladimir Gusev, Taylor Sparks """

import pandas as pd
import numpy as np
import plotly.io as pio
from assets import plots
from assets import utils
from assets import tasks
from assets.preprocessing import preprocess_dataset, add_column
import settings
from settings import *
import matplotlib.pyplot as plt
from models.baseline import concat, elem_concat
from sklearn.preprocessing import MinMaxScaler
import itertools
import warnings
import pickle
import torch
from models.discover_augmentation_v2 import DiscoAugment
from models.random_augmentation import RandomAugment
from settings import *

warnings.filterwarnings('ignore')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

pio.renderers.default="svg"    # 'svg' or 'browser'
pio.templates.default="simple_white"

"""PROPERTIES"""
props_list = [ 'bulkmodulus' ]      # 'thermalcond',
                                    # 'superconT',
                                    # 'seebeck',
                                    # 'rho'
                                    # 'sigma',
                                    # 'bandgap',
                                    # 'bulkmodulus',
                                    # 'shearmodulus'

"""TASKS""" #override from settings
tasks_list = ['random_forest_regression',
              'crabnet_regression']


"""SETTINGS""" #override from settings
initial_size                  = 0.05 #initial size of self-augmented dataset
discover_kwargs['percentage'] = 1

#DataFrame to store results
columns      = pd.MultiIndex.from_product([tasks_list, ['disco','random'],range(1,n_repetitions+1)]) 
results      = {f'{prop}': pd.DataFrame(data=np.nan, columns=columns, index=range(1000)) for prop in props_list}

# main loop
def plot_all():
    for prop in props_list:     
        """LOADING"""
        # load datsets
        data_raw = utils.load_dataset(prop)  
        keys_list = list(data_raw.keys())
        key_A = pairs[prop][0]; assert key_A != 'mpds'
        key_B = pairs[prop][1]
        utils.print_info(data_raw, prop); print('')
        
        """PREPROCESSING"""
        # preprocessing
        data_clean = preprocess_dataset(data_raw, prop, merging,
                                        epsilon_T, 
                                        med_sigma_multiplier,
                                        mult_outliers,
                                        ascending_setting[prop])
        print(''); utils.print_info(data_clean, prop)
        
        # add extraord column to all datasets(0/1)
        data_clean = add_column(data_clean, extraord_size, ascending_setting[prop])
        
        '''Training and computing class-MAE before augmentation'''
        for n in range(n_repetitions):
            print(f'\n### seed = {n+1} ###\n')
            random_state = n
            # SPLIT DATASETS IN TRAIN AND TEST
            train, _, test = tasks.apply_split(split_type = split,
                                                df = data_clean[key_A],
                                                val_size=0, test_size=test_size, k_test=k_test,
                                                random_state=random_state,
                                                ascending=ascending_setting[prop],
                                                shuffle=shuffle_after_split)
            data_B = data_clean[key_B]
            
            # FEATURIZE TEST
            test_feat  = utils.featurize(test, elem_prop=elem_prop)
            
            discover_kwargs['batch_size'] = int(0.05*len(train))
            # running Discover augmentation
            print('performing disco augmentation')
            DAM = DiscoAugment(dfs_dict={key_A: train, key_B: train},
                            self_augment_frac = initial_size, # initial fraction of A for self_aumgent
                            random_state = random_state)
            
            disco_augmentations = DAM.apply_augmentation(crabnet_kwargs=crabnet_kwargs,
                                                            **discover_kwargs)
            
            rnd_kwargs['batch_size'] = int(0.05*len(train))
            #Running rnd_augment
            print('performing random augmentation')
            rnd_augment = RandomAugment(dfs_dict={key_A: train, key_B: train},
                                        self_augment_frac = initial_size, # initial fraction for self_aumgent
                                        random_state = random_state)
            
            rnd_augmentations = rnd_augment.apply_augmentation(**rnd_kwargs)
            
            # check
            last_rnd    = rnd_augmentations[-1].sort_values(by=['target', 'formula'], axis=0).reset_index(drop=True)
            last_disco  = disco_augmentations[-1].sort_values(by=['target', 'formula'], axis=0).reset_index(drop=True)
            train_sorted = train.sort_values(by=['target', 'formula'], axis=0).reset_index(drop=True)
            assert len(disco_augmentations)==len(rnd_augmentations)
            pd.testing.assert_frame_equal(train_sorted, last_disco)
            pd.testing.assert_frame_equal(last_disco, last_rnd)
            
            
            print(f'performing tasks for disco augmentations...')
            # scores = []
            print('it = ', end=' ')
            for i,augment in enumerate(disco_augmentations):
                print(f'{i}...', end=' ')
                train_feat = utils.featurize(augment, elem_prop=elem_prop)
                
                '''tasks'''
                out, _ = tasks.apply_all_tasks(train_feat, test_feat, key_A,
                                                tasks_list, crabnet_kwargs,
                                                reg_metrics = ['mae'],
                                                clas_metrics = ['acc'],
                                                random_state=random_state, verbose=False)
                
                for task in tasks_list:
                    if task != 'logistic_classification':
                        results[f'{prop}'].loc[:,(f'{task}','disco', n+1)][i] = out[task]['mae']
                    else:
                        results[f'{prop}'].loc[:,(f'{task}','disco', n+1)][i] = out[task]['acc']
                        
            #     scores.append(out[reg_method][metric])
            #     scores.append(out[class_method][metric])
            # print('')
            # results[prop].loc[:,('disco',n+1)] = pd.Series(data=scores)
            
            print(f'performing tasks for random augmentations...')
            # scores = []
            print('it = ', end=' ')
            for i,augment in enumerate(rnd_augmentations):
                print(f'{i}...', end=' ')
                train_feat = utils.featurize(augment, elem_prop=elem_prop)
                
                '''tasks'''
                out, _ = tasks.apply_all_tasks(train_feat, test_feat, key_A,
                                            tasks_list, crabnet_kwargs,
                                            reg_metrics = ['mae'],
                                            clas_metrics = ['acc'],
                                            random_state=random_state, verbose=False)
                
                for task in tasks_list:
                    if task != 'logistic_classification':
                        results[f'{prop}'].loc[:,(f'{task}','random', n+1)][i] = out[task]['mae']
                    else:
                        results[f'{prop}'].loc[:,(f'{task}','random', n+1)][i] = out[task]['acc']
                
                # scores.append(out[reg_method][metric])
                # scores.append(out[class_method][metric])
            # print('')
            # results[prop].loc[:,('random',n+1)] = pd.Series(data=scores)
        
        if split == 'novelty':
            with open(f'./results/results_6_discotest_{prop}.pkl', 'wb') as handle:
                pickle.dump(results[f'{prop}'], handle)
        else:
            with open(f'./results/results_6_{prop}.pkl', 'wb') as handle:
                pickle.dump(results[f'{prop}'], handle)
    
    # plot
    plots.plot_self_augment(props_list[0],
                            discotest=True if split=='novelty' else False)

if __name__ == '__main__': plot_all()

    # clean result data
    # results[prop] = results[prop].drop(np.where(results[prop].isna())[0],axis=0)
    # get x axis labels    
    # ratios = np.array([len(my_augmentations[i])/len(train) for i in range(len(my_augmentations))])    
    # assert (ratios==np.array([len(rnd_augmentations[i])/len(train) for i in range(len(rnd_augmentations))])).all()
    # results[prop].index = ratios.round(2)
    
    # x = np.array(results[prop].index)
    
    # means_disco = np.array(results[prop].mean(axis=1, level=0)['disco'].values)
    # stds_disco = np.array(results[prop].std(axis=1, level=0)['disco'].values)
    
    # means_random = np.array(results[prop].mean(axis=1, level=0)['random'].values)
    # stds_random = np.array(results[prop].std(axis=1, level=0)['random'].values)
    
    # fig, ax = plt.subplots(figsize=(8,6))
    # ax.plot(x,
    #         means_disco,
    #         color='#2c7fb8',
    #         linestyle='--',
    #         marker='o',
    #         markersize=5,
    #         label='DiSCoVeR')
    
    # ax.fill_between(x,
    #                 means_disco-stds_disco, 
    #                 means_disco+stds_disco,
    #                 color='#2c7fb8',
    #                 alpha=0.1)
    
    # ax.plot(x,
    #         means_random,
    #         color='#31a354',
    #         linestyle='--',
    #         marker='o',
    #         markersize=5,
    #         label='Random')
    
    # ax.fill_between(x,
    #                 means_random-stds_random, 
    #                 means_random+stds_random,
    #                 color='#31a354',
    #                 alpha=0.1)
    
    # ax.grid()
        
    # ax.set_title(f'{prop} self_augment')
    # ax.set_xlabel('Train ratio (%)', labelpad=10)
    # ax.set_ylabel('Accuracy', labelpad=10)
    
    # plt.legend(loc='upper right')
            