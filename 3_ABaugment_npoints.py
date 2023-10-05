import numpy as np
import pandas as pd

import assets.plots as plots
from assets import utils, tasks
from assets.preprocessing import preprocess_dataset, add_column
import pickle
from settings import ascending_setting
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt

from models.baseline import elem_concat, concat
from models.discover_augmentation_v2 import DiscoAugment
from settings import *

import warnings
warnings.filterwarnings('ignore')

"""PROPERTIES"""
props_list = [ 'rho' ]      # 'thermalcond',
                            # 'superconT',
                            # 'seebeck',
                            # 'rho'
                            # 'sigma',
                            # 'bandgap',
                            # 'bulkmodulus',
                            # 'shearmodulus'

"""TASKS"""
tasks_list = [
                # 'linear_regression',
                'random_forest_regression',    
                # 'crabnet_regression',
                # 'logistic_classification',  
                # 'crabnet_classification'
                ]

"""MODELS"""
models_list = [ 
                # 'baseline',
                'concat',
                # 'elem_concat',
                'disco'
                ]



"""SETTINGS""" #override from settings.py
extraord_size = 0.2                               # best 20% will be extraord.
train_size, val_size, test_size = [0.7, 0.1, 0.2] # % train /val /test
k_val, k_test = [0.33, 0.33]                      # % top for val and test.
ratios = np.arange(0.05, 1.05, 0.05).round(3)
k_elemconcat   = 5
n_elemconcat   = 10
crabnet_kwargs = {'epochs':300, 'verbose':False, 'discard_n':10}
discover_kwargs = {'exit_mode': 'percentage',  #'thr' / 'percentage'
                   'batch_size': 20,
                   #------
                   # in threshold mode
                   'thresh' : 0.9999, 
                   # in percentage mode
                   'percentage' : 0.1,
                   #------
                   'scaled' : True,
                   'scaler' : RobustScaler(), 
                   'density_weight':1.0,
                   'target_weight':1.0,
                   'scores': ['density']
                   }
metric_reg = 'mae'
metric_class = 'acc'
    
# To store results
iterables = [[task for task in tasks_list], [model for model in models_list]]
columns = pd.MultiIndex.from_product(iterables, names=["task", "model"])
iterables = [props_list, list(range(n_repetitions)), list(ratios.round(2))]
index =  pd.MultiIndex.from_product(iterables, names=["prop", "rep", "ratios"])
results = pd.DataFrame(columns=columns, index=index)

# main loop
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
    
    for ratio in ratios:
        # ITERATE over different random seed to estimate unc.
        for n in range(n_repetitions):
            print(f'\n### seed = {n+1} ###\n')
            seed = n
            
            # SPLIT DATASETS IN TRAIN AND TEST
            train, _, test = tasks.apply_split(split_type = split,
                                               df = data_clean[key_A],
                                               val_size=0, test_size=0.2, k_test=0.5,
                                               random_state=seed,
                                               ascending=ascending_setting[prop],
                                               shuffle=shuffle_after_split)
            data_B = data_clean[key_B]
            
            # FEATURIZE TEST
            test_feat  = utils.featurize(test, elem_prop=elem_prop)
            
            # """BASELINE"""  
            # # does not merge datasets, every dataset is tested on its own test
            # if 'baseline' in models_list:
            #     print(f'--- baseline ---')
            #     # FEATURIZE TRAIN
            #     train_feat = utils.featurize(train, elem_prop=elem_prop)
            #     # TASKS
            #     output, _ = tasks.apply_all_tasks(train_feat, test_feat, key_A,
            #                                       tasks_list, crabnet_kwargs,
            #                                       random_state=seed)
            #     num_results = [output[task][metric_reg] if ('regression' in task) 
            #                    else output[task][metric_class] for task in tasks_list]
            #     cols = results.columns.get_level_values('model')=='baseline'
            #     results.loc[(prop,n),cols] = num_results
            
            
            # n_points = int(len(data_B)*ratio)
            
            """CONCAT MODEL"""
            if 'concat' in models_list:
                print(f'--- concat ---')
                # CONCATENATE
                train_concat = concat(dfs_dict={key_A:train, 
                                                key_B:data_B.sample(frac=ratio, random_state=seed)}, 
                                      merging_opt=merging, 
                                      ascending=ascending_setting[prop])
                # FEATURIZE TRAIN
                train_feat = utils.featurize(train_concat, elem_prop=elem_prop)
                # TASKS
                output, _ = tasks.apply_all_tasks(train_feat, test_feat, key_A,
                                                  tasks_list, crabnet_kwargs,
                                                  random_state=seed)
                num_results = [output[task][metric_reg] if ('regression' in task) 
                               else output[task][metric_class] for task in tasks_list]
                cols = results.columns.get_level_values('model')=='concat'
                results.loc[(prop,n,ratio),cols] = num_results
        
            # """ELEM CONCAT MODEL"""
            # if 'elem_concat' in models_list:
            #     print(f'--- elem concat ---')
            #     # CONCATENATE
            #     train_concat = elem_concat(dfs_dict={key_A:train, key_B:data_B}, merging_opt=merging, 
            #                                ascending=ascending_setting[prop],
            #                                k=k_elemconcat, n=n_elemconcat, verbose=False,
            #                                random_state=seed)
            #     # FEATURIZE TRAIN
            #     train_feat = utils.featurize(train_concat, elem_prop=elem_prop)
            #     # TASKS
            #     output, _ = tasks.apply_all_tasks(train_feat, test_feat, key_A,
            #                                       tasks_list, crabnet_kwargs,
            #                                       random_state=seed)
            #     num_results = [output[task][metric_reg] if ('regression' in task) 
            #                    else output[task][metric_class] for task in tasks_list]
            #     cols = results.columns.get_level_values('model')=='elem_concat'
            #     results.loc[(prop,n),cols] = num_results
        
            discover_kwargs['percentage'] = ratio
            """DISCOVER MODEL"""
            if 'disco' in models_list:
                print(f'--- discover ---')
                # CONCATENATE
                DAM = DiscoAugment(dfs_dict={key_A:train, key_B:data_B},
                                   self_augment_frac = None,         # initial fraction for self_aumgent
                                   random_state = seed)
                
                aug_list = DAM.apply_augmentation(crabnet_kwargs=crabnet_kwargs,
                                                  **discover_kwargs)
                train_discoaug = aug_list[-1]
                # FEATURIZE TRAIN
                train_feat = utils.featurize(train_discoaug, elem_prop=elem_prop)
                # TASKS
                output, _ = tasks.apply_all_tasks(train_feat, test_feat, key_A,
                                                  tasks_list, crabnet_kwargs,
                                                  random_state=seed)
                num_results = [output[task][metric_reg] if ('regression' in task) 
                               else output[task][metric_class] for task in tasks_list]
                cols = results.columns.get_level_values('model')=='disco'
                results.loc[(prop,n,ratio),cols] = num_results
                    
                
                
            
            
# saving results
with open('results_ratios_bandgap.pkl', 'wb') as handle:
    pickle.dump(results, handle)
        
#average across repetitions
results_mean = results.groupby('ratios').mean()
results_std = results.groupby('ratios').std()
# results_table = round(results_mean,3).astype(str) + " ± " + round(results_std,3).astype(str)       

# print('\n')
# from tabulate import tabulate        
# h = [' '] + list(map('\n'.join, results.columns.tolist()))
# content = tabulate(results_table, headers=h)
# print(content)  
# text_file=open("3_output_mae_new.csv","w")
# text_file.write(content)
# text_file.close()      

x = np.array(results_mean.index)

means_disco = np.array(results_mean.loc[:,(tasks_list[0],'disco')].values)
stds_disco = np.array(results_std.loc[:,(tasks_list[0],'disco')].values)

means_random = np.array(results_mean.loc[:,(tasks_list[0],'concat')].values)
stds_random = np.array(results_std.loc[:,(tasks_list[0],'concat')].values)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x,
        means_disco,
        linestyle='--',
        marker='o',
        markersize=5,
        label='DiSCoVeR')

ax.fill_between(x,
                means_disco-stds_disco, 
                means_disco+stds_disco,
                alpha=0.1)

ax.plot(x,
        means_random,
        linestyle='--',
        marker='o',
        markersize=5,
        label='Random')

ax.fill_between(x,
                means_random-stds_random, 
                means_random+stds_random,
                alpha=0.1)

# ax.set_title(f'{prop} self_augment')
ax.set_xlabel('MPDS ratio')
ax.set_ylabel('MAE')

plt.legend(loc='upper left')







    