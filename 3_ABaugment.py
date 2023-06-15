import numpy as np
import pandas as pd
import plots
import utils
import tasks
import pickle
from preprocessing import preprocess_dataset, add_column
from settings import ascending_setting
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from models.baseline import elem_concat, concat
from models.discover_augmentation_v2 import DiscoAugment

import warnings
warnings.filterwarnings('ignore')

props_list = [ 
                # 'bulkmodulus',
                # 'bandgap',
                # 'seebeck',
                # 'rho',
                'sigma',
                # 'shearmodulus'                
              ]

pairs={
        'bulkmodulus'  : ['aflow', 'mp'],   #'mp' (calculated)
        'bandgap'      : ['zhuo', 'mpds'],  #'mp'
        'seebeck'      : ['te', 'mpds'],
        'rho'          : ['te', 'mpds'],
        'sigma'        : ['te', 'mpds'],
        'shearmodulus' : ['aflow', 'mp']    #'mp' (calculated)
        }

tasks_list = [  'crabnet_regression',
                # 'roost_regression',
                # 'linear_regression',
                # 'random_forest_regression',    
                # 'logistic_classification',  
                # 'crabnet_classification'
                ]

models_list = [ 
                'baseline',
                # 'concat',
                # 'elem_concat',
                # 'disco'
                ]


"""global params"""
n_repetitions = 5
# preprocessing
epsilon_T = 15               # controls the window size around ambient temperature
merging='median'              # 'median'/'best' (drop duplicates and save best value) 
med_sigma_multiplier = 0.5  # in 'median' merging values with duplicates with std > 0.5*median are discarted
mult_outliers = 3           # values above mean + 3*sigma are discarted
# split
split = 'random' # 'top' # 'novelty'
shuffle_after_split = True
extraord_size = 0.2                               # best 20% will be extraord.
train_size, val_size, test_size = [0.7, 0.1, 0.2] # % train /val /test
k_val, k_test = [0.33, 0.33]                      # % top for val and test. 
# featurization
elem_prop = 'magpie'
# kwarg
k_elemconcat = 5
n_elemconcat = 10
crabnet_kwargs = {'epochs':300, 'verbose':False, 'discard_n':10}
discover_kwargs = {'exit_mode': 'percentage',  #'thr' / 'percentage'
                   'batch_size': 5,
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
# metrics
metric_reg = 'mae'
metric_class = 'acc'
    
# To store results
iterables = [[task for task in tasks_list], [model for model in models_list]]
columns = pd.MultiIndex.from_product(iterables, names=["task", "model"])
iterables = [props_list, list(range(n_repetitions))]
index =  pd.MultiIndex.from_product(iterables, names=["prop", "rep"])
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
    
    # ITERATE over different random seed to estimate unc.
    for n in range(n_repetitions):
        print(f'### seed = {n+1} ###')
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
        
        """BASELINE"""  
        # does not merge datasets, every dataset is tested on its own test
        if 'baseline' in models_list:
            print(f'--- baseline ---')
            # FEATURIZE TRAIN
            train_feat = utils.featurize(train, elem_prop=elem_prop)
            # TASKS
            output, _ = tasks.apply_all_tasks(train_feat, test_feat, key_A,
                                              tasks_list, crabnet_kwargs,
                                              random_state=seed)
            num_results = [output[task][metric_reg] if ('regression' in task) 
                            else output[task][metric_class] for task in tasks_list]
            cols = results.columns.get_level_values('model')=='baseline'
            results.loc[(prop,n),cols] = num_results
            
        """CONCAT MODEL"""
        if 'concat' in models_list:
            print(f'--- concat ---')
            # CONCATENATE
            train_concat = concat(dfs_dict={key_A:train, key_B:data_B}, merging_opt=merging, 
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
            results.loc[(prop,n),cols] = num_results

        """ELEM CONCAT MODEL"""
        if 'elem_concat' in models_list:
            print(f'--- elem concat ---')
            # CONCATENATE
            train_concat = elem_concat(dfs_dict={key_A:train, key_B:data_B}, merging_opt=merging, 
                                        ascending=ascending_setting[prop],
                                        k=k_elemconcat, n=n_elemconcat, verbose=False,
                                        random_state=seed)
            # FEATURIZE TRAIN
            train_feat = utils.featurize(train_concat, elem_prop=elem_prop)
            # TASKS
            output, _ = tasks.apply_all_tasks(train_feat, test_feat, key_A,
                                              tasks_list, crabnet_kwargs,
                                              random_state=seed)
            num_results = [output[task][metric_reg] if ('regression' in task) 
                            else output[task][metric_class] for task in tasks_list]
            cols = results.columns.get_level_values('model')=='elem_concat'
            results.loc[(prop,n),cols] = num_results

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
            results.loc[(prop,n),cols] = num_results
            
# saving results
# with open('results.pkl', 'wb') as handle:
#     pickle.dump(results, handle)

# saving results (bulk & shear)
with open('results_bulk_shear.pkl', 'wb') as handle:
    pickle.dump(results, handle)

#average across repetitions
results_mean = results.groupby('prop').mean()
results_std = results.groupby('prop').std()
results_table = round(results_mean,3).astype(str) + " ± " + round(results_std,3).astype(str)       

print('\n')
from tabulate import tabulate        
h = [' '] + list(map('\n'.join, results.columns.tolist()))
content = tabulate(results_table, headers=h)
print(content)  
# text_file=open("3_output_mae_new.csv","w")
# text_file.write(content)
# text_file.close()      
