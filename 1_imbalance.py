import numpy as np
import pandas as pd

import plots
import utils
import tasks
from preprocessing import preprocess_dataset, add_column
from settings import ascending_setting

props_list = [ 
                # 'bulkmodulus',
                # 'bandgap',
                # 'seebeck',
                'rho',
                # 'sigma',
                # 'shearmodulus'                
              ]
pairs={
        'bulkmodulus'  : ['aflow', 'mpds'],   #'mp'
        'bandgap'      : ['zhuo', 'mpds'],    #'mp'
        'seebeck'      : ['te', 'mpds'],
        'rho'          : ['te', 'mpds'],
        'sigma'        : ['te', 'mpds'],
        'shearmodulus' : ['aflow', 'mpds']   #'mp'
        }

"""global params"""
n_repetitions = 10
# preprocessing
epsilon_T = 15               # controls the window size around ambient temperature
merging='median'            # 'median'/'best' (drop duplicates and save best value) 
med_sigma_multiplier = 0.5  # in 'median' merging values with duplicates with std > 0.5*median are discarted
mult_outliers = 3           # values above mean + 3*sigma are discarted
# split
split = 'random' # 'top' # 'novelty'
shuffle_after_split = True
extraord_size = 0.2                               # best 20% will be extraord.
train_size, val_size, test_size = [0.7, 0.1, 0.2] # % train /val /test
k_val, k_test = [0.33, 0.33]                      # % top for val and test. 
# featurization
crabnet_kwargs = {'epochs':100, 'verbose':False}

task = 'random_forest_regression'   #'random_forest_regression'   # crabnet_regression
metric = 'mae'

elem_prop = 'magpie'


# main loop
for prop in props_list:
    freq_df_complete = pd.DataFrame()        
    """LOADING"""
    # load datsets
    data_raw = utils.load_dataset(prop)  
    keys_list = list(data_raw.keys())
    key_A = pairs[prop][0]; assert key_A != 'mpds'
    key_B = pairs[prop][1]
    # utils.print_info(data_raw, prop); print('')
    
    
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
    
    # add extraord column to all datasets(0/1)
    data_clean = add_column(data_clean, extraord_size, ascending_setting[prop])
    
    outputs={
        'mae':[],
        'mse':[],
        'mape':[],
        'r2':[],
        'mre':[],
        # 'acc':[]
        }
    
    print('\niters:')
    for n in range(n_repetitions):
        print(f'{n+1}-',end='')
        random_state = n
        train, _, test = tasks.apply_split(split_type = split,
                                           df = data_clean[key_A],
                                           test_size=test_size,
                                           random_state=random_state,
                                           verbose=False,
                                           shuffle=shuffle_after_split)
        # print(f'\n\t---Baseline model on {key_A}')
        train = utils.featurize(train, elem_prop=elem_prop)
        test  = utils.featurize(test, elem_prop=elem_prop)
        '''tasks'''
        out, freq_df = tasks.apply_all_tasks(train, test, 
                                             key_A, [task], 
                                             crabnet_kwargs,
                                             random_state=random_state,
                                             verbose=False)
        freq_df_complete = pd.concat([freq_df_complete, freq_df], axis=0)
        
        for score in outputs.keys():
            # outputs[score].append(out[task][key_A][score])
            outputs[score].append(out[task][score])


    # print('\n')
    
    means = freq_df_complete.groupby('elem_test', sort=True).mean().loc[:,['occ_train', f'{task}_{metric}']]
    stds  = freq_df_complete.groupby('elem_test', sort=True).std().loc[:,['occ_train', f'{task}_{metric}']]
    stds.columns = [f'{col}_std' for col in stds.columns]
    of_interest = pd.concat([means,stds], axis=1)
    plots.plot_elem_class_score_matplotlib(of_interest, task, metric, prop, web=True)
    # plots.plot_elem_class_score(of_interest, task, metric, prop, web=True)

    print('\n')
    for score in outputs.keys():
        print(f'AVERAGE {score} = {round(np.mean(outputs[score]),3)} ', end='')
        print(f'+- {round(np.std(outputs[score]),3)}')
    
    








