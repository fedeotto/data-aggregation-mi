import pandas as pd
import numpy as np
import plotly.io as pio
from preprocessing import preprocess_dataset, add_column
import settings
import tasks
import matplotlib.pyplot as plt
from models.baseline import concat, elem_concat
from settings import ascending_setting
from sklearn.preprocessing import MinMaxScaler
import utils
import warnings
import torch
from models.discover_augmentation_v2 import DiscoAugment
from models.random_augmentation import RandomAugment

warnings.filterwarnings('ignore')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

pio.renderers.default="svg"    # 'svg' or 'browser'
pio.templates.default="simple_white"


props_list = [ 
                # 'bulkmodulus',
                'bandgap',
                # 'seebeck',
                # 'rho',
                # 'sigma',
                # 'shearmodulus'                
              ]

pairs={
        'bulkmodulus'  : ['aflow', 'mpds'],   #'mp'
        'bandgap'      : ['zhuo', 'mpds'],    #'mp'
        'seebeck'      : ['te',   'mpds'],
        'rho'          : ['te', 'mpds'],
        'sigma'        : ['te', 'mpds'],
        'shearmodulus' : ['aflow', 'mpds']   #'mp'
        }


reg_method   = 'random_forest_regression'
class_method = 'logistic_classification'
tasks_list = [class_method]
model = 'disco'
initial_size = 0.05
batch_size = 100

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



rnd_kwargs      = {'exit_mode': 'percentage',    # percentage or iters
                   'batch_size': batch_size,
                   'n_iters': 10,     # used if 'exit_mode' = 'iters'
                   'percentage': 1.,  # used if 'exit_mode' = 'percentage'
                   }

discover_kwargs = {'exit_mode': 'percentage',  #'thr' / 'percentage'
                   'batch_size': batch_size,
                   #------
                   # in threshold mode
                   'thresh' : 0.9999, 
                   # in percentage mode
                   'percentage' : 1.,
                   #------
                   'scaled' : True,
                   'scaler' : MinMaxScaler(), 
                   'density_weight':1.0,
                   'target_weight':1.0,
                   'scores': ['density']
                   }

metric = 'acc'
columns = pd.MultiIndex.from_product([['disco','random'],range(1,n_repetitions+1)]) 
results = {f'{prop}': pd.DataFrame(data=np.nan, columns=columns, index=range(1000)) for prop in props_list}

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
    
    '''Training and computing class-MAE before augmentation'''
    for n in range(n_repetitions):
        print(f'\n### seed = {n+1} ###\n')
        random_state = n
        # SPLIT DATASETS IN TRAIN AND TEST
        train, _, test = tasks.apply_split(split_type = split,
                                           df = data_clean[key_A],
                                           val_size=0, test_size=0.2, k_test=0.5,
                                           random_state=random_state,
                                           ascending=ascending_setting[prop],
                                           shuffle=shuffle_after_split)
        data_B = data_clean[key_B]
        
        # FEATURIZE TEST
        test_feat  = utils.featurize(test, elem_prop=elem_prop)
    
        
        
        # running Discover augmentation
        print('performing disco augmentation')
        DAM = DiscoAugment(dfs_dict={key_A: train, key_B: train},
                           self_augment_frac = initial_size, # initial fraction for self_aumgent
                           random_state = random_state)
        
        my_augmentations = DAM.apply_augmentation(crabnet_kwargs=crabnet_kwargs,
                                                  **discover_kwargs)
        
        #Running rnd_augment
        print('performing random augmentation')
        rnd_augment = RandomAugment(dfs_dict={key_A: train, key_B: train},
                                    self_augment_frac = initial_size, # initial fraction for self_aumgent
                                    random_state = random_state)
        
        rnd_augmentations = rnd_augment.apply_augmentation(**rnd_kwargs)
        
        # check
        last_rnd = rnd_augmentations[-1].sort_values(by=['target', 'formula'], axis=0).reset_index(drop=True)
        last_my  = my_augmentations[-1].sort_values(by=['target', 'formula'], axis=0).reset_index(drop=True)
        train_sorted = train.sort_values(by=['target', 'formula'], axis=0).reset_index(drop=True)
        assert len(my_augmentations)==len(rnd_augmentations)
        pd.testing.assert_frame_equal(train_sorted, last_my)
        pd.testing.assert_frame_equal(last_my, last_rnd)
        
        
        
        print(f'performing {reg_method} for disco augmentations...')
        scores = []
        print('it = ', end=' ')
        for i,augment in enumerate(my_augmentations):
            print(f'{i}...', end=' ')
            train_feat = utils.featurize(augment, elem_prop=elem_prop)
            
            '''tasks'''
            out, _ = tasks.apply_all_tasks(train_feat, test_feat, key_A,
                                           tasks_list, crabnet_kwargs,
                                           reg_metrics = [metric],
                                           clas_metrics = [metric],
                                           random_state=random_state, verbose=False)
            
            # scores.append(out[reg_method][metric])
            scores.append(out[class_method][metric])
        print('')
        results[prop].loc[:,('disco',n+1)] = pd.Series(data=scores)
        
        
        print(f'performing {reg_method} for random augmentations...')
        scores = []
        print('it = ', end=' ')
        for i,augment in enumerate(rnd_augmentations):
            print(f'{i}...', end=' ')
            train_feat = utils.featurize(augment, elem_prop=elem_prop)
             
            '''tasks'''
            out, _ = tasks.apply_all_tasks(train_feat, test_feat, key_A,
                                           tasks_list, crabnet_kwargs,
                                           reg_metrics = [metric],
                                           clas_metrics = [metric],
                                           random_state=random_state, verbose=False)
            
            # scores.append(out[reg_method][metric])
            scores.append(out[class_method][metric])
        print('')
        results[prop].loc[:,('random',n+1)] = pd.Series(data=scores)    
    
    # clean result data
    results[prop] = results[prop].drop(np.where(results[prop].isna())[0],axis=0)
    # get x axis labels    
    ratios = np.array([len(my_augmentations[i])/len(train) for i in range(len(my_augmentations))])    
    assert (ratios==np.array([len(rnd_augmentations[i])/len(train) for i in range(len(rnd_augmentations))])).all()
    results[prop].index = ratios.round(2)
    
    
    x = np.array(results[prop].index)
    
    means_disco = np.array(results[prop].mean(axis=1, level=0)['disco'].values)
    stds_disco = np.array(results[prop].std(axis=1, level=0)['disco'].values)
    
    means_random = np.array(results[prop].mean(axis=1, level=0)['random'].values)
    stds_random = np.array(results[prop].std(axis=1, level=0)['random'].values)
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(x,
            means_disco,
            color='#2c7fb8',
            linestyle='--',
            marker='o',
            markersize=5,
            label='DiSCoVeR')
    
    ax.fill_between(x,
                    means_disco-stds_disco, 
                    means_disco+stds_disco,
                    color='#2c7fb8',
                    alpha=0.1)
    
    ax.plot(x,
            means_random,
            color='#31a354',
            linestyle='--',
            marker='o',
            markersize=5,
            label='Random')
    
    ax.fill_between(x,
                    means_random-stds_random, 
                    means_random+stds_random,
                    color='#31a354',
                    alpha=0.1)
    
    ax.grid()
        
    # ax.set_title(f'{prop} self_augment')
    ax.set_xlabel('Train ratio (%)', labelpad=10)
    ax.set_ylabel('Accuracy', labelpad=10)
    
    # plt.legend(loc='upper right')
            