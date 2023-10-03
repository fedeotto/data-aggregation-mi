import pandas as pd
import numpy as np
import plotly.io as pio
from assets.preprocessing import preprocess_dataset, add_column
import settings
from assets import tasks
import matplotlib.pyplot as plt
from models.baseline import concat, elem_concat
from settings import ascending_setting
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from assets import utils
import warnings
import torch
from models.discover_augmentation_v2 import DiscoAugment

warnings.filterwarnings('ignore')

pio.renderers.default="svg"    # 'svg' or 'browser'
pio.templates.default="simple_white"

plt.rcParams['font.size'] = 25
plt.rcParams['figure.dpi'] = 400
device = torch.device('cpu')

"""PROPERTIES"""
props_list = [ 
                'bulkmodulus',
                # 'bandgap',
                # 'seebeck',
                # 'rho',
                # 'sigma',
                'shearmodulus'                
              ]

reg_method = 'random_forest_regression'
tasks_list = [reg_method]
model = 'disco'
n_top = 5

"""global params"""
n_repetitions = 5
# preprocessing
epsilon_T = 15               #controls the window size around ambient temperature
merging='median'              #'median'/'best' (drop duplicates and save best value) 
med_sigma_multiplier = 0.5  #in 'median' merging values with duplicates with std > 0.5*median are discarted
mult_outliers = 3           #values above mean + 3*sigma are discarted
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

metric = 'mae'

# main loop
for prop in props_list:
    freq_df_complete_before = pd.DataFrame()        
    freq_df_complete_after = pd.DataFrame()        

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
        print(f'### seed = {n+1} ###')
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
        
        '''tasks'''
        print(f'--- tasks before augmentation ---')
        # FEATURIZE TRAIN
        train_feat = utils.featurize(train, elem_prop=elem_prop)
        # TASKS
        #out: global scores, freq_df_before: scores on occ before augment.
        out, freq_df_before = tasks.apply_all_tasks(train_feat, test_feat, key_A,
                                                    tasks_list, crabnet_kwargs,
                                                    reg_metrics = [metric],
                                                    random_state=random_state, verbose=False)
        num_results_before = [out[task][metric] for task in tasks_list]
        
        print(f'--- computing augmentation ---')
        '''Applying augmentation''' #acceptor training augmented with all mpds.        
        if model == 'concat':
            # CONCATENATE
            augmented_df = concat(dfs_dict={key_A: train, key_B: data_B}, 
                                  merging_opt=merging, 
                                  ascending=ascending_setting[prop])
            
        elif model == 'elem_concat':
            # CONCATENATE
            augmented_df = elem_concat(dfs_dict={key_A: train, key_B: data_B}, 
                                        merging_opt=merging, 
                                        ascending=ascending_setting[prop],
                                        k=k_elemconcat, n=n_elemconcat, verbose=True,
                                        random_state=random_state)
            
            
        elif model =='disco':
            # CONCATENATE
            DAM = DiscoAugment(dfs_dict={key_A: train, key_B: data_B},
                                self_augment_frac = None, # initial fraction for self_aumgent
                                random_state = random_state)
            
            augmentations = DAM.apply_augmentation(crabnet_kwargs=crabnet_kwargs,
                                                    **discover_kwargs)
            augmented_df = augmentations[-1]
    
        print(f'--- tasks after augmentation ---')
        '''Training and computing class-MAE after augmentation'''
        # New train dataset is the augmented one, test remains the same from A
        train_feat = utils.featurize(augmented_df, elem_prop='magpie')
        
        '''tasks'''
        out, freq_df_after = tasks.apply_all_tasks(train_feat, test_feat, key_A,
                                                    tasks_list, crabnet_kwargs,
                                                    reg_metrics = [metric],
                                                    random_state=random_state, verbose=False)
        num_results_after = [out[task][metric] for task in tasks_list]
        
        freq_df_complete_before = pd.concat([freq_df_complete_before, freq_df_before], axis=0)
        freq_df_complete_after = pd.concat([freq_df_complete_after, freq_df_after], axis=0)
    
    # group iterations, get mean and std
    
    # before
    before = freq_df_complete_before.groupby('elem_test', sort=False).mean()
    stds = freq_df_complete_before.groupby('elem_test', sort=False).std()
    for col in stds.columns:
        before[f'{col}_std'] = stds[f'{col}']
        
    # after
    after = freq_df_complete_after.groupby('elem_test', sort=False).mean()
    stds = freq_df_complete_after.groupby('elem_test', sort=False).std()
    for col in stds.columns:
        after[f'{col}_std'] = stds[f'{col}']

    # Check the new MAE for worst (high test mae) and best (high occ train) elements before augmentation
    worst_before = before.sort_values(f'{reg_method}_{metric}', ascending=False).iloc[:n_top]
    best_before  = before.sort_values('occ_train', ascending=False).iloc[:n_top]
    print(f'\n--- Elements {list(worst_before.index)} have highest {metric} in test set. ---\n')
    print(f'\n--- Elements {list(best_before.index)} have highest occ_train. ---\n')
    
    # same after augmentation
    worst_after = after.loc[worst_before.index]
    best_after  = after.loc[best_before.index]

    '''Plotting augmentation results with respect to worst elems'''
    # sorting by alphabetic order
    worst_before = worst_before.sort_index()
    worst_after  = worst_after.sort_index()
    best_before  = best_before.sort_index()
    best_after   = best_after.sort_index()
    
    # scatterplot    
    fig, ax = plt.subplots(nrows=1,  ncols=2,figsize=(20,12))
    
    plt.rcParams['font.size'] = 16
    
    markers_size = 145
    fontsize     = 16
    offset_y     = 0.018
    offset_x     = -0.001
    
    ax[0].scatter(x=worst_before['occ_train'],
                  y=worst_before[f'{reg_method}_{metric}'],
                  s=markers_size,
                  edgecolor='k',
                  color='blue',
                  label='Before aggregation')
        
    # count=0
    # for x,y in zip(worst_before['occ_train'], 
    #                worst_before[f'{reg_method}_{metric}']):
    #     ax[0].text(x + offset_x ,y + offset_y,worst_before.index[count],fontsize=fontsize)
    #     count+=1
    
    ax[0].scatter(x=worst_after['occ_train'],
                  y=worst_after[f'{reg_method}_{metric}'],
                  s=markers_size,
                  edgecolor='k',
                  color='orange',
                  label='After aggregation'
                  )
    
    # count=0
    # for x,y in zip(worst_after['occ_train'], 
    #                worst_after[f'{reg_method}_{metric}']):
    #     ax[0].text(x+offset_x, y + offset_y,worst_after.index[count], fontsize=fontsize)
    #     count+=1

    ax[1].scatter(x=best_before['occ_train'],
                  y=best_before[f'{reg_method}_{metric}'],
                  s=markers_size,
                  edgecolor='k',
                  color='blue',
                  label='Before aggregation')
        
    # count=0
    
    # offset_y     = 0.010
    # offset_x     = -15
    # for x,y in zip(best_before['occ_train'], 
    #                best_before[f'{reg_method}_{metric}']):
    #     ax[1].text(x+offset_x, y+offset_y,best_before.index[count], fontsize=fontsize)
    #     count+=1
    
    ax[1].scatter(x=best_after['occ_train'], 
                  y=best_after[f'{reg_method}_{metric}'],
                  s=markers_size,
                  edgecolor='k',
                  color='orange',
                  label='After aggregation')
        
    # count=0
    # for x,y in zip(best_after['occ_train'], 
    #                best_after[f'{reg_method}_{metric}']):
    #     ax[1].text(x+offset_x,y+offset_y,best_after.index[count], fontsize=fontsize)
    #     count+=1
    
    # xticks = np.arange(0, best_after['occ_train'].max(), 100)
    # ax[1].set_xticks(xticks)

    ax[0].set_ylabel('MAE', labelpad=15)
    ax[0].set_xlabel('Train occurrences', labelpad=15)
    ax[1].set_xlabel('Train occurrences', labelpad=15)

    plt.legend()
    plt.savefig('parity_bandgap.png')
    
    # END SCATTERPLOT
    
    # barplot (worst elems)
    # fig, ax= plt.subplots(nrows=1,ncols=2,figsize=(12,6))
    
    # elems           = tuple(worst_before.index)
    
    # mean_mae_before = list(worst_before[f'{reg_method}_{metric}'])
    # std_mae_before  = list(worst_before[f'{reg_method}_{metric}_std'])

    # mean_mae_after  = list(worst_after[f'{reg_method}_{metric}'])
    # std_mae_after   = list(worst_after[f'{reg_method}_{metric}_std'])
    
    # x = np.arange(len(elems))
    # width = 0.25  #width of bars.
    # multiplier = 0
    
    # bar1 = ax[0].bar(x-width/2, 
    #               mean_mae_before, 
    #               width, 
    #               yerr=std_mae_before if not np.isnan(std_mae_before).any() else None, 
    #               label='Before augment')
    
    # bar2 = ax[0].bar(x+width/2, 
    #               mean_mae_after, 
    #               width, 
    #               yerr=std_mae_after if not np.isnan(std_mae_after).any() else None, 
    #               label ='After augment')
    
    # # ax[0].set_title(f'{prop} elements score comparison')
    
    # ax[0].tick_params(
    #     axis='x',
    #     which='major',
    #     direction='in',
    #     width=2.5,
    #     length=10)
    
    # ax[0].set_ylabel('MAE')
    # ax[0].set_xticks(x)
    # ax[0].set_xticklabels(elems)
    
    # # barplot (best elems)
    # elems           = tuple(best_before.index)
    # mean_mae_before = list(best_before[f'{reg_method}_{metric}'])
    # std_mae_before  = list(best_before[f'{reg_method}_{metric}_std'])

    # mean_mae_after  = list(best_after[f'{reg_method}_{metric}'])
    # std_mae_after   = list(best_after[f'{reg_method}_{metric}_std'])
    
    # x = np.arange(len(elems))
    # width = 0.25  #width of bars.
    # multiplier = 0
    
    # bar1 = ax[1].bar(x-width/2, 
    #               mean_mae_before, 
    #               width, 
    #               yerr=std_mae_before if not np.isnan(std_mae_before).any() else None,
    #               label='Before augment')
    
    # bar2 = ax[1].bar(x+width/2, 
    #               mean_mae_after, 
    #               width, 
    #               yerr=std_mae_after if not np.isnan(std_mae_after).any() else None,
    #               label ='After augment')
    
    # # ax[0].set_title(f'{prop} elements score comparison')
    
    # ax[1].tick_params(
    #                     axis='x',
    #                     which='major',
    #                     direction='in',
    #                     width=2.5,
    #                     length=10
    #                     )
    
    # ax[1].set_xticks(x)
    # ax[1].set_xticklabels(elems)
    
    # plt.legend()
    








