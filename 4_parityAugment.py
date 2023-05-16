import pandas as pd
import numpy as np
import plotly.io as pio
from preprocessing import preprocess_dataset, add_column
import settings
import tasks
import matplotlib.pyplot as plt
from models.baseline import concat
from settings import ascending_setting
from sklearn.preprocessing import MinMaxScaler
import utils
import warnings
import torch
from models.discover_augmentation_v2 import DiscoAugment

warnings.filterwarnings('ignore')

pio.renderers.default="svg"    # 'svg' or 'browser'
pio.templates.default="simple_white"

device = torch.device('cpu')

props_list = [ 
                # 'bulkmodulus',
                # 'bandgap',
                # 'seebeck',
                'rho',
                # 'sigma',
                # 'shearmodulus'   
        
            ]

split='random'
model = 'disco'
n_repetitions = 5
med_sigma_multiplier = 0.5  # in 'median' merging values with duplicates with std > 0.5*median are discarted
epsilon_T = 5               # controls the window size around ambient temperature
mult_outliers = 3           # values above mean + 3*sigma are discarted
n_top = 5
reg_method = 'random_forest_regression'
merging='median'              # 'median'/'best' (drop duplicates and save best value) 
shuffle_after_split = True
extraord_size = 0.2                               # best 20% will be extraord.
train_size, val_size, test_size = [0.7, 0.1, 0.2] # % train /val /test
k_val, k_test = [0.33, 0.33]
metric = 'mae'

discover_kwargs = {'thresh' : 0.9, 
                   'n_iter':5,
                   'batch_size':30,
                   'scaled' : True,
                   'scaler' : MinMaxScaler, 
                   'density_weight':1.0,
                   'target_weight':1.0,
                   'scores': ['density']
                   }

crabnet_kwargs = {'epochs':2, 'verbose':False, 'discard_n':10}

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
                            extraord_size, 
                            ascending_setting[prop])
    
    outputs={'mae':[],
             'mse':[],
             'R2':[],
             'mre':[]}
    
    '''Training and computing class-MAE before augmentation'''
    for n in range(n_repetitions):
        print(f'{n+1}-',end='')
        random_state = n
        # train/test split of acceptor dataset
        train, _, test = tasks.apply_split(split_type = split,
                                           df = data_clean[key_star],
                                           test_size=test_size,
                                           random_state=random_state,
                                           shuffle=shuffle_after_split,
                                           verbose=False)
        
        # print(f'\n\t---Baseline model on {key_star}')
        train_feat = utils.featurize(train, elem_prop='magpie')
        test_feat  = utils.featurize(test, elem_prop='magpie')
        '''tasks'''
        #out: global scores, freq_df_before: scores on occ before augment.
        out, freq_df_before = tasks.apply_all_tasks(train_feat, test_feat, 
                                                    key_star, [reg_method], 
                                                    crabnet_kwargs,
                                                    random_state=random_state,
                                                    verbose=False)
        
        '''Applying augmentation''' #acceptor training augmented with all mpds.        
        if model == 'concat':
            augmented_df = concat(dfs_dict={key_star:train, 'mpds':data_clean['mpds']}, 
                                  merging=settings.merging, 
                                  elem_prop='magpie', ascending=ascending_setting[prop])
            
        elif model =='disco':
            disco = DiscoAugment({key_star : train, 
                                  'mpds':data_clean['mpds'].iloc[:30]},
                                 random_state=random_state)

            augmentations = disco.apply_augmentation(
                                    **discover_kwargs,
                                    crabnet_kwargs = crabnet_kwargs)

            augmented_df = augmentations[-1]
    
        '''Training and computing class-MAE after augmentation'''
        #New train dataset is the augmented one, test remains the same from acceptor
        train_feat = utils.featurize(augmented_df, elem_prop='magpie')
        test_feat  = utils.featurize(test, elem_prop='magpie')
        '''tasks'''
        out, freq_df_after = tasks.apply_all_tasks(train_feat, test_feat, 
                                                    key_star, [reg_method], 
                                                    crabnet_kwargs,
                                                    random_state=random_state,
                                                    verbose=False)
        
        freq_df_complete_before = pd.concat([freq_df_complete_before, freq_df_before], axis=0)
        freq_df_complete_after = pd.concat([freq_df_complete_after, freq_df_after], axis=0)
    
    # Computing class scores before and after augmentation
    class_score_before = freq_df_complete_before.groupby('elem_test', sort=False).mean().reset_index(drop=False)
    
    std_devs = freq_df_complete_before.groupby('elem_test', sort=False).std().reset_index(drop=False)
    # we don't need occ_test but just for completeness.
    class_score_before['occ_test_std'] = std_devs['occ_test']
    class_score_before['occ_train_std'] = std_devs['occ_train']
    class_score_before[f'{reg_method}_{metric}_std'] = std_devs[f'{reg_method}_{metric}']

    class_score_after = freq_df_complete_after.groupby('elem_test', sort=False).mean().reset_index(drop=False)
    std_devs          = freq_df_complete_after.groupby('elem_test', sort=False).std().reset_index(drop=False)
    
    class_score_after[f'{reg_method}_{metric}_std'] = std_devs[f'{reg_method}_{metric}']
    class_score_after['occ_test_std'] = std_devs['occ_test']
    class_score_after['occ_train_std'] = std_devs['occ_train']
    class_score_after[f'{reg_method}_{metric}_std'] = std_devs[f'{reg_method}_{metric}']

    # Test elements with highest MAE and highest train occurrence before doing the augmentation.
    worst_class_score_before = class_score_before.sort_values(f'{reg_method}_{metric}', ascending=False)
    best_class_score_before  = class_score_before.sort_values('occ_train', ascending=False)
    
    worst_class_score_before = worst_class_score_before.iloc[:n_top]
    worst_elems_before       = list(worst_class_score_before['elem_test'])
    
    best_class_score_before = best_class_score_before.iloc[:n_top]
    best_elems_before       = list(best_class_score_before['elem_test'])
    
    print(f'\n--- Elements {worst_elems_before} have highest {metric} in test set. ---\n')
    print(f'\n--- Elements {best_elems_before} have highest occ_train. ---\n')

    # Check the new MAE for worst(high test mae) and best (high occ train) elements after augmentation
    worst_class_score_after = class_score_after[class_score_after['elem_test'].isin(worst_elems_before)]
    best_class_score_after = class_score_after[class_score_after['elem_test'].isin(best_elems_before)]

    '''Plotting augmentation results with respect to worst elems'''
    # sorting by alphabetic order
    worst_class_score_before = worst_class_score_before.sort_values('elem_test')
    worst_class_score_after  = worst_class_score_after.sort_values('elem_test')
    
    best_class_score_before  = best_class_score_before.sort_values('elem_test')
    best_class_score_after   = best_class_score_after.sort_values('elem_test')
    
    # scatterplot    
    fig, ax = plt.subplots(nrows=1,  ncols=2,figsize=(14,9))
    
    ax[0].set_title('Worst elems (high MAE)')
    # ax[0].errorbar(x=worst_class_score_before['count_train'],
    #                xerr = worst_class_score_before['count_std'],
    #                yerr = worst_class_score_before['score_std'],
    #                y=worst_class_score_before['MAE'],
    #                fmt='o',
    #                color='blue',
    #                label='Before augment')
    
    ax[0].scatter(x=worst_class_score_before['occ_train'],
                  y=worst_class_score_before[f'{reg_method}_{metric}'],
                  color='blue',
                  label='Before augment')
        
    count=0
    for x,y in zip(worst_class_score_before['occ_train'], 
                   worst_class_score_before[f'{reg_method}_{metric}']):
        ax[0].text(x,y,worst_class_score_before.iloc[count]['elem_test'])
        count+=1
    
    # ax[0].errorbar(x=worst_class_score_after['count_train'],
    #                xerr = worst_class_score_after['count_std'],
    #                yerr = worst_class_score_after['score_std'],
    #                y=worst_class_score_after['MAE'],
    #                fmt='o',
    #                color='orange',
    #                label='After augment'
    #                )
    
    ax[0].scatter(x=worst_class_score_after['occ_train'],
                  y=worst_class_score_after[f'{reg_method}_{metric}'],
                  color='orange',
                  label='After augment'
                  )
    
    count=0
    for x,y in zip(worst_class_score_after['occ_train'], 
                   worst_class_score_after[f'{reg_method}_{metric}']):
        ax[0].text(x,y,worst_class_score_after.iloc[count]['elem_test'])
        count+=1
    
    # ax[0].set_xscale('log')
    
    ax[1].set_title('Best elems (high count train)')

    # ax[1].errorbar(x=best_class_score_before['count_train'],
    #                xerr = best_class_score_before['count_std'],
    #                yerr = best_class_score_before['score_std'],
    #                y=best_class_score_before['MAE'],
    #                fmt='o',
    #                color='blue',
    #                label='Before augment'
    #                )
    
    ax[1].scatter(x=best_class_score_before['occ_train'],
                  y=best_class_score_before[f'{reg_method}_{metric}'],
                  color='blue',
                  label='Before augment')
    
    count=0
    for x,y in zip(best_class_score_before['occ_train'], 
                   best_class_score_before[f'{reg_method}_{metric}']):
        ax[1].text(x,y,best_class_score_before.iloc[count]['elem_test'])
        count+=1
    
    # ax[1].errorbar(x=best_class_score_after['count_train'], 
    #                xerr = best_class_score_after['count_std'],
    #                yerr = best_class_score_after['score_std'],
    #                y=best_class_score_after['MAE'],
    #                fmt='o',
    #                color='orange',
    #                label='After augment')
    
    ax[1].scatter(x=best_class_score_after['occ_train'], 
                  y=best_class_score_after[f'{reg_method}_{metric}'],
                  color='orange',
                  label='After augment')
        
    count=0
    for x,y in zip(best_class_score_after['occ_train'], 
                   best_class_score_after[f'{reg_method}_{metric}']):
        ax[1].text(x,y,best_class_score_after.iloc[count]['elem_test'])
        count+=1
        
    ax[1].set_xscale('log')
    ax[0].set_ylabel(f'{metric}')
    
    plt.legend()
    
    # barplot (worst elems)
    fig, ax= plt.subplots(nrows=1,ncols=2,figsize=(12,6))
    
    elems           = tuple(worst_class_score_before['elem_test'])
    
    mean_mae_before = list(worst_class_score_before['MAE'])
    std_mae_before  = list(worst_class_score_before['score_std'])

    mean_mae_after  = list(worst_class_score_after['MAE'])
    std_mae_after   = list(worst_class_score_after['score_std'])
    
    x = np.arange(len(elems))
    width = 0.25  #width of bars.
    multiplier = 0
    
    bar1 = ax[0].bar(x-width/2, 
                  mean_mae_before, 
                  width, 
                  yerr=std_mae_before if not np.isnan(std_mae_before).any() else None, 
                  label='Before augment')
    
    bar2 = ax[0].bar(x+width/2, 
                  mean_mae_after, 
                  width, 
                  yerr=std_mae_after if not np.isnan(std_mae_after).any() else None, 
                  label ='After augment')
    
    # ax[0].set_title(f'{prop} elements score comparison')
    
    ax[0].tick_params(
        axis='x',
        which='major',
        direction='in',
        width=2.5,
        length=10)
    
    ax[0].set_ylabel('MAE')
    ax[0].set_xticks(x, elems)
    
    # barplot (best elems)
    elems           = tuple(best_class_score_before['elem_test'])
    mean_mae_before = list(best_class_score_before['MAE'])
    std_mae_before  = list(best_class_score_before['score_std'])

    mean_mae_after  = list(best_class_score_after['MAE'])
    std_mae_after   = list(best_class_score_after['score_std'])
    
    x = np.arange(len(elems))
    width = 0.25  #width of bars.
    multiplier = 0
    
    bar1 = ax[1].bar(x-width/2, 
                  mean_mae_before, 
                  width, 
                  yerr=std_mae_before if not np.isnan(std_mae_before).any() else None,
                  label='Before augment')
    
    bar2 = ax[1].bar(x+width/2, 
                  mean_mae_after, 
                  width, 
                  yerr=std_mae_after if not np.isnan(std_mae_after).any() else None,
                  label ='After augment')
    
    # ax[0].set_title(f'{prop} elements score comparison')
    
    ax[1].tick_params(
        axis='x',
        which='major',
        direction='in',
        width=2.5,
        length=10)
    
    ax[1].set_xticks(x, elems)
    
    plt.legend()
    
