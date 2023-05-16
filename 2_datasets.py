import numpy as np
import pandas as pd

import plots
import utils
import tasks
from preprocessing import preprocess_dataset, add_column
from settings import ascending_setting

props_list = [ 
                # 'bulkmodulus',
                'bandgap',
                # 'seebeck',
                # 'rho',
                # 'sigma',
                # 'shearmodulus'                
              ]

"""global params"""
# preprocessing
epsilon_T = 15               # controls the window size around ambient temperature
merging='median'              # 'median'/'best' (drop duplicates and save best value) 
med_sigma_multiplier = 0.5  # in 'median' merging values with duplicates with std > 0.5*median are discarted
mult_outliers = 5           # values above mean + 3*sigma are discarted
# split
extraord_size = 0.2                               # best 20% will be extraord.
train_size, val_size, test_size = [0.7, 0.1, 0.2] # % train /val /test
k_val, k_test = [0.33, 0.33]                      # % top for val and test. 
# featurization
crabnet_kwargs = {'epochs':100, 'verbose':False}

split = 'random'

    


# main loop
for prop in props_list:
     
    
    freq_df_complete = pd.DataFrame()        
    """LOADING"""
    # load datsets
    data_raw = utils.load_dataset(prop)  
    keys_list = list(data_raw.keys())
    key_star = keys_list[0]; assert key_star != 'mpds'
    utils.print_info(data_raw, prop); print('')
    
    
    """PREPROCESSING"""
    # preprocessing
    data_clean = preprocess_dataset(data_raw, prop, merging,
                                    epsilon_T, 
                                    med_sigma_multiplier,
                                    mult_outliers,
                                    ascending_setting[prop]) 
    print(''); utils.print_info(data_clean, prop)
    plots.plot_super_histos(data_clean, 60, prop, op1=0.6, extraord=False)
    
    # add extraord column to all datasets(0/1)
    data_clean = add_column(data_clean, extraord_size, ascending_setting[prop])
    plots.plot_distinct_histos(data_clean, 60, prop, extraord=True)
    # plots.plot_super_histos(data_clean, 60, prop, op1=0.65, op2=0.8, extraord=True)








