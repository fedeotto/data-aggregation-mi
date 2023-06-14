import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

from plots import plot_super_histos, plot_distinct_histos, add_prop_to_violins, plot_violins
from utils import load_dataset, print_info
from preprocessing import preprocess_dataset, add_column
from settings import ascending_setting

props_list = [  'seebeck',
                'rho',
                'sigma',
                'bandgap',
                'bulkmodulus',
                'shearmodulus'                
              ]
pairs={
        'bulkmodulus'  : ['aflow', 'mp'],   #'mp'
        'bandgap'      : ['zhuo', 'mpds'],    #'mp'
        'seebeck'      : ['te', 'mpds'],
        'rho'          : ['te', 'mpds'],
        'sigma'        : ['te', 'mpds'],
        'shearmodulus' : ['aflow', 'mp']   #'mp'
        }

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

    

# violin plot
violin_fig = make_subplots(rows=1, cols=len(props_list), 
                           shared_xaxes=True, 
                           horizontal_spacing=0.07)
l=[]

# main loop
for i, prop in enumerate(props_list):
    
    freq_df_complete = pd.DataFrame()        
    """LOADING"""
    # load datsets
    data_raw = load_dataset(prop)  
    keys_list = list(data_raw.keys())
    key_A = pairs[prop][0]; assert key_A != 'mpds'
    key_B = pairs[prop][1]
    data_raw = {key_A:data_raw[key_A], key_B:data_raw[key_B]}
    print_info(data_raw, prop)
    print('')
    
    """PREPROCESSING"""
    # preprocessing
    data_clean = preprocess_dataset(data_raw, prop, merging,
                                    epsilon_T, 
                                    med_sigma_multiplier,
                                    mult_outliers,
                                    ascending_setting[prop]) 
    print('')
    print_info(data_clean, prop)
    # plot_super_histos(data_clean, 60, prop, op1=0.6, extraord=False)
    
    violin_fig = add_prop_to_violins(violin_fig, i+1, data_clean, prop, l)
    # add extraord column to all datasets(0/1)
    # data_clean = add_column(data_clean, extraord_size, ascending_setting[prop])
    # plot_distinct_histos(data_clean, 60, prop, extraord=True)
    # plots.plot_super_histos(data_clean, 60, prop, op1=0.65, op2=0.8, extraord=True)

plot_violins(violin_fig)






