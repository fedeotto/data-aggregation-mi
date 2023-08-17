"""this script reproduce Fig. 2 of the paper:
Not as simple as we thought: a rigorous examination of data aggregation in materials informatics
09 August 2023, Version 1
Federico Ottomano, Giovanni De Felice, Vladimir Gusev, Taylor Sparks """

import numpy as np
import pandas as pd
import pickle
from assets import plots
from assets import utils
from assets import tasks
from assets.preprocessing import preprocess_dataset, add_column

import warnings
warnings.filterwarnings('ignore')

# settings are imported from settings.py
from settings import *

props_list = [ 'rho' ]      # 'thermalcond',
                            # 'superconT',
                            # 'seebeck',
                            # 'rho'
                            # 'sigma',
                            # 'bandgap',
                            # 'bulkmodulus',
                            # 'shearmodulus'
task = 'random_forest_regression'   # task to perform and plot: ('random_forest_regression' / 'crabnet_regression')
metric = 'mae' # metric to plot

to_save = {}

def plot_all():
    # main loop
    for prop in props_list:
        # init for storing results for each property
        freq_df_complete = pd.DataFrame()  
        outputs={key:[] for key in ['mae','mse','mape','r2','mre']}

        """LOADING"""
        # load datset A
        data_raw = utils.load_dataset(prop)  
        key_A = pairs[prop][0]; assert key_A != 'mpds'
        
        """PREPROCESSING"""
        data_clean = preprocess_dataset(data_raw, 
                                        prop, 
                                        merging,
                                        epsilon_T, 
                                        med_sigma_multiplier,
                                        mult_outliers,
                                        ascending_setting[prop])    ;   print('')
        utils.print_info(data_clean, prop)
        # add 0/1 column indicating extraordinarity
        data_clean = add_column(data_clean, extraord_size, ascending_setting[prop])
        
        """LOOP"""
        # loop over different random seeds
        print('\niters:')
        for n in range(n_repetitions):
            print(f'{n+1}-',end='')
            # init seed
            random_state = n
            # split into train and test according to seed
            train, _, test = tasks.apply_split(split_type = split,
                                               df = data_clean[key_A],
                                               test_size=test_size,
                                               random_state=random_state,
                                               verbose=False,
                                               shuffle=shuffle_after_split)
            # featurize with 'elem_prop', default = magpie
            train = utils.featurize(train, elem_prop=elem_prop)
            test  = utils.featurize(test, elem_prop=elem_prop)

            # perform tasks
            out, freq_df = tasks.apply_all_tasks(train, test, 
                                                key_A, [task], 
                                                crabnet_kwargs,
                                                random_state=random_state,
                                                verbose=False)
            freq_df_complete = pd.concat([freq_df_complete, freq_df], axis=0)
            
            for score in outputs.keys():
                outputs[score].append(out[task][score])

        to_save[prop] = freq_df_complete   ;   print('\n')
        
        # compute statistics to plot
        means = freq_df_complete.groupby('elem_test', sort=True).mean().loc[:,['occ_train', f'{task}_{metric}']]
        stds  = freq_df_complete.groupby('elem_test', sort=True).std().loc[:,['occ_train', f'{task}_{metric}']]
        stds.columns = [f'{col}_std' for col in stds.columns]
        of_interest = pd.concat([means,stds], axis=1)

        # plot
        plots.plot_elem_class_score_matplotlib(of_interest, task, metric, prop, web=True)   ;    print('\n')
        for score in outputs.keys():
            print(f'AVERAGE {score} = {round(np.mean(outputs[score]),3)} ', end='')
            print(f'+- {round(np.std(outputs[score]),3)}')
    


    # save results for eventual later use
    if task=='random_forest_regression':  met = 'rf'
    elif task=='crabnet_regression':  met = 'crab'
    with open(f'results/results_1_{met}.pkl', 'wb') as f:
        pickle.dump(to_save, f)
    
    

if __name__ == '__main__': 
    plot_all()






