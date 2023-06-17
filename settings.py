from sklearn.preprocessing import MinMaxScaler, RobustScaler

"""global params"""
n_repetitions = 1
# preprocessing
epsilon_T = 15              # controls the window size around ambient temperature
merging='median'            # 'median'/'best' (drop duplicates and save best value) 
med_sigma_multiplier = 0.5  # in 'median' merging, values with duplicates with std > 0.5*median are discarted
mult_outliers = 3           # values above mean + 3*sigma are discarted
# split
split = 'random' # 'top' # 'novelty'
shuffle_after_split = True
extraord_size = 0.2                               # best 20% will be extraord.
train_size, val_size, test_size = [0.7, 0.1, 0.2] # % train /val /test
k_val, k_test = [0.33, 0.33]                      # % top for val and test. 
# featurization
elem_prop = 'magpie'
# models
crabnet_kwargs = {'epochs':300, 'batch_size':128, 'verbose':False, 'discard_n':10}
roost_kwargs   = {'epochs': 300, 'batch_size':128}

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

ascending_setting = {'thermalcond': False,
                     'superconT'  : False,
                     'bulkmodulus' : False,
                     'bandgap'     : False,
                     'seebeck'     : False,
                     'rho'         : True,
                     'sigma'       : False,
                     'shearmodulus': False
                    }

pairs={'thermalcond'   : ['citrine', 'mpds'], 
       'superconT'     : ['japdata', 'mpds'],
        'bulkmodulus'  : ['aflow', 'mp'],   #'mp'
        'bandgap'      : ['zhuo', 'mpds'],    #'mp'
        'seebeck'      : ['te', 'mpds'],
        'rho'          : ['te', 'mpds'],
        'sigma'        : ['te', 'mpds'],
        'shearmodulus' : ['aflow', 'mp']   #'mp'
        }