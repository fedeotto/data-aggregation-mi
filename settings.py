from sklearn.preprocessing import MinMaxScaler, RobustScaler

"""global params"""
n_repetitions = 5
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
k_val, k_test = [0., 2./3.]                      # % top for val and test. 
# featurization
elem_prop = 'magpie'

# models
models_list = [ 
                'baseline',
                'concat',
                'elem_concat',
                'disco'
                ]

#elemconcat
k_elemconcat = 5
n_elemconcat = 10

#crabnet
crabnet_kwargs = {'epochs':300, 'batch_size':32, 'verbose':False, 'discard_n':10}

#roost
roost_kwargs   = {'epochs': 300, 'batch_size':32}

#discover
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
#random
rnd_kwargs      = {'exit_mode': 'percentage',    # percentage or iters
                   'batch_size': 5,
                   'n_iters': 10,     # used if 'exit_mode' = 'iters'
                   'percentage': 1.,  # used if 'exit_mode' = 'percentage'
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

#can change `mp` with `mpds` if you have access to mpds data
pairs={'thermalcond'   : ['citrine', 'mpds'], 
       'superconT'     : ['japdata', 'mpds'],
        'bulkmodulus'  : ['aflow', 'mpds'], #mpds or mp 
        'bandgap'      : ['zhuo', 'mpds'], #mpds or mp
        'seebeck'      : ['te', 'mpds'],
        'rho'          : ['te', 'mpds'],
        'sigma'        : ['te', 'mpds'],
        'shearmodulus' : ['aflow', 'mp']  #mpds or mp
        }