"""global params"""
random_state = 1234
plot_augmentation = True      # plot the incremental accuracy at each agumentation iteration
initial_self_size = 0.1
# preprocessing
epsilon_T = 5               # controls the window size around ambient temperature
merging='best'              # 'median'/'best' (drop duplicates and save best value) 
med_sigma_multiplier = 0.5  # in 'median' merging values with duplicates with std > 0.5*median are discarted
mult_outliers = 3           # values above mean + 3*sigma are discarted
# split
extraord_size = 0.2                               # best 20% will be extraord.
train_size, val_size, test_size = [0.7, 0.1, 0.2] # % train /val /test
k_val, k_test = [0.33, 0.33]                      # % top for val and test. 
# featurization
elem_prop='mat2vec'
# models
crabnet_kwargs = {'epochs':100, 'discard_n':3,'verbose':False}

discover_kwargs = {'thresh' : 0.9, 
                   'n_iter':100000, 
                   'batch_size':1, 
                   'proxy_weight':1.0,
                   'pred_weight':1.0,
                   'clusters' : False}

ascending_setting = {
                    'bulkmodulus' : False,
                    'bandgap'     : False,
                    'seebeck'     : False,
                    'rho'         : True,
                    'sigma'       : False,
                    'shearmodulus': False
                    }

target_dataset = {
                    'bulkmodulus' : 'aflow',
                    'bandgap'     : 'zhuo',
                    'seebeck'     : 'te',
                    'rho'         : 'te',
                    'sigma'       : 'te',
                    'shearmodulus': 'aflow'
                    
                    }