# Internal imports
from preprocessing import preprocess_dataset, add_column
import tasks
import plots
import utils
from settings import ascending_setting

# models
from models.baseline import concat
from models.discover_augmentation import DiscoAugment
from models.random_augmentation import RandomAugment

# warnings
import warnings
warnings.filterwarnings('ignore')


props_list = [ 
                # 'bulkmodulus',
                # 'bandgap',
                # 'seebeck',
                # 'rho',
                'sigma',
                # 'shearmodulus'                
              ]

splits_list = [
                'random',
                # 'top',
                # 'novelty'
                ]

tasks_list = [
                'linear_regression',
                # 'logistic_classification',      
                # 'crabnet_regression',
                # 'crabnet_classification'
                ]

models_list = [ 
                'baseline',
                # 'concat',
                # 'disco_augmentation'
                ]


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
# discover_params = {'n_epochs':100, 'score':'peak'} #do we need this?
crabnet_kwargs = {'epochs':100, 'verbose':False}

discover_kwargs = {'thresh' : 0., 
                   'n_iter':20000, 
                   'batch_size':50, 
                   'proxy_weight':1.0,
                   'pred_weight':1.0,
                   'clusters' : False,
                   'random_state':random_state}

random_kwargs = {'n_iter':20000, 
                'batch_size':50,
                'random_state':random_state}

# to host results and tabulate them
results = {prop:{} for prop in props_list}
header = ['Dataset name', 'Baseline', 'Concat']


# main loop
for prop in props_list:
            
    """LOADING"""
    # load datsets
    data_raw = utils.load_dataset(prop)  
    keys_list = list(data_raw.keys())
    utils.print_info(data_raw, prop); print('')
    
    # to host results
    results[prop] = utils.host_results(tasks_list, models_list, splits_list, keys_list)
    
    
    """PREPROCESSING"""
    # preprocessing
    data_clean = preprocess_dataset(data_raw, prop, merging,
                                    epsilon_T, 
                                    med_sigma_multiplier,
                                    mult_outliers,
                                    ascending_setting[prop]) 
    print(''); utils.print_info(data_clean, prop)
    # plots.plot_super_histos(data_clean, 60, prop, op1=0.6, extraord=False)
    
    # add extraord column to all datasets(0/1)
    data_clean = add_column(data_clean, extraord_size, ascending_setting[prop])
    # plots.plot_distinct_histos(data_clean, 60, prop, extraord=True)
    # plots.plot_super_histos(data_clean, 60, prop, op1=0.65, op2=0.8, extraord=True)
    
    
    """SPLIT"""
    for split in splits_list:
          train_dict, _, test_dict = tasks.apply_split(split_type = split,
                                                        df_collection = data_clean,
                                                        val_size=0, test_size=0.2, k_test=0.5,
                                                        random_state=random_state,
                                                        ascending=ascending_setting[prop])
          # featurize test
          test_feat = {}
          for test_key in test_dict.keys():
              test_feat[test_key]  = utils.featurize(test_dict[test_key],  elem_prop=elem_prop)
          
          """BASELINE"""  
          if 'baseline' in models_list:
              # does not merge datasets, every dataset is tested on its own test
              for test_key in test_dict.keys():
                  print(f'\n\t---Baseline model on {test_key}')
                  '''featurize'''
                  train = utils.featurize(train_dict[test_key], elem_prop=elem_prop)
                  test  = test_feat[test_key]
                  '''tasks'''
                  output = tasks.apply_all_tasks(train, test, test_key,
                                                 crabnet_kwargs, tasks_list,
                                                 random_state=random_state)
                  '''save results'''
                  results[prop][test_key][split]['baseline'] = {task: output[task][test_key] 
                                                                for task in tasks_list}
  
          """CONCAT MODEL"""
          if 'concat' in models_list:
              train = concat(dfs_dict=train_dict, merging=merging, 
                             elem_prop=elem_prop, ascending=ascending_setting[prop])
              train = utils.featurize(train, elem_prop=elem_prop)
              
              for test_key in test_dict.keys():
                  print(f'\n\t---Concat model on {test_key}')
                  test  = test_feat[test_key]
                  '''tasks'''
                  output = tasks.apply_all_tasks(train, test, test_key,
                                                 crabnet_kwargs, tasks_list,
                                                 random_state=random_state)
                  '''save results'''
                  results[prop][test_key][split]['concat'] = {task: output[task][test_key] 
                                                              for task in tasks_list}
                  
          """DISCOVER AUGMENTATION MODEL"""
          if 'disco_augmentation' in models_list:
              for test_key in test_dict.keys():
                  if test_key=='mpds':
                      '''save results'''
                      results[prop][test_key][split]['disco_augmentation'] = {task: {'nan':0}
                                                                              for task in tasks_list} 
                  elif test_key!='mpds':
                      print(f'\n\t---DiscoAugment model on {test_key}')
                      DAM = DiscoAugment(dfs_dict = train_dict,
                                         self_augment=initial_self_size,
                                         a_key = test_key,  # acceptor
                                         d_key = 'mpds')  # donor   
                      
                      train_list = DAM.apply_augmentation(model_type='crabnet',
                                                          crabnet_kwargs=crabnet_kwargs,
                                                          **discover_kwargs)
                      # ricordare gestire duplicati
                       # plots.plot_umap_augmentation(train_list, random_state=random_state)
                      
                      train = utils.featurize(train_list[-1], elem_prop=elem_prop)
                      test  = test_feat[test_key]
                      
                      '''Plotting augmentation'''
                      plots.periodic_table(train_list)
                      if not plot_augmentation: train_list = train_list[-1]
                      outs = []
                      for step in train_list:
                          step = train.iloc[step.reset_index(drop=True).index, :]
                          '''tasks'''
                          output = tasks.apply_all_tasks(step, test, test_key,
                                                         crabnet_kwargs, tasks_list,
                                                         random_state=random_state)
                          outs.append({task: output[task][test_key] 
                                       for task in tasks_list})
                      plot_augmentation = plots.plot_augmentation(test_key, 'linear_regression', prop)
                      plot_augmentation.load_disco(outs, train_list)   
                      '''save results'''
                      results[prop][test_key][split]['disco_augmentation'] = outs[-1]          
                  
                    
                      print(f'\n\t---RandomAugment model on {test_key}')
                      RAM = RandomAugment(dfs_dict = train_dict,
                                          self_augment=initial_self_size,
                                          a_key = test_key,  # acceptor
                                          d_key = 'mpds')  # donor   
                      
                      train_list = RAM.apply_augmentation(**random_kwargs)
                      
                      # ricordare gestire duplicati
                       # plots.plot_umap_augmentation(train_list, random_state=random_state)
                      
                      train = utils.featurize(train_list[-1], elem_prop=elem_prop)
                      test  = test_feat[test_key]
                      
                      '''Plotting augmentation'''
                      if not plot_augmentation: train_list = train_list[-1]
                      outs = []
                      for step in train_list:
                          step = train.iloc[step.reset_index(drop=True).index, :]
                          '''tasks'''
                          output = tasks.apply_all_tasks(step, test, test_key,
                                                         crabnet_kwargs, tasks_list,
                                                         random_state=random_state)
                          outs.append({task: output[task][test_key] 
                                       for task in tasks_list})
                      plot_augmentation.load_rnd(outs, train_list)
                      plot_augmentation.plot()
                      plot_augmentation.plot_equitability()
                        
            
print('\n')        
utils.create_tabulate(results,
                      props_list, models_list, tasks_list, splits_list,
                      metrics_list=['mae/acc'])       
    
    
    

    