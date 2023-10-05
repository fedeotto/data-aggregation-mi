import numpy as np
import pandas as pd
import sklearn
import os
from tabulate import tabulate
from cbfv.composition import generate_features
from assets.chem import _element_composition

from assets.preprocessing import clean_MPDS_dataset, clean_TE_dataset
import assets.tasks as tasks


def load_dataset(property_name: str = 'bulkmodulus'):
    print('\n-------------------------------------------------------------------------------------------')
    print('loading datasets...')
    data_raw = {}
    
    filelist = [file for file in os.listdir(f'./datasets/') if file.startswith(f'{property_name}')]
    # put mpds as last
    mpds_file = [f for f in filelist if "mpds" in f]
    others    = [f for f in filelist if "mpds" not in f]
    filelist = others + mpds_file
    
    for file in filelist:
        
        file = file.split('.')[0]
        df = pd.read_csv(f'./datasets/{file}.csv')
        df = df.rename(columns={df.columns[-1] : 'target'})

        if file=='bulkmodulus_mpds':
            df = clean_MPDS_dataset(df)
            df = df[df['formula']!='Ca3Al2SiO43']
        
        if file=='bulkmodulus_aflow':
            pass
        
        if file=='bandgap_mpds':
            sup_thr = 20
            inf_thr = 0
            df = clean_MPDS_dataset(df)
            df = df[(df['target']<sup_thr)]
            df = df[(df['target']>inf_thr)]
            
        if file=='bandgap_mp':
            continue
            
        if file=='bandgap_zhuo':
            sup_thr = 20
            inf_thr = 0
            df = df[(df['target']<sup_thr)]
            df = df[(df['target']>inf_thr)]
        
        if file=='seebeck_mpds':
            thr = 3000
            df = clean_MPDS_dataset(df)
            df = df[(df['target']<=thr) & (df['target']>=-thr)].reset_index(drop=True)
        
        if file=='seebeck_te':
            df = clean_TE_dataset(df)
            df = df[df['formula']!='Bi0.46Sb1.54Te3']
            
        if file=='rho_mpds':
            df = clean_MPDS_dataset(df)
            df = df[df['target']>0]
            df = df[df['target']!=1]
        
        if file=='rho_te':
            df = clean_TE_dataset(df)
            df = df[df['formula']!='Bi2O2Se']
            df['target'] = df['target'].astype('float')
            df = df[df['target']!=1]

        if file=='sigma_mpds':
            df = clean_MPDS_dataset(df)
            df = df[df['target']!=1]
            
        if file=='sigma_te':
            df = clean_TE_dataset(df)
            df = df[df['formula']!='Bi2O2Se']
            df['target'] = df['target'].astype('float')
            df = df[df['target']!=1]
        
        if file=='shearmodulus_mpds':
            df = clean_MPDS_dataset(df)
        
        if file=='shearmodulus_aflow':
            pass
            
        if file=='thermalcond_mpds':
            df = clean_MPDS_dataset(df)
        
        if file=='thermalcond_citrine':
            pass

        if file=='superconT_mpds':
            df = clean_MPDS_dataset(df)
        
        if file=='superconT_japdata':
            pass
                   
        key = file.split('_')[1]
        
        data_raw[key] = df
    return data_raw



def featurize(df, elem_prop='mat2vec'):
        
    extraords = df.iloc[:, [-1,-2]].reset_index(drop=True)    
    
    X,y, formulae,_ = generate_features(df, 
                                        elem_prop=elem_prop,
                                        extend_features=False
                                        )  
    scaler = sklearn.preprocessing.StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    
    feat_df = pd.concat([formulae,X_scaled,y,extraords],axis=1)
    feat_df = feat_df.dropna()
        
    return feat_df


def print_info(df_dict, property_name):
    printout = [['Dataset name', 'N formulas', 'min target', 'max target']]
    for key, df in df_dict.items():
        printout.append([property_name + ' ' + key, len(df), df['target'].min(), df['target'].max()])
    print(tabulate(printout, headers="firstrow"))
    return



def create_tabulate(results,
                    property_names, models_list, tasks_list, split_list,
                    metrics_list=['mae/acc']):
    
    for metric in metrics_list:
        
        if metric=='mae/acc':
            header    = ['dataset name', 'model'] + ['' for i in range(len(tasks_list) * len(split_list) - 1)]
            table = []
            
            first_row = ['']
            for split in split_list:
                first_row.append(split)
                first_row = first_row+['' for i in range(len(tasks_list)-1)]
            table.append(first_row)
            
            second_row = ['']
            for split in split_list:
                second_row = second_row+[task for task in tasks_list]
            table.append(second_row)
            
            for prop in property_names:
                for key in list(results[prop].keys()):
                    row = [f'{prop} {key}']
                    for split in split_list:
                        for task in tasks_list:
                            row.append([ round(list(results[prop][key][split][model][task].values())[0],3) 
                                         for model in models_list ])
                    table.append(row)
                                      
            content = tabulate(table, 
                               headers=header)
            print(content);
            text_file=open("output_mae_acc.xls","w")
            text_file.write(content)
            text_file.close()
    

        if metric=='mse':
            tasks_list = filter(lambda x: x not in ['logistic_classification','crabnet_classification'], tasks_list)
            
            header    = ['dataset name', 'model'] + ['' for i in range(len(tasks_list) * len(split_list) - 1)]
            table = []
            
            first_row = ['']
            for split in split_list:
                first_row.append(split)
                first_row = first_row+['' for i in range(len(tasks_list)-1)]
            table.append(first_row)
            
            second_row = ['']
            for split in split_list:
                second_row = second_row+[task for task in tasks_list]
            table.append(second_row)
            
            for prop in property_names:
                for key in list(results[prop].keys()):
                    row = [f'{prop} {key}']
                    for split in split_list:
                        for task in tasks_list:
                            row = row + [ round(list(results[prop][key][split][model][task].values())[1],3) 
                                         for model in models_list ]
                    table.append(row)
                                      
            content = tabulate(table, 
                               headers=header)
            print(content)
            text_file=open("output_mse.xls","w")
            text_file.write(content)
            text_file.close()


def self_augmentation(train, val, test, which, ascending):
    
    key = list(train.keys())[which]
    trainA, trainB, _ = tasks.rnd_split({key:train[key]}, 
                                        0.5, 0.0001,
                                        ascending=ascending)
    valA, valB, _ = tasks.rnd_split({key:val[key]}, 
                                    0.5, 0.0001,
                                    ascending=ascending)
    
    train_self = {'1sthalf': trainA[key], '2ndhalf': trainB[key]} 
    val_self = {'1sthalf': valA[key], '2ndhalf': valB[key]}  
    test_self = {'1sthalf': test[key]}
    return train_self, val_self, test_self


def host_results(tasks_list, models_list, splits_list, keys_list):
    return {key:{split:{model:{task:{}
                               for task in tasks_list} 
                        for model in models_list} 
                 for split in splits_list} 
            for key in keys_list}


def count_occurrences_traintest(train, test):
    train_elems_df = train.apply(_element_composition)
    test_elems_df  = test.apply(_element_composition)
    
    # list elems
    train_elems = [elem for i in train_elems_df.index for elem in train_elems_df[i].keys()]
    test_elems = [elem for i in test_elems_df.index for elem in test_elems_df[i].keys()]
    
    # unique elems in test and their occurrences in test
    test_elems_unique = list(set(test_elems.copy())); test_elems_unique.sort()
    occ_test = [[elem, test_elems.count(elem)] for elem in test_elems_unique]
    df_out = pd.DataFrame(data=np.array(occ_test)[:,1].astype(int), 
                          index=np.array(occ_test)[:,0].astype(str), 
                          columns=['occ_test'])
    
    # occurrences in train
    occ_train = [[elem, train_elems.count(elem)] for elem in df_out.index]
    df_out['occ_train'] = np.array(occ_train)[:,1].astype(int)
    
    #index name
    df_out.index.name = 'elem_test'
    
    return df_out
    

def count_occurrences(dataset):
    # dataset must be 'formula', 'target'
    elems_df = dataset['formula'].apply(_element_composition)
    
    # list elems
    elems = [elem for i in elems_df.index for elem in elems_df[i].keys()]
    
    # unique elems in test and their occurrences in test
    elems_unique = list(set(elems.copy())); elems_unique.sort()
    occ = [[elem, elems.count(elem)] for elem in elems_unique]
    df_out = pd.DataFrame(data=np.array(occ)[:,1].astype(int), 
                          index=np.array(occ)[:,0].astype(str), 
                          columns=['occ'])
    
    # #index name
    # df_out.index.name = 'elem'
    
    return df_out
    
    
    
    
def add_accuracies_to_count(occ, true, pred, task, metrics):
    assert (true['formula']==pred['formula']).all()
    block = pd.DataFrame(data=np.nan, index=occ.index, columns=[f'{task}_{metric}' for metric in metrics])
    for elem in occ.index:
        # indices with elem in formula
        idx = [i for i, f in true.iterrows() if elem in _element_composition(f['formula'])]
        # subset of pred and true with elem in it
        true_sub = true.iloc[idx]['target']
        pred_sub = pred.iloc[idx]['target']
        # average error for elem
        for metric in metrics:
            col_name = f'{task}_{metric}'
            block.loc[elem,col_name] = tasks.score_evaluation(true_sub, pred_sub, metric)
    return pd.concat([occ, block], axis=1)
    
    
    
    
     
