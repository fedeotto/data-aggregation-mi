import pandas as pd
import numpy as np
from assets import utils
from chem import _element_composition

def merging(option, dfs_dict, final_df, ascending=False):
    if merging =='drop':
        final_df = final_df.drop_duplicates(subset=['formula'])
        
    elif merging =='median':
        final_df = final_df.groupby('formula', sort=False).median().reset_index()
        # fix 0/1 in extraord
        for column in [f'extraord|{i}' for i in dfs_dict.keys()]:
            final_df[column] = round(final_df[column]+0.1).astype(int)
    
    elif merging == 'best':
        if ascending == False:
            final_df = final_df.groupby('formula', sort=False).max().reset_index()
        elif ascending == True:
            final_df = final_df.groupby('formula', sort=False).min().reset_index()
        # fix 0/1 in extraord
        # no need
         
    elif merging is None:
        pass
    return final_df



def concat(dfs_dict: dict,
           merging_opt: str = 'median',
           ascending: bool = False,
           ):
    
    dfs = list(dfs_dict.values())
    
    final_df = pd.concat(dfs, axis=0, ignore_index=True)

    final_df = merging(merging_opt, dfs_dict, final_df, ascending)
    
    return final_df



def elem_concat(dfs_dict: dict,
                merging_opt: str = 'median',
                ascending: bool = False,
                verbose=False,
                k=5,
                n=5,
                random_state=1234
                ):
    
    dfs = list(dfs_dict.values())
    df_1 = dfs[0]
    df_2 = dfs[1]
    # count occurrences in first train
    occ_1 = utils.count_occurrences(df_1)
    # top k
    occ_1_sorted = occ_1.sort_values('occ', ascending=True)
    topk = occ_1_sorted.iloc[:k]
    # for each, grab n random formulas from df_2
    idx = {}
    elems_df_2 = df_2['formula'].apply(_element_composition)
    elems_df_2 = elems_df_2.sample(frac=1, random_state=random_state)
    for elem in topk.index:
        idx[elem] = []
        for i in elems_df_2.index:
            row = elems_df_2[i]
            if elem in row.keys():  idx[elem].append(i)
            if len(idx[elem])==n: 
                if verbose: print(f'stopped but there were more {elem}')
                break
    if verbose:
        print('added elem_concat:',[[elem,len(idx[elem])] for elem in idx])
    # all in one array
    all_idx = np.sort(np.concatenate(list(idx.values())))
    # select subsection of df_2 containing those indices
    df_to_concatenate = df_2.iloc[all_idx]
    
    final_df = pd.concat([df_1, df_to_concatenate], axis=0, ignore_index=True)

    final_df = merging(merging_opt, dfs_dict, final_df, ascending)
    
    return final_df    

# if __name__=='__main__':
    
#     data_raw = load_dataset('bulkmodulus')
#     data_clean = preprocess_dataset(data_raw,
#                                     property_name='bulkmodulus')
    
#     featurize(data_clean,
#               elem_prop='mat2vec'
#               )    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    