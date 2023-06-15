import pandas as pd
import numpy as np
from chem import _fractional_composition_L, _element_composition_L

def clean_outliers(df_, mult=5):
    df = df_.copy()
    std = np.std(df['target'])
    mean = np.mean(df['target'])
    output = df[df['target']<=(mean+mult*std)]          # drop above mean + x*std
    output = output[output['target']>=(mean-mult*std)]  # drop below mean - x*std

    return output, len(df)-len(output)


def clean_unstable_elements(df_):
    all_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                   'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
                   'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                   'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                   'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                   'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                   'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                   'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                   'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                   'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                   'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                   'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    
    df = df_.copy()
    
    bad_idxs = []
    for i, f in enumerate(df['formula']):
        elems, _ = _element_composition_L(f)
        
        for el in elems:
            if el in all_symbols[93:]:
                bad_idxs.append(i)
    
    df = df.drop(index=bad_idxs)
    df = df.reset_index(drop=True)
    
    return df
    

def preprocess_dataset(data_raw_: dict,
                       property_name: str,
                       merging: str='median', 
                       epsilon_T: int = 10,
                       med_sigma_multiplier: float=0.5,
                       mult_outliers: int=5,
                       ascending:bool =False,
                       ):
    data_clean = {}
    
    for k,v in data_raw_.copy().items():
        
        print(f'{k} preprocessing:', end='')
        
        # selecting ambient temperature
        if 'Temperature' in v:
            l_bef = len(v)
            mask = (v['Temperature']<=298 + epsilon_T) & (v['Temperature']>=298-epsilon_T)
            v = v[mask]
            l_af = len(v)
            v = v.drop('Temperature',axis=1)
            print(f' ambient T ({l_bef-l_af}); ', end='')
            
        # logarithm of large range properties    
        if property_name in ['sigma','rho', 'thermalcond']:
            v['target'] = np.log10(v['target'])
            print(' log10; ', end='')
        
        # median of duplicates, also from similar temperature   
        if merging == 'median':
            l = len(v)
            # drop values for which the median has been obtained 
            # from extremely different values
            std = v.groupby('formula',sort=False).std().values   # compute std
            v = v.groupby('formula',sort=False).median()  # groupby median
            l_med = len(v)
            v['std'] = std    # add std column in grouped dataset
            v = v.reset_index(drop=False)
            # save indices of values with std > 0.5*median (high deviation)
            i_drop = v[v['std']>med_sigma_multiplier*v['target']].index
            # drop rows and reset indices
            v = v.drop(index=i_drop, axis=0).reset_index(drop=True)    
            v = v.drop('std', axis=1)
            print(f' median of duplicates ({l-l_med} + {len(i_drop)} drop); ',end='')
        
        # if duplicates, take the best 
        if merging == 'best':
            l = len(v)
            # take best value, the lowest or the highest depending on the property
            if ascending == False:
                v = v.groupby('formula', sort=False).max().reset_index()
            elif ascending == True:
                v = v.groupby('formula', sort=False).min().reset_index()
            l_after = len(v)
            print(f' best of duplicates ({l-l_after}); ',end='')            
        
        
        # clean outliers  
        v, n = clean_outliers(v, mult_outliers)
        print(f' outliers drop ({n})')    
        
        v = clean_noble_gases(v)
        v = clean_unstable_elements(v)
        
        # # -------- serve? --------------    
        # valid_elems = list(pd.read_csv(f"./cbfv/element_properties/{elem_prop}.csv",index_col=0).index)
        # skipped=0
        # for i,f in enumerate(v['formula']):
        
        #     elems, _ = _fractional_composition_L(f)
            
        #     for el in elems:
                
        #         if el not in valid_elems:
                    
        #             v = v.drop(index=i)
        #             skipped+=1
                    
        #             break
    
        # v = v.reset_index(drop=True)
        
        # if skipped!=0: print(f' n invalid = {skipped}')
        # # ---------------------------    
        
        data_clean[k]=v.reset_index(drop=True)
        
    return data_clean


def clean_noble_gases(df):
    
    noble_gases = ['Kr','Ne','Xe','He','Ar','Og','Rn' ]
    # Filtering noble gases
    to_be_removed = []
    for i, row in df.iterrows():
        elems, _ = _element_composition_L(row['formula'])
        if True in [el in noble_gases for el in elems]:
            # print(df.iloc[i,0])
            to_be_removed.append(i)
            
    df = df.drop(index=to_be_removed)
    df = df.reset_index(drop=True)
    
    return df

def clean_TE_dataset(df):
    
    df_copy = df.copy()
    
    df_copy['formula'] = df_copy['formula'].str.split('+').str[0]
    df_copy['formula'] = df_copy['formula'].str.split('(').str[0]
    df_copy['formula'] = df_copy['formula'].str.split('///').str[0]
    df_copy['formula'] = df_copy['formula'].str.split('///').str[0]
    df_copy['formula'] = df_copy['formula'].str.split(' ').str[0]
    df_copy = df_copy[~df_copy['formula'].str.contains('x')]
    mask = df_copy['formula'] == ''
    df_copy = df_copy[~mask]
    
    return df_copy
          


def clean_MPDS_dataset(df):
    
    
    """ 
    clean the MPDS dataset
    
    Parameters:
        df (pandas dataset): input dataset
    
    Returns:
        df (pandas dataset): output dataset
    """
    
    df_copy = df.copy()
    df_copy['formula'] = df_copy['formula'].str.split(' ',n=1).str[0]
    df_copy['formula'] = df_copy['formula'].str.split(',',n=1).str[0]
    df_copy['formula'] = df_copy['formula'].map(lambda x: x.lstrip('[').rstrip(']'))
    df_copy['formula'] = df_copy['formula'].str.replace('[','',regex=False)
    df_copy['formula'] = df_copy['formula'].str.replace(']','',regex=False)
    df_copy['formula'] = df_copy['formula'].str.replace('rt','',regex=False)
    df_copy = df_copy[~df_copy['formula'].str.contains('x')]
    df_copy = df_copy.reset_index(drop=True)
    
    #Cleaning from D
    for i, formula in enumerate(df_copy['formula']):
        
        if 'D' in formula:
            
            try:
                if formula[formula.index('D') + 1] !='y':
                    
                    df_copy = df_copy.drop(index=i)
                    
            except:
                    
                df_copy = df_copy.drop(index=i)
    
    #Cleaning from G
    for i, formula in enumerate(df_copy['formula']):
        
        if 'G' in formula:
            
            try:
                if formula[formula.index('G') + 1] not in ['a','e','d']:
                    df_copy = df_copy.drop(index=i) 
            except:
                df_copy = df_copy.drop(index=i)
                
    df_copy = df_copy.reset_index(drop=True)
    
    return df_copy


# def grouping(df):
    
#     """ 
#     group different measures with different temperatures, and save the 
#     maximum deviation
    
#     Parameters:
#         df (pandas dataset): input dataset
    
#     Returns:
#         df (pandas dataset): output dataset
        
#     """
#     #Calculating max_deviation
#     values = df.groupby(['formula'], sort=False)
#     prov = (values.max() - values.min()).values / 2
#     df = df.groupby(['formula'],sort=False).median().reset_index()
#     df['max_dev'] = prov
    
#     return df

        
def add_column(df_collection_, size, ascending=False):
    df_collection = df_collection_.copy()
    out={}
    
    # get the thresholds in correspondence of size% of the property range
    thr = {}
    for key, df in df_collection.items():
        N = len(df)
        df_sorted = df.sort_values('target', ascending=ascending).reset_index(drop=True)
        thr[key]  = df_sorted['target'][int(N*size)]
    
    # add extraord columns w.r.t. all datasets thresholds
    for key_o in df_collection.keys():
        for key_i in df_collection.keys():
            df_collection[key_o][f'extraord|{key_i}'] = 0
            # change to 1 values above the calculated threshold
            if  ascending==False:
                mask = df_collection[key_o]['target']>thr[key_i]   
            elif ascending==True: 
                mask = df_collection[key_o]['target']<thr[key_i]
            df_collection[key_o].loc[mask,f'extraord|{key_i}'] = 1
            
    return df_collection
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        