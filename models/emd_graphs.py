# -*- coding: utf-8 -*-

"""
Spyder Editor

This is a temporary script file.

"""

from chem_wasserstein.ElM2D_ import ElM2D
import pandas as pd
import os
import joblib
from scipy.linalg import eig
from preprocessing import preprocess_dataset
from utils import load_dataset
from cbfv.composition import generate_features
from sklearn.manifold import LocallyLinearEmbedding
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def construct_graph_laplacian(dfs_dict: dict,
                              property_name: str,
                              cutoff: float = 1000,
                              precomputed = True,
                            ):
    
    if not precomputed:
        
        total_df = pd.concat(dfs_dict, axis=0, ignore_index=True)
        formulae = total_df['formula']
        
        mapper = ElM2D()
        emd = mapper.EM2D(formulae)
        emd = pd.DataFrame(emd, index=formulae, columns=formulae)
        # emd.to_csv(f'./distance_matrices/{property_name}_emd.csv')
    
    else:
        
        print('\n--- Loading precomputed distance matrix.. ---\n')
        emd = pd.read_csv(f'./distance_matrices/{property_name}_emd.csv',index_col=0)
    
    emd = np.exp(-emd)
    # emd = emd.values
        
    for i in range(len(emd)):
        
        neighs = list(emd.iloc[i].sort_values(ascending=False).index[:cutoff])
        emd.loc[emd.index[i], ~emd.columns.isin(neighs)] = 0
    
    W = emd.values.copy() #weighted adjacency matrix
    
    # compute degree matrix
    d_i = np.sum(W,axis=1)
    D = np.eye(len(emd))*d_i
    
    L = D - W
            
    return L


def compute_d_eigenvectors(L: np.array,
                           d: int = 200):
    
    eigenvals, eigenvecs = eig(L)
    
    idx = np.argsort(eigenvals)
    F = eigenvecs[:,idx[:d]]
    
    return F.real


def locally_linear_manifold(dfs_dict,
                            prop_name,
                            d=200,
                            n_neighbors=5,
                            fit=True
                            ):
    
    targets = [df['target'] for df in dfs_dict.values()]
    targets = pd.concat(targets,axis=0,ignore_index=True)
    extraord = [df['extraord'] for df in dfs_dict.values()]
    extraord = pd.concat(extraord,axis=0,ignore_index=True)
    
    features = []
    
    for df in dfs_dict.values():
        
        df = df.drop('extraord',axis=1)
        X, _, _, _ = generate_features(df,
                                       elem_prop='mat2vec'
                                       )
        features.append(X)
        
    features = pd.concat(features,axis=0,ignore_index=True)
    
    if fit:
        
        loc_linear = LocallyLinearEmbedding(n_neighbors=n_neighbors,
                                            n_components=d
                                            )
        F = loc_linear.fit_transform(features)
        joblib.dump(loc_linear, './loc_linear.pkl',compress=2)
    
    else:
        
        loc_linear = joblib.load('./loc_linear.pkl')
        F = loc_linear.transform(features)
        
    F = pd.DataFrame(F)
    
    final_df = pd.concat([F,targets,extraord],axis=1)
    
    return final_df


def manifold_alignment_fusion(dfs_dict,
                              prop_name,
                              d=200,
                              precomputed=True):
    
    targets = [df['target'] for df in dfs_dict.values()]
    targets = pd.concat(targets,axis=0,ignore_index=True)
    extraord = [df['extraord'] for df in dfs_dict.values()]
    extraord = pd.concat(extraord,axis=0,ignore_index=True)
    
    features=[]
    
    for df in dfs_dict.values():
        
        X, _, _, _ = generate_features(df,
                                       elem_prop='mat2vec'
                                       )
        features.append(X)
        
    features = np.vstack(features)
    
    L = construct_graph_laplacian(dfs_dict, 
                                  precomputed=precomputed,
                                  property_name=prop_name)
    
    F = compute_d_eigenvectors(L, d)
    
    F = F + features
    
    F = pd.DataFrame(F)
    F = pd.concat([F,targets,extraord],axis=1)
    
    return F
    
    
    
    
