#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 12:15:13 2023

@author: federico
"""

from chem_wasserstein.ElM2D_ import ElM2D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

def plot_chemical_space(property_name: str = 'bulk_modulus'):
    
    """
    Utility function to visualize the intersection of chemical space
    
    """
    
    mpds_data = pd.read_csv(f'./datasets_new/mpds_{property_name}.csv')
    aflow_data = pd.read_csv(f'./datasets_new/aflow_{property_name}.csv')
    
    formulas_mpds = set(mpds_data['formula'])
    formulas_aflow = set(aflow_data['formula'])
    
    n_shared = len(formulas_mpds & formulas_aflow)
    
    print(f'\n--- Number of formulas in MPDS: {len(formulas_mpds)} ---\n')
    print(f'\n--- Number of formulas in AFLOW: {len(formulas_aflow)} ---\n')
    print(f'\n--- Number of shared formulas: {n_shared} ---\n')
    
    mapper = ElM2D()
    
    print('\n--- Fitting formulas for MPDS dataset... ----\n')
    mpds_ump = mapper.fit_transform(mpds_data['formula'])
    
    print('\n--- Fitting formulas for AFLOW dataset... ----\n')
    aflow_ump = mapper.fit_transform(aflow_data['formula'])
    
    plt.rcParams['font.size'] = 14
    plt.rcParams['figure.dpi'] = 600
    
    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.scatter(mpds_ump[:,0], 
               mpds_ump[:,1],
               label='MPDS formulas'
               )
    
    ax.scatter(aflow_ump[:,0], 
               aflow_ump[:,1],
               label='AFLOW formulas'
               )
    
    ax.set_title(f'{property_name}')
    
    ax.legend()
    
if __name__ == '__main__':
    
    plot_chemical_space('shear_modulus')


    
    
    
    
    
    
    
    
    
    
    
    
    
    