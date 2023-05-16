#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:55:11 2023

@author: federico
"""

# IMPORTS
from CrabNet.kingcrab import CrabNet
from CrabNet.model import Model
from matminer.datasets import load_dataset
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd
from matbench.bench import MatbenchBenchmark

device = torch.device('cpu')

mb = MatbenchBenchmark(autoload=False)

for task in mb.tasks:
    if task.dataset_name == 'matbench_expt_gap':
        task.load()
        for fold in task.folds:
            train_inputs, train_outputs = task.get_train_and_val_data(fold)
            
            train_df = pd.DataFrame({'formula':train_inputs,
                                     'target' :train_outputs})
            
            val_df   = train_df.sample(frac=0.10, random_state=1234)
            train_df = train_df.drop(index=val_df.index)
            
            model = Model(CrabNet(compute_device=device).to(device),
                                  classification=False,
                                  random_state=1234,
                                  verbose=True,
                                  discard_n=10)
            
            model.load_data(train_df, train=True)
            model.load_data(val_df, train=False)
            
            model.fit(epochs=150)
            
            test_inputs = task.get_test_data(fold, include_target=True)
            test_df     = pd.DataFrame({'formula' : test_inputs[0],
                                        'target'  : test_inputs[1]})
            
            model.load_data(test_df, train=False)
            _, predictions, _ , _ = model.predict(model.data_loader)
            
            task.record(fold,predictions)

mb.to_file("crabnet_benchmarks.json.gz")




