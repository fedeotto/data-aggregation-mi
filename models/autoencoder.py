#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:50:00 2023

@author: federico
"""

import torch
import torch.nn as nn
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import warnings
from torch.utils.data import Dataset, DataLoader
from cbfv.composition import generate_features
from chem import _fractional_composition_L
from chem_wasserstein.ElM2D_ import ElM2D
import pytorch_lightning as pl
import numpy as np

warnings.filterwarnings('ignore')

"""MODELS"""

class EmbeddingNetwork(pl.LightningModule):
    
    def __init__(self, 
                 n_datasets: int,
                 property_name:str = 'bulkmodulus',
                  # autoenc_params:dict,
                 *args, 
                 **kwargs):
        
        super().__init__()
        
        self.autoencoders = nn.ModuleList([Autoencoder() for n in range(n_datasets)])
        
        self.W = pd.read_csv(f'./distance_matrices/{property_name}_emd.csv', index_col=0)
        # W = 1/distance_matrix
        
    def forward(self, x):
        
        pass
        
        # latent_embs = (autoenc.encode(x) for autoenc in self.autoencoders)
        
        # return latent_embs
    
    def configure_optimizers(self):
        pass
    
    def training_step(self, batch, batch_idx):
        
        total_embs = []
        total_formulae = []
        
        for ae, data_batch in zip(self.autoencoders, batch):
            
            features, formulae = data_batch
            latent_emb = ae.encode(features)
            
            total_embs.append(latent_emb)
            total_formulae.append(formulae)
        
        total_embs = torch.cat(total_embs,dim=0)
        total_formulae = np.hstack(total_formulae)
        
        distant_matr = torch.cdist(total_embs, total_embs)
        
            
        
    

class Autoencoder(nn.Module):
    
    def __init__(self, 
                 input_dim: int = 200,
                 hidden_dims:list = [512,256,128],
                 latent_dim: int = 64,
                 batchnorm=True,
                 activation=nn.LeakyReLU
                 ):
        
        super().__init__()
        
        enc_dims = [input_dim] + hidden_dims
        dec_dims = list(reversed(enc_dims))
        
        
        self.encoder = nn.ModuleList(
            [nn.Linear(enc_dims[i], enc_dims[i + 1]) for i in range(len(enc_dims) - 1)]
        )
        
        self.decoder = nn.ModuleList(
            
            [nn.Linear(dec_dims[i], dec_dims[i + 1]) for i in \
             
             range(len(dec_dims) - 1)]
            )
        
        self.acts = nn.ModuleList([activation() for _ in range(len(enc_dims) - 1)])
        
        self.fc_latent = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_from_latent = nn.Linear(latent_dim, dec_dims[0])
        
        self.fc_out = nn.Linear(dec_dims[-1], input_dim)
                
    def encode(self, x):
        
        for fc, act in zip(self.encoder, self.acts):
            x = act(fc(x))
        
        return self.fc_latent(x)
        
    def decode(self, x):
        
        x = self.fc_from_latent(x)
        
        for fc, act in zip(self.decoder, self.acts):
            x = act(fc(x))
            
        return self.fc_out(x)
    
    def forward(self,x):
        
        enc = self.encoder(x)
        dec = self.decoder(x)
        
        return dec
    
    
"""DATA"""

class CompositionDataModule(pl.LightningDataModule):
    
    def __init__(self,
                 property_name: str ='dummy',
                 path:str = './',
                 return_target:bool = False,
                 batch_size: int = 256,
                 precomp_distances:bool = False,
                 **kwargs
                 ):
        
        super().__init__()
        
        self.batch_size = batch_size
        self.property_name = property_name
        self.precomp_distances = precomp_distances
        self.return_target = return_target
        filelist = [file for file in os.listdir(path) if file.startswith(f'{property_name}')]
        self.df_list = [pd.read_csv(file) for file in filelist]
        self.mapper = ElM2D()
        
    def prepare_data(self):
        
        print(f'\n--- Generating features for {len(self.df_list)} datasets. ---\n')
        self.comp_datasets = [SingleCompositionDataset(df,return_target=self.return_target) \
                              for df in self.df_list]
            
        # list of series of formulae for different datasets.
        total_formulae = np.hstack([dataset.formulae for dataset in self.comp_datasets])
        
        if not self.precomp_distances:
            
            print('\n--- Computing distance matrix for all formulas.. ---\n')
            W = self.mapper.EM2D(total_formulae)
        
            self.W = pd.DataFrame(W, columns=total_formulae, index=total_formulae)
            self.W.to_csv(f'./distance_matrices/{self.property_name}_emd.csv')
        
        else:
            
            path_emd = f'./distance_matrices/{self.property_name}_emd.csv'
            self.W = pd.read_csv(path_emd,index_col=0)
        
    def train_dataloader(self):
        
        loaders = [DataLoader(dataset, batch_size=self.batch_size) for dataset in self.comp_datasets]
        
        return loaders
    
    def val_dataloader(self):
        pass
    
    def test_dataloader():
        pass
    
    
        

class SingleCompositionDataset(Dataset):
    
    """TORCH DATASET OF CHEMICAL COMPOSITIONS (1D)"""
    
    def __init__(self, df, elem_prop='mat2vec', return_target=False):
        
        super().__init__()
        
        self.df = df
        self.scaler = MinMaxScaler()
        
        X, y, formulae, skipped = generate_features(df,
                                                    drop_duplicates=False,
                                                    elem_prop=elem_prop)
        
        self.X = X.values
        self.y = y.values
        self.formulae = formulae.values
        self.skipped = skipped
        self.return_target = return_target
        
        if elem_prop=='magpie':
            X = self.scaler.fit_transform(X)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        if not self.return_target:
            
            return (torch.tensor(self.X[idx],dtype=torch.float32),
                    self.formulae[idx])

        else:
            
            return (torch.tensor(self.X[idx],dtype=torch.float32),
                    self.formulae[idx],
                    torch.tensor(self.y[idx],dtype=torch.float32))

        
if __name__=='__main__':
    
    dataset_A = pd.read_csv('./dummy_A.csv')
    dataset_B = pd.read_csv('./dummy_B.csv')
    
    # comp_dataset_A = SingleCompositionDataset(dataset_A)
    # comp_dataset_B = SingleCompositionDataset(dataset_B)
    
    comp_dataset = CompositionDataModule(precomp_distances=False)
    
    comp_dataset.prepare_data()
    
    loaders = comp_dataset.train_dataloader()
    
    model = EmbeddingNetwork(n_datasets=2,
                             property_name='dummy'
                             )
    
    trainer = pl.Trainer(max_epochs=5)
    
    trainer.fit(model, train_dataloaders=loaders)
#     autoenc = Autoencoder()
        
#     dataset_A = torch.randn(400, 128)
#     dataset_B = torch.randn(600,128)

#     latent_A = autoenc.encode(dataset_A)
#     decoded_A = autoenc.decode(latent_A)
    
    
    # def fuse_training(datasets_list: list = [dataset_A,dataset_B]):
        
    #     encodings = []
    #     decodings = []
        
    #     for data in datasets_list:
            
    #         latent_data = model.encode(data)
    #         encodings.append(latent_data)
            
    #         decoded_data = model.decode(latent_data)
    #         decodings.append(decoded_data)
            
    # model = Autoencoder()
        
        
        
        
        
        
        
        






