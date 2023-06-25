#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:22:49 2023

@author: federico
"""

from chem_wasserstein.ElM2D_ import ElM2D
import umap
from operator import attrgetter
from crabnet.crabnet_ import CrabNet
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import numpy as np
import warnings
from hdbscan.hdbscan_ import HDBSCAN
from mat_discover.mat_discover_ import Discover

warnings.filterwarnings('ignore')

def my_mvn(mu_x, mu_y, r):
        """Calculate multivariate normal at (mu_x, mu_y) with constant radius, r."""
        return multivariate_normal([mu_x, mu_y], [[r, 0], [0, r]])

def weighted_score(pred, proxy, pred_weight=1.0, proxy_weight=1.0):
        """Calculate weighted discovery score using the predicted target and proxy."""
        pred = pred.ravel().reshape(-1, 1)
        proxy = proxy.ravel().reshape(-1, 1)
        # Scale and weight the cluster data
        pred_scaler = RobustScaler().fit(pred)
        pred_scaled = pred_weight * pred_scaler.transform(pred)
        proxy_scaler = RobustScaler().fit(-1*proxy)
        proxy_scaled = proxy_weight * proxy_scaler.transform(-1*proxy)

        # combined cluster data
        comb_data = pred_scaled + proxy_scaled
        comb_scaler = RobustScaler().fit(comb_data)

        # cluster scores range between 0 and 1
        score = comb_scaler.transform(comb_data).ravel()
        
        return score


def run_discover(train_df:pd.DataFrame, 
                 val_df: pd.DataFrame,
                 score: str = 'peak',
                 precomputed_dm = None
                 ):
    
    crabnet_model = CrabNet(losscurve=False,
                            learningcurve=False,
                            random_state=42,
                            epochs=5)
    
    crabnet_model.fit(train_df, save=False)
    
    train_pred, train_sigma, train_true = crabnet_model.predict(
        train_df, return_uncertainty=True, return_true=True)

    val_pred, val_sigma, val_true = crabnet_model.predict(
        val_df, return_uncertainty=True, return_true=True)
    
    pred = np.concatenate((train_pred, val_pred), axis=0)
    
    train_formula = train_df["formula"]
    train_target = train_df["target"]
    val_formula = val_df["formula"]
    val_target = val_df["target"]
    
    all_formula = pd.concat((train_formula, val_formula), axis=0)
    all_target = pd.concat((train_target, val_target), axis=0)
    ntrain, nval = len(train_formula), len(val_formula)
    ntot = ntrain + nval
    train_ids, val_ids = np.arange(ntrain), np.arange(ntrain, ntot)
    
    # Distance calculations
    if precomputed_dm is not None:
        dm = pd.read_csv(precomputed_dm)
    else:
        mapper = ElM2D()
        mapper.fit(all_formula)
        dm = mapper.dm #distance matrix.
    
    umap_trans = umap.UMAP(
        densmap=True,
        output_dens=True,
        dens_lambda=1.0,
        n_neighbors=30,
        min_dist=0,
        n_components=2,
        metric="precomputed",
        random_state=42,
        low_memory=False,
    ).fit(dm)
    
    # Extracts densMAP embedding and radii
    # r_orig: array, shape (n_samples)
    #     Local radii of data points in the original data space (log-transformed).
    # r_emb: array, shape (n_samples)
    #     Local radii of data points in the embedding (log-transformed). 
    
    umap_emb, r_orig_log, r_emb_log = attrgetter("embedding_", "rad_orig_", "rad_emb_")(
        umap_trans
    )
    umap_r_orig = np.exp(r_orig_log)
        
    # Train contribution to validation density
    train_emb = umap_emb[:ntrain]
    train_r_orig = umap_r_orig[:ntrain]
    val_emb = umap_emb[ntrain:]
    val_r_orig = umap_r_orig[ntrain:]

    train_df["emb"] = list(map(tuple, train_emb))
    train_df["r_orig"] = train_r_orig
    val_df["emb"] = list(map(tuple, val_emb))
    val_df["r_orig"] = val_r_orig

    
    #we calculate a list of mvns based on each pair of embeddings of our compounds
    mvn_list = list(map(my_mvn, train_emb[:, 0], train_emb[:, 1], train_r_orig))
    pdf_list = [mvn.pdf(val_emb) for mvn in mvn_list]
    val_dens = np.sum(pdf_list, axis=0)
    val_log_dens = np.log(val_dens)
    val_df["dens"] = val_dens #first possible proxy : validation density
    
    #Nearest neighbors calculations
    r_strength = 1.5
    mean, std = (np.mean(dm), np.std(dm))
    radius = mean - r_strength * std
    n_neighbors = 10
    NN = NearestNeighbors(radius=radius, n_neighbors=n_neighbors, metric="precomputed")
    NN.fit(dm)

    neigh_ind = NN.kneighbors(return_distance=False)
    num_neigh = n_neighbors * np.ones(neigh_ind.shape[0])

    neigh_target = np.array([pred[ind] for ind in neigh_ind], dtype="object")
    k_neigh_avg_targ = np.array([np.mean(t) if len(t) > 0 else float(0) for t in neigh_target])
    val_k_neigh_avg = k_neigh_avg_targ[val_ids] # second proxy: nearest nbrs.
    
    peak_score = weighted_score(val_pred, val_k_neigh_avg)
    dens_score = weighted_score(val_pred, val_dens)
    
    return dens_score, peak_score
    
    
if __name__=='__main__':
    
    # running Discover bare bones
    train_df = pd.read_csv('dummy_A.csv')
    val_df = pd.read_csv('dummy_B.csv')
    # train_df = train_df.iloc[:100]
    # val_df = val_df.iloc[:100]
    # dens_bone, peak_bone = run_discover(train_df, val_df)
    
    # running Discover full
    disc = Discover(crabnet_kwargs={'epochs': 5, 'random_state':42})
    disc.fit(train_df)
    
    dens_full, peak_full = disc.predict(val_df, return_peak=True)
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    