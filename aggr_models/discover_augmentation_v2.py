from chem_wasserstein.ElM2D_ import ElM2D
import umap
from operator import attrgetter
from dl_models.CrabNet.kingcrab import CrabNet
from dl_models.CrabNet.model import Model
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import numpy as np
# from hdbscan.hdbscan_ import HDBSCAN
import assets.plots as plots
import torch
from tqdm import tqdm
import settings

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def my_mvn(mu_x, mu_y, r):
    """Calculate multivariate normal at (mu_x, mu_y) with constant radius, r."""
    return multivariate_normal([mu_x, mu_y], [[r, 0], [0, r]])


class DiscoAugment(object):
    def __init__(self, dfs_dict:dict, 
                 self_augment_frac = None,              # initial fraction for self_aumgent
                 random_state:int = 1234,               # seeds
                 ):
        
        dfs  = list(dfs_dict.values())
        keys = list(dfs_dict.keys())
        df_A, key_A = dfs[0], keys[0]
        
        # augmentation normally flows from B to A
        if self_augment_frac is None:
            df_B, key_B = dfs[1], keys[1]
            
        # if self augment, A is a small random fraction of A, 
        # while B is the remaining fraction of A        
        elif self_augment_frac > 0 and self_augment_frac < 1:
            key_B = key_A
            # randomly shuffle the dataframe
            shuffled = df_A.sample(frac=1, random_state=random_state).reset_index(drop=True)
            # calculate the number of rows in first block
            nA = int(len(shuffled) * self_augment_frac)  
            # split the dataframe into two blocks
            df_A  = shuffled[:nA].reset_index(drop=True)
            df_B = shuffled[nA:].reset_index(drop=True)
            
        else:
            raise ValueError('self_augment_frac is invalid')
            
        # print the sizes of the two blocks
        print("initial size of A:", len(df_A))
        print("initial size of B:", len(df_B))        
        
        # store things
        self.df_A = df_A
        self.df_B = df_B
        self.key_A = key_A
        self.key_B = key_B   
        self.nA = len(df_A)
        self.nB = len(df_B)
        self.random_state = random_state
        
        self.A_ilist = list(range(self.nA))  # to retrieve A points from embedding
        self.B_ilist = list(range(self.nB))  # to retrieve B points from embedding
        
    def predictive_model(self):
        
        if self.model_type=='crabnet':
            
            train_df = self.df_A.loc[:,:'target']
            test_df  = self.df_B.loc[:,:'target']
            
            # training on acceptor (train) then computing scores on donor (val)
            crabnet_model = Model(CrabNet(compute_device=device).to(device),
                                  classification=False,
                                  random_state=self.random_state,
                                  verbose=self.crabnet_kwargs['verbose'],
                                  discard_n=self.crabnet_kwargs['discard_n'])
            
            # little validation for crabnet.
            little_val = train_df.sample(frac=0.10, random_state=self.random_state)
            train_df = train_df.drop(index=little_val.index)
        
            # loading acceptor data
            crabnet_model.load_data(train_df, train=True)
            crabnet_model.load_data(little_val, train=False)
            crabnet_model.fit(epochs = crabnet_kwargs['epochs'])
            
            # predicting donor data
            crabnet_model.load_data(test_df, train=False)
            true, pred, _, _ = crabnet_model.predict(crabnet_model.data_loader)
            
            return pred, true
    
    
    def compute_umap_embs(self):
        
        formula_A = self.df_A["formula"]
        formula_B = self.df_B["formula"]
        
        all_formula = pd.concat((formula_A, formula_B), axis=0)
        
        mapper = ElM2D(verbose=False)
        mapper.fit(all_formula)
        dm = mapper.dm #distance matrix.
        
        umap_init = umap.UMAP(densmap=True,
                               output_dens=True,
                               dens_lambda=1.0,
                               n_neighbors=10,
                               min_dist=0,
                               n_components=2,
                               metric="precomputed",
                               random_state=self.random_state,
                               low_memory=False)
        umap_trans = umap_init.fit(dm)
        umap_emb, r_orig_log, r_emb_log = attrgetter("embedding_", 
                                                     "rad_orig_", 
                                                     "rad_emb_")(umap_trans)
        
        # plots.plot_umap(umap_emb, self.n_a)
        umap_r_orig = np.exp(r_orig_log)
        
        self.umap_emb = umap_emb
        self.umap_r_orig = umap_r_orig 
    

    def compute_density_scores(self, A_ilist_, B_ilist_, new=False):
        # B_ilist with respect to dataset B
        """given umap embeddings, calculate the density scores"""
        """uses current A_ilist and B_ilist"""
        A_ilist = A_ilist_.copy()
        B_ilist = B_ilist_.copy()
        if new:
            new_points = A_ilist[self.nA:]
            del A_ilist[self.nA:]
            new_points = list(np.array(new_points)+self.nA)
            A_ilist = A_ilist + new_points
        B_ilist = list(np.array(B_ilist)+self.nA)
        
        emb_A = self.umap_emb[A_ilist]
        emb_B = self.umap_emb[B_ilist]
        r_orig_A = self.umap_r_orig[A_ilist]
        r_orig_B = self.umap_r_orig[B_ilist] # umap emb has A and B concatenated
        
        #we calculate a list of mvns based on each pair of embeddings of our compounds
        mvn_list = list(map(my_mvn, emb_A[:, 0], emb_A[:, 1], r_orig_A))
        pdf_list = [mvn.pdf(emb_B) for mvn in mvn_list]
        dens_B = np.sum(pdf_list, axis=0)
        # log_d_dens = np.log(d_dens)

        density = dens_B.reshape(-1, 1)
        return density
    
    
    def compute_target_scores(self):
        """calculate the target scores"""
        # apply CrabNet to the whole dataset B, 
        pred, true = self.predictive_model()
        pred = pred.ravel().reshape(-1, 1)
        return pred
    
    
    def compute_final_score(self, target, density):
        """from density and target, calculate/update final score"""
        # determine output score
        if len(self.scores)==1:
            if 'target' in self.scores:
                final_score = target
            if 'density' in self.scores:   
                final_score = density
                
        elif len(self.scores)==2:
            if ('target' in self.scores and 'density' in self.scores):
                # combined scores
                target  = self.target_weight * target
                density = self.density_weight * density
                final_score = (target + density)/(self.density_weight+self.target_weight)
                if self.scaled: final_score = self.scaler().fit_transform(final_score)        
        return final_score
    
    
    def apply_augmentation(self,       
                           exit_mode: str = 'percentage',
                           thresh: float = 0.5,
                           percentage: float = 0.1,
                           batch_size: int = 5,
                           model_type :str ='crabnet',            # predictive model for target score
                           crabnet_kwargs: dict = {'epochs':40},  # keywargs for crabnet
                           scaled = True,
                           scaler = MinMaxScaler,                 # scaler in score computation
                           density_weight=1.0,
                           target_weight=1.0, 
                           scores:list = ['density', 'target']):
        
        self.target_weight = target_weight
        self.density_weight = density_weight
        self.scores = scores
        self.model_type = model_type
        self.crabnet_kwargs = crabnet_kwargs
        self.scaled = scaled
        self.scaler = scaler
        df_A_sub = self.df_A.copy()
        df_B_sub = self.df_B.copy()
        output_list = [df_A_sub]
        
        # for dataset B, we need the discover score for each row
        target_weigthed = np.zeros(len(self.df_B))
        target =None #dummy
        density=None
        if 'target' in scores:
            target = self.compute_target_scores()
            if scaled: target = self.scaler().fit_transform(target)
            
        # for dataset B, we need the distances in the UMAP embedding   
        density_weigthed = np.zeros(len(self.df_B))
        if 'density' in scores:
            self.compute_umap_embs()
            density = self.compute_density_scores(self.A_ilist, self.B_ilist, new=False)
            if scaled: density = self.scaler.fit_transform(-1*density)
        
        final_scores = self.compute_final_score(target, density)
        
        print("Size of A:", len(self.df_A), ' ➪ ', end='')
        
        if exit_mode == 'percentage':
            thr_size = int(percentage * self.nB)
            thresh = -np.inf
        elif exit_mode == 'thr':
            thr_size = np.inf
        
        while (len(self.A_ilist)-self.nA) < thr_size:
            # extract indices of top (batch_size) points which are above threshold
            df_B_sub['score'] = final_scores
            mask_thr  = (df_B_sub['score'] >= thresh)
            ordered  = df_B_sub[(mask_thr)].sort_values(by=['score'],ascending=False)
            # store indices
            idxs     = list(ordered.iloc[:batch_size].index)
            
            # exit loop if no more points above threshold
            if not idxs: break
        
            # remove from B
            df_B_sub = df_B_sub.drop(idxs, axis=0)
            # remove from B ilist
            self.B_ilist = [i for i in self.B_ilist if i not in idxs]
            # NOT USED
            # target = target[B_ilist_sub] # remove from target scores as well, because they will not be updated
            
            # add to A
            df_A_sub = pd.concat([df_A_sub, self.df_B.iloc[idxs]])
            output_list.append(df_A_sub.reset_index(drop=True))  # append iteration to output list (all augmentations)
            # add to A ilist
            self.A_ilist = self.A_ilist + idxs
            print(f'{len(df_A_sub)} ➪ ', end='')
            
            # exit loop if no more points to add
            if not self.B_ilist: break
                
            # update density scores
            density = self.compute_density_scores(self.A_ilist, self.B_ilist, new=True)
            if scaled: density = self.scaler.transform(-1*density)
            final_scores = self.compute_final_score(target, density)
        
        print('finished')
        return output_list