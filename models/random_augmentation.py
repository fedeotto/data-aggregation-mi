import numpy as np
import pandas as pd
import plots

class RandomAugment(object):
    def __init__(self, 
                 dfs_dict : dict,
                 a_key: str,         # acceptor
                 d_key: str,         # donor
                 self_augment = 0.1,
                 score: str = 'dens',
                 ):
        
        self.a_key = a_key
        self.d_key = d_key
        
        self.a_df = dfs_dict[a_key]
        self.d_df = dfs_dict[d_key]
        
        if self_augment is not None:
            #we split acceptor in half and auto-augment it.
            initial_size = int(len(self.a_df)*self_augment)
            self.a_df = dfs_dict[a_key].iloc[:initial_size].reset_index(drop=True)
            self.d_df = dfs_dict[a_key].iloc[initial_size:].reset_index(drop=True)
        
        self.dfs_dict = dfs_dict
        self.score = score
        
        self.n_a = len(self.a_df)
        self.a_ilist = list(self.a_df.index)
        self.d_ilist = list(self.d_df.index)
    
    
    def new_dataframe(self):
        combo = pd.concat([self.a_df, self.d_df], axis=0).reset_index(drop=True)
        combo = combo.iloc[self.a_ilist]
        # print(len(self.a_ilist))
        
        return combo
    
    
    def apply_augmentation(self,
                           n_iter: int = 15,
                           batch_size: int = 10,
                           random_state:int = 1234
                           ):
        
        output = []
        output.append(self.new_dataframe())
        proxy = self.d_df
        
        for n in range(n_iter):
            try:
                idxs       = list(proxy.sample(n=batch_size, 
                                               random_state=random_state).index)
            except:
                idxs = list(proxy.index)
            idxs_combo = [i+self.n_a for i in idxs]
            if idxs: 
                # augment destination indices 
                self.a_ilist = self.a_ilist + idxs_combo
                proxy = proxy.drop(index=idxs, axis=0)
                
            if len(proxy) == 0:
                output.append(self.new_dataframe())  
                
                return output
            
            output.append(self.new_dataframe()) 
                
        return output        
                