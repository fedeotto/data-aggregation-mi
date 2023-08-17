import numpy as np
import pandas as pd
import assets.plots as plots

class RandomAugment(object):
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
    
    
    def apply_augmentation(self,       
                           exit_mode: str = 'percentage',    # percentage or iters
                           percentage: float = 0.1,
                           n_iters: int = 10,
                           batch_size: int = 5):
        
        df_A_sub = self.df_A.copy()
        df_B_sub = self.df_B.copy()
        output_list = [df_A_sub]
        
        print("Size of A:", len(self.df_A), ' ➪ ', end='')
        
        if exit_mode == 'percentage':
            thr_size = int(percentage * self.nB)
            thresh = -np.inf
        elif exit_mode == 'iters':
            thr_size = np.inf
        
        it=0
        while (len(self.A_ilist)-self.nA) < thr_size:
            # sample batch_size random indices
            try:
                idxs = list(df_B_sub.sample(n=batch_size, random_state=self.random_state).index)
            except:
                idxs = list(df_B_sub.index)
        
            # remove from B
            df_B_sub = df_B_sub.drop(idxs, axis=0)
            # remove from B ilist
            self.B_ilist = [i for i in self.B_ilist if i not in idxs]
            
            # add to A
            df_A_sub = pd.concat([df_A_sub, self.df_B.iloc[idxs]])
            output_list.append(df_A_sub.reset_index(drop=True))  # append iteration to output list (all augmentations)
            # add to A ilist
            self.A_ilist = self.A_ilist + idxs
            print(f'{len(df_A_sub)} ➪ ', end='')
            
            # exit loop if no more points to add
            if not self.B_ilist: break
            it+=1
            if exit_mode=='iters' and it==n_iters: break
        
        print('finished')
        return output_list
    
    
    
    
    