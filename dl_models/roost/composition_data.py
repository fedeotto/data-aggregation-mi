import functools
import os
import numpy as np
import pandas as pd
import torch
from pymatgen.core.composition import Composition
from torch.utils.data import Dataset, DataLoader
from roost_refactored.core import Featurizer

class CompositionData(Dataset):
    
    """
    The CompositionData dataset is a wrapper for a dataset data points are
    automatically constructed from composition strings.
    """

    def __init__(
            
        self,
        df: pd.DataFrame,
        fea_path : str = './roost_refactored/data/el-embeddings/matscholar-embedding.json',
        formula_column=["formula"]
        
    ):
        """[summary]

        Args:
            data_path (str): [description]
            fea_path (str): [description]
            task_dict ({name: task}): list of tasks
            inputs (list, optional): column name for compositions.
                Defaults to ["composition"].
            identifiers (list, optional): column names for unique identifier
                and pretty name. Defaults to ["id", "composition"].
        """
        
        df = df.rename(columns={df.columns[1] : 'target'})
        df['identifier'] = list(range(len(df)))
                
        self.df = df
        self.identifier = 'identifier'

        self.elem_features = Featurizer.from_json(fea_path)
        self.elem_emb_len = self.elem_features.embedding_size
        

    def __len__(self):
        return len(self.df)

    @functools.lru_cache(maxsize=None)  # Cache data for faster training
    def __getitem__(self, idx):
        """[summary]

        Args:
            idx (int): dataset index

        Raises:
            AssertionError: [description]
            ValueError: [description]

        Returns:
            atom_weights: torch.Tensor shape (M, 1)
                weights of atoms in the material
            atom_fea: torch.Tensor shape (M, n_fea)
                features of atoms in the material
            self_fea_idx: torch.Tensor shape (M*M, 1)
                list of self indices
            nbr_fea_idx: torch.Tensor shape (M*M, 1)
                list of neighbor indices
            target: torch.Tensor shape (1,)
                target value for material
            cry_id: torch.Tensor shape (1,)
                input id for the material

        """
        
        df_idx = self.df.iloc[idx]
        composition = df_idx[0]
        cry_ids = df_idx[self.identifier]

        comp_dict = Composition(composition).get_el_amt_dict()
        elements = list(comp_dict.keys())

        weights = list(comp_dict.values())
        weights = np.atleast_2d(weights).T / np.sum(weights)

        try:
            atom_fea = np.vstack(
                [self.elem_features.get_fea(element) for element in elements]
            )
        except AssertionError:
            raise AssertionError(
                f"cry-id {cry_ids[0]} [{composition}] contains element types not in embedding"
            )
        except ValueError:
            raise ValueError(
                f"cry-id {cry_ids[0]} [{composition}] composition cannot be parsed into elements"
            )

        nele = len(elements)
        self_fea_idx = []
        nbr_fea_idx = []
        
        # here defines connectivity. Everyhting is connected.
        for i, _ in enumerate(elements):
            self_fea_idx += [i] * nele
            nbr_fea_idx += list(range(nele))
            
        target = df_idx['target']

        # convert all data to tensors
        atom_weights = torch.Tensor(weights)
        atom_fea = torch.Tensor(atom_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        
        cry_ids = torch.tensor(cry_ids,dtype=torch.int32).unsqueeze(-1)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(-1)

        return (
                (atom_weights, atom_fea, self_fea_idx, nbr_fea_idx),
                target,
                *cry_ids, #cry_ids is the number of elements.
                )

def collate_batch(dataset_list):
    
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point. OK
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      self_fea_idx: torch.LongTensor shape (n_i, M)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_weights: torch.Tensor shape (N, 1)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_self_fea_idx: torch.LongTensor shape (N, M)
        Indices of mapping atom to copies of itself
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_comps: list
    batch_ids: list
    """
    
    # define the lists
    batch_atom_weights = []
    batch_atom_fea = []
    batch_self_fea_idx = []
    batch_nbr_fea_idx = []
    crystal_atom_idx = []
    batch_targets = []
    batch_cry_ids = []

    cry_base_idx = 0
    for i, (inputs, target, *cry_ids) in enumerate(dataset_list):
        atom_weights, atom_fea, self_fea_idx, nbr_fea_idx = inputs

        # number of atoms for this crystal
        n_i = atom_fea.shape[0]

        # batch the features together
        batch_atom_weights.append(atom_weights)
        batch_atom_fea.append(atom_fea)

        # mappings from bonds to atoms FOCUS ON THIS!!!!
        batch_self_fea_idx.append(self_fea_idx + cry_base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_atom_idx.append(torch.tensor([i] * n_i))
        batch_cry_ids.append(cry_ids)

        # batch the targets and ids
        batch_targets.append(target)
        cry_base_idx += n_i

    return (
        
        (
            torch.cat(batch_atom_weights, dim=0),
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.cat(crystal_atom_idx),
           ),
            
        torch.stack(batch_targets,dim=0))
        
        # tuple(torch.stack(b_target, dim=0) for b_target in zip(*batch_targets)),
        # *zip(*batch_cry_ids),
    # )
#%% 
