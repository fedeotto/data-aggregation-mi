import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_, calculate_gain
from torch_geometric.utils import scatter
from dl_models.roost.Model import GATRoostLayer, Simple_linear, WeightedAttentionPooling, WeightedAttentionPooling_comp, softmax_weights
from torch.nn import L1Loss
import pandas as pd
import pytorch_lightning as pl
import itertools
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dl_models.roost.Data import data_from_composition
from torch_geometric.loader import DataLoader

roost_config={
              'seed': 0, 
              'model_name': 'roost',
              'epochs': 50,
              'patience': 10,
              'test_size': 0.1,
              'val_size': 0.1,
              'train': True,
              'evaluate': True,
              'data_params': {"batch_size": 128,
                              "pin_memory": False,
                              "shuffle": True,
                              "data_seed": 0},
              'setup_params': {"optim": "AdamW",
                               "learning_rate": 3e-4, #3e-4
                               "weight_decay": 1e-6, #1e-6
                               "momentum": 0.9,
                               "loss": L1Loss}, #0.9
              'model_params': {'input_dim': 200,
                                'output_dim': 1,
                                'hidden_layer_dims': [1024,512,256,128,64],
                                'n_graphs': 3,
                                'elem_heads': 4,
                                'internal_elem_dim': 64,
                                'g_elem_dim': 256,
                                'f_elem_dim': 256,
                                'comp_heads': 4,
                                'g_comp_dim': 256,
                                'f_comp_dim': 256,
                                'batchnorm': False,     
                                'negative_slope': 0.2
                                }}


class PrintRoostLoss(pl.Callback):
    def __init__(self):
        super().__init__()
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics['train_loss']
        val_loss   = trainer.callback_metrics['val_loss']

        print(f'Epoch {trainer.current_epoch}/{trainer.max_epochs}')
        print(f'Train loss: {train_loss:.4f} , \t Val. loss: {val_loss:.4f}')

        
class RoostLightning(pl.LightningModule):
    def __init__(self, **roost_config):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self.model = Roost(**roost_config['model_params'])
        self.loss_module=L1Loss()
    
    def forward(self, batch):
        return self.model(batch.x, batch.edge_index, batch.pos, batch.batch)

    def configure_optimizers(self):
        # We use AdamW optimizer with MultistepLR scheduler as in the original Roost model
        optimizer = torch.optim.AdamW(self.parameters(),lr=roost_config['setup_params']['learning_rate'], 
                                      weight_decay=roost_config['setup_params']['weight_decay']) 
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.3)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        preds = self.forward(batch)
        act   = batch.y
        loss  = self.loss_module(act,preds)
        self.log("train_loss", loss,on_epoch=True, prog_bar=True, batch_size=roost_config['data_params']['batch_size'])
        return loss
    
    def load_data(self, df, which='train'):
        data_list = data_from_composition(df, elem_prop='mat2vec')
        if which=='train':
            self.train_loader = DataLoader(data_list, 
                                           batch_size = roost_config['data_params']['batch_size'],
                                           shuffle=True
                                           )
        elif which == 'val':
            self.val_loader = DataLoader(data_list, 
                                         batch_size = 64,
                                         shuffle=False
                                         )
        
        elif which == 'test':
            self.test_loader = DataLoader(data_list, 
                                          batch_size = 64,
                                          shuffle=False)
    
    def predict(self, test_loader):
        predictions = []
        for batch in test_loader:
            preds = self.forward(batch)
            predictions.append(preds)
        return torch.hstack(predictions).detach().cpu().numpy()        

    def validation_step(self, batch, batch_idx):
        preds = self.forward(batch)
        act   = batch.y
        loss  = self.loss_module(act,preds)
        self.log("val_loss", loss,  on_epoch=True,batch_size=roost_config['data_params']['batch_size'])

        return loss
    
    def save_network(self, path):
        save_dict = {'weights': self.state_dict()}
        torch.save(save_dict, path)
        
    def load_network(self, path):
        network = torch.load(path)
        self.load_state_dict(network['weights'])        

    def test_step(self, batch, batch_idx):
        pred , loss, mae, rmse = self._get_preds_loss_mae_rmse(batch, batch_idx)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=roost_config['data_params']['batch_size'])
        self.log("test_mae", mae, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=roost_config['data_params']['batch_size'])
        self.log("test_rmse", rmse, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=roost_config['data_params']['batch_size'])
        return pred, loss, mae, rmse

    def _get_preds_loss_mae_rmse(self, batch, batch_idx):
        '''convenience function since train/valid/test steps are similar'''
        y    = batch.y
        pred = self.forward(batch)
        loss = self.loss_module(pred, y)
        mae=mean_absolute_error(pred, y)
        rmse=mean_squared_error(pred, y, squared=False)
        return pred, loss, mae, rmse

class DescriptorNetwork(nn.Module):
    """
    The Descriptor Network is the message passing section of the
    Roost Model.
    """

    def __init__(self, input_dim, n_graphs, elem_heads, internal_elem_dim, g_elem_dim, f_elem_dim, 
                 comp_heads, g_comp_dim, f_comp_dim, negative_slope=0.2,bias=False):
        
        super().__init__()

        # apply linear transform to the input to get a trainable embedding
        # NOTE -1 here so we can add the weights as a node feature
        self.project_fea = nn.Linear(input_dim, internal_elem_dim - 1,bias=False)

        # create a list of Message passing layers
        self.graphs = nn.ModuleList(
            [
                GATRoostLayer(internal_elem_dim, internal_elem_dim, g_elem_dim, f_elem_dim, elem_heads, negative_slope, bias)             
                for i in range(n_graphs)
            ]
        )

        # define a global pooling function for materials
        self.comp_pool = nn.ModuleList(
            [
                WeightedAttentionPooling_comp(
                    gate_nn=Simple_linear(internal_elem_dim, 1, f_comp_dim, negative_slope,bias),
                    message_nn=Simple_linear(internal_elem_dim, internal_elem_dim, g_comp_dim, negative_slope,bias),
                )
                for _ in range(comp_heads)
            ]
        )

    def reset_parameters(self):
        gain=calculate_gain('leaky_relu', self.negative_slope)
        xavier_uniform_(self.project_fea.weight, gain=gain)
        for graph in self.graphs:
            graph.reset_parameters()
        for head in self.comp_pool:
            head.reset_parameters()

    def forward(self, x, edge_index, pos, batch_index=None):
        """
        """
        # embed the original features into a trainable embedding space
        weights=pos
        # constructing internal representations of the elements
        x = self.project_fea(x)
        x = torch.cat([x,weights.unsqueeze(1)],dim=1)

        # apply the message passing functions
        for graph_func in self.graphs:
            x = graph_func(x,edge_index,pos)

        # generate crystal features by pooling the elemental features
        head_fea = []
        if batch_index is not None:
            for attnhead in self.comp_pool:
                head_fea.append(
                    attnhead(x, edge_index, pos, batch_index)
                )
        else:
            for attnhead in self.comp_pool:
                head_fea.append(
                    attnhead(x, edge_index, pos)
                )
                
        return torch.mean(torch.stack(head_fea), dim=0)
    

    def __repr__(self):
        return self.__class__.__name__
    

class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network (copied from Roost Repository)
    """

    def __init__(self, internal_elem_dim, output_dim, hidden_layer_dims, batchnorm=False, negative_slope=0.2):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super().__init__()

        dims = [internal_elem_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 1)]
            )
        else:
            self.bns = nn.ModuleList([nn.Identity() for i in range(len(dims) - 1)])

        self.res_fcs = nn.ModuleList(
            [
                nn.Linear(dims[i], dims[i + 1], bias=False)
                if (dims[i] != dims[i + 1])
                else nn.Identity()
                for i in range(len(dims) - 1)
            ]
        )
        self.acts = nn.ModuleList([nn.LeakyReLU(negative_slope) for _ in range(len(dims) - 1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, res_fc, act in zip(self.fcs, self.bns, self.res_fcs, self.acts):
            x = act(bn(fc(x))) + res_fc(x)

        return  self.fc_out(x)

    def __repr__(self):
        return self.__class__.__name__

class Roost(nn.Module):
    """
    Roost model (copied from Roost Repository)
    """

    def __init__(self, input_dim, output_dim, hidden_layer_dims, 
                 n_graphs, elem_heads, internal_elem_dim, g_elem_dim, f_elem_dim, 
                 comp_heads, g_comp_dim, f_comp_dim,
                 batchnorm=True, negative_slope=0.2):
        """
        Inputs
        ----------
        input_dim: int, initial size of the element embeddings
        output_dim: int, dimensinality of the target
        hidden_layer_dims: list(int), list of sizes of layers in the residual network
        n_graphs: int, number of graph layers
        elem_heads: int, number of attention heads for the element attention
        interanl_elem_dim: int, size of the element embeddings inside the model
        g_elem_dim: int, size of hidden layer in g_func in elemental graph layers
        f_elem_dim: int, size of hidden layer in f_func in elemental graph layers
        comp_heads: int, number of attention heads for the composition attention
        g_comp_dim: int, size of hidden layer in g_func in composition graph layers
        f_comp_dim: int, size of hidden layer in f_func in composition graph layers
        batchnorm: bool, whether to use batchnorm in the residual network
        negative_slope: float, negative slope for leaky relu

        """
        super().__init__()

        self.n_graphs = n_graphs
        self.negative_slope = negative_slope
        self.comp_heads = comp_heads
        self.internal_elem_dim = internal_elem_dim

        self.material_nn = DescriptorNetwork(input_dim, n_graphs, elem_heads, internal_elem_dim, g_elem_dim, f_elem_dim, 
                 comp_heads, g_comp_dim, f_comp_dim, negative_slope=0.2,bias=False)
        
        self.resnet = ResidualNetwork(internal_elem_dim, output_dim, hidden_layer_dims,
                                      batchnorm=batchnorm, negative_slope=negative_slope)
        self.reset_parameters()
    
    def reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('leaky_relu'))
    
    def embed(self, 
              data_loader,
              resnet = True):
        
        embs = []
        for data in data_loader:
            mat_nn_out = self.material_nn(data.x, 
                                          data.edge_index, 
                                          data.pos,
                                          data.batch)
            
            if resnet:
                layers = list(self.resnet.children())[0]
                acts   = list(self.resnet.children())[-2][:-1] #exclude last
                for layer,act in zip(layers,acts):
                    mat_nn_out = act(layer(mat_nn_out))
                
                crys_fea = layers[-1](mat_nn_out)
                embs.append(crys_fea)
            else:
                embs.append(mat_nn_out)                
        embs = torch.vstack(embs)
        embs = embs.detach().numpy()
        
        return embs

    def forward(self, x, edge_index, pos, batch_index=None):
        if batch_index is not None:
            x = self.material_nn(x, edge_index, pos, batch_index)
        else:
            x = self.material_nn(x, edge_index, pos)
            
        x = self.resnet(x)
        if(x.dim()==2):
            x=x.squeeze(-1)
        return x

    def __repr__(self):
        return self.__class__.__name__
    
    
    