from dl_models.roost.Model import roost_config, RoostLightning, PrintRoostLoss
from dl_models.CrabNet.kingcrab import CrabNet
from dl_models.CrabNet.model import Model
from assets.utils import load_dataset, print_info
from settings import *
from assets.preprocessing import preprocess_dataset, add_column
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

props_list = [ 
            'thermalcond',
            'bulkmodulus',
            'seebeck',
            'sigma',
            'bandgap',
            'bulkmodulus',
            'shearmodulus'
            ]

def train_transfer_models():
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)

    for prop in props_list:
        data_raw = load_dataset(prop)  
        print_info(data_raw, prop); print('')
        
        data_clean = preprocess_dataset(data_raw, prop, merging,
                                        epsilon_T, 
                                        med_sigma_multiplier,
                                        mult_outliers,
                                        ascending_setting[prop])
        
        print(''); print_info(data_clean, prop)
        keys_B = [key for key in list(data_clean.keys()) if key=='mpds' or key=='mp']
        for key_B in keys_B:
            df = data_clean[key_B].copy()
            df_val = df.sample(frac=0.10, random_state=seed)
            df_train = df.drop(df_val.index)

            roost_config['data_params']['batch_size'] = roost_kwargs['batch_size']
            roost_config['seed'] = seed
            roost = RoostLightning(**roost_config)

            roost.load_data(df_train, which='train')
            roost.load_data(df_val, which='val')

            name = f'roost_{prop}_{key_B}'
            trainer = pl.Trainer(accelerator=device.type,
                                callbacks=[ModelCheckpoint(monitor     ='val_loss',
                                                            save_top_k =1,
                                                            dirpath    =f'transfer_models/',
                                                            filename   = name,
                                                            mode       ='min'),
                                            EarlyStopping(monitor='val_loss', patience=30),
                                            PrintRoostLoss()],
                                max_epochs=roost_kwargs['epochs'])
            
            trainer.fit(roost, 
                        train_dataloaders=roost.train_loader, 
                        val_dataloaders=roost.val_loader)  

            crabnet = Model(CrabNet(compute_device=device).to(device),
                            classification=False,
                            random_state=seed,
                            verbose=crabnet_kwargs['verbose'],
                            discard_n=crabnet_kwargs['discard_n'])
            
            df_train = df_train[['formula','target']]
            df_val   = df_val[['formula','target']]
            crabnet.load_data(df_train, train=True, batch_size=crabnet_kwargs['batch_size'])
            crabnet.load_data(df_val, train=False, batch_size=crabnet_kwargs['batch_size'])

            crabnet.fit(epochs=crabnet_kwargs['epochs'])
            crabnet.save_network(f'./transfer_models/crabnet_{prop}_{key_B}.pth')

if __name__ == "__main__":
    train_transfer_models()