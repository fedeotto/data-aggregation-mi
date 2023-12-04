import torch
import numpy as np
import pandas as pd
# train test split
from sklearn.model_selection import train_test_split
# # Discover
from chem_wasserstein.ElM2D_ import ElM2D
from aggr_models.discover_augmentation_v2 import my_mvn
import umap
from operator import attrgetter
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from dl_models.CrabNet.kingcrab import CrabNet
from dl_models.CrabNet.model import Model
from dl_models.roost.Model import RoostLightning, roost_config, PrintRoostLoss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
# metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from sklearn.metrics import explained_variance_score, mean_absolute_percentage_error, accuracy_score

# tasks
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from assets import utils 
from assets.preprocessing import add_column

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

""" SPLITTING FUNCTIONS """
def rnd_split(df_: pd.DataFrame,
              val_size=0.,
              test_size=0.2,
              random_state=1234,
              verbose=False,
              shuffle=True):
    if verbose:
        print('\n--random split in train-val-test')
    df = df_.copy()
    
    # train and test
    df_train, df_test = train_test_split(df, 
                                         test_size=test_size,
                                         shuffle=True,
                                         random_state=random_state)
    
    # val if needed
    if val_size!=0:
        val_size = val_size/(1-test_size)
        df_train, df_val = train_test_split(df_train, 
                                            test_size=val_size,
                                            random_state=random_state+1) 
    else: df_val = []
    assert (len(df_train) + len(df_val) + len(df_test)) == len(df)

    #shuffle
    if shuffle:
        df_train = df_train.sample(frac=1, random_state=random_state).reset_index(drop=True) 
        if df_val: df_val = df_val.sample(frac=1, random_state=random_state).reset_index(drop=True)  
        df_test = df_test.sample(frac=1, random_state=random_state).reset_index(drop=True)   
    
    return df_train, df_val, df_test


def top_split(df_: pd.DataFrame,
              column_name='target',
              val_size=0.,
              test_size=0.2,
              k_val=0.5,
              k_test=0.5,
              random_state=1234,
              ascending=False,
              verbose=False,
              shuffle=True):# default is highest on top ---> highest in test
    if verbose:
        print('\n--top split in train-val-test')
    df = df_.copy()
    N = len(df)
    
    # sort best targets on top
    df = df.sort_values(column_name,ascending=ascending)
    
    # put top in test
    test_p_size     = int(N * test_size * k_test)
    df_test_partial = df[: test_p_size]
    
    if val_size!=0:
        # next top in validation
        val_p_size     = int(N * val_size * k_val)
        df_val_partial = df[test_p_size : (test_p_size + val_p_size)]
    else: val_p_size = 0
    
    remaining = df[(test_p_size + val_p_size) :]
    
    # fill test with random 
    sampled = remaining.sample(int(N * test_size * (1-k_test)), 
                               replace=False, random_state=random_state)
    df_test = pd.concat([df_test_partial, sampled],axis=0)
    remaining = remaining.drop(index=list(sampled.index))
    
    if val_size!=0:
        # fill val with random
        sampled = remaining.sample(int(N * val_size * (1-k_val)), 
                                   replace=False, random_state=random_state+10)
        df_val = pd.concat([df_val_partial, sampled],axis=0)
        remaining = remaining.drop(index=list(sampled.index))
    else: df_val = []
        
    # remain go in the train
    df_train = remaining
    
    assert (len(df_train) + len(df_val) + len(df_test)) == N

    #shuffle
    if shuffle:
        df_train = df_train.sample(frac=1, random_state=random_state).reset_index(drop=True) 
        if df_val: df_val = df_val.sample(frac=1, random_state=random_state).reset_index(drop=True)  
        df_test = df_test.sample(frac=1, random_state=random_state).reset_index(drop=True)           
    
    return df_train, df_val, df_test


def rndtop_split(df_: pd.DataFrame,
                column_name='target',
                val_size=0.,
                test_size=0.2,
                k_val=0.5,
                k_test=0.5,
                random_state=1234,
                ascending=False,
                verbose=False,
                shuffle=True):# default is highest on top ---> highest in test
    if verbose:
        print('\n--top split in train-val-test')
    df = df_.copy()
    N = len(df)
    
    df_scores = add_column({column_name: df}, size=test_size, 
                           column='disco_score', 
                           ascending=ascending)[column_name]
    
    test_p_size     = int(N * test_size * k_test)
    df_test_partial = df_scores[df_scores[f'extraord|{column_name}']==1].sample(test_p_size)
    
    if val_size!=0:
       raise NotImplementedError
    else: val_p_size = 0
    
    # select formulas that are not in test yet
    remaining = df_scores[~df_scores['formula'].isin(df_test_partial['formula'])]
    # remove extraordinary formulas
    remaining = remaining[remaining[f'extraord|{column_name}']==0]
    
    # fill test with random 
    sampled = remaining.sample(int(N * test_size * (1-k_test)), 
                               replace=False, random_state=random_state)
    df_test = pd.concat([df_test_partial, sampled],axis=0)
    remaining = remaining.drop(index=list(sampled.index))
    
    if val_size!=0:
        raise NotImplementedError
    else: df_val = []
        
    # remain go in the train
    df_train = df_scores[~df_scores['formula'].isin(df_test['formula'])]
    
    assert (len(df_train) + len(df_val) + len(df_test)) == N

    #shuffle
    if shuffle:
        df_train = df_train.sample(frac=1, random_state=random_state).reset_index(drop=True) 
        if df_val: df_val = df_val.sample(frac=1, random_state=random_state).reset_index(drop=True)  
        df_test = df_test.sample(frac=1, random_state=random_state).reset_index(drop=True)           
    
    return df_train, df_val, df_test



def disco_split(df_: pd.DataFrame,
                val_size=0.1,
                test_size=0.2,
                k_val=0.5,
                k_test=0.5,
                random_state=1234,
                shuffle=True,
                density_weight=1.0,
                target_weight=1.0, 
                scores=['density'],
                scaled=True,
                scaler=RobustScaler()):
    
    print('\n--discover split in train-val-test')
    out_train, out_val, out_test = {}, {}, {}
    df = df_.copy()
    
    if 'density' in scores:
        # apply Discover
        # ilist = list(range(len(df)))  # to retrieve A points from embedding
        all_formula = df["formula"]
        
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
                                random_state=random_state,
                                low_memory=False)
        umap_trans = umap_init.fit(dm)
        umap_emb, r_orig_log, r_emb_log = attrgetter("embedding_", 
                                                        "rad_orig_", 
                                                        "rad_emb_")(umap_trans)
        
        # plots.plot_umap(umap_emb, self.n_a)
        umap_r_orig = np.exp(r_orig_log)

        emb    = umap_emb
        r_orig = umap_r_orig
        
        #we calculate a list of mvns based on each pair of embeddings of our compounds
        mvn_list = list(map(my_mvn, emb[:, 0], emb[:, 1], r_orig))
        pdf_list = [mvn.pdf(emb) for mvn in mvn_list]
        dens = np.sum(pdf_list, axis=0)
        density = dens.reshape(-1, 1)
        if scaled: density = scaler.fit_transform(-1*density)

    if 'target' in scores:
        raise NotImplementedError
        target = pred.ravel().reshape(-1, 1)
        if scaled: target = scaler.fit_transform(pred)

    """from density and target, calculate/update final score"""
    # determine output score
    if len(scores)==1:
        if 'target' in scores:
            final_score = target
        if 'density' in scores:   
            final_score = density
            
    elif len(scores)==2:
        if ('target' in scores and 'density' in scores):
            # combined scores
            target  = target_weight * target
            density = density_weight * density
            final_score = (target + density)/(density_weight+target_weight)
            if scaled: final_score = scaler.fit_transform(final_score)        

    df['disco_score'] = final_score

    df_train, df_val, df_test = rndtop_split(df,
                                            column_name='disco_score',
                                            val_size=val_size,
                                            test_size=test_size,
                                            k_val=k_val,
                                            k_test=k_test,
                                            random_state=random_state,
                                            ascending=False,
                                            verbose=False,
                                            shuffle=shuffle) 
       
    assert (len(df_train) + len(df_val) + len(df_test)) == len(df)
        
    return df_train, df_val, df_test


def apply_split(split_type,
                df: pd.DataFrame,
                val_size=0.,
                test_size=0.2,
                k_val=0.5,
                k_test=0.5,
                random_state=1234,
                ascending=False,
                verbose=False,
                shuffle=True):

    if split_type=='random':
        train, val, test = rnd_split(df,
                                     val_size=val_size,
                                     test_size=test_size,
                                     random_state=random_state,
                                     verbose=verbose,
                                     shuffle=shuffle)
    if split_type=='top':
        train, val, test = top_split(df,
                                     val_size=val_size,
                                     test_size=test_size,
                                     k_val=k_val,
                                     k_test=k_test,
                                     random_state=random_state,
                                     ascending=ascending,
                                     verbose=verbose,
                                     shuffle=shuffle)
    if split_type=='novelty':
        train, val, test = disco_split(df,
                                        val_size=val_size,
                                        test_size=test_size,
                                        k_val=k_val,
                                        k_test=k_test,
                                        random_state=random_state,
                                        shuffle=shuffle,
                                        density_weight=1.0,
                                        target_weight=0,
                                        scores=['density'],
                                        scaled=False,
                                        scaler=RobustScaler())
    
    return train, val, test


""" TASKS """
epsilon = 1e-10
def score_evaluation(Y_true, Y_pred, metric):
        if   metric=='mae':  score = mean_absolute_error(Y_true, Y_pred)       
        elif metric=='mse':  score = mean_squared_error(Y_true, Y_pred)  
        elif metric=='r2':   score = explained_variance_score(Y_true, Y_pred)  
        elif metric=='mape': score = mean_absolute_percentage_error(Y_true, Y_pred)
        elif metric=='mre':  score = np.abs(Y_pred - Y_true).sum() / np.abs(Y_true).sum()
        elif metric=='acc':  score = accuracy_score(Y_true, Y_pred)
        else: raise RuntimeError('Invalid regression metric')    
        return score
    
    
    
def linear_regression(train_in, train_out,
                      # val_in, val_out,
                      test_in, test_out,
                      lambda_vals=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                      val_metric='mae',
                      random_state=1234): 
    
    formulae_train = train_in.pop('formula')
    formulae_test  = test_in.pop('formula')
    
    # cross validation to estimate lambda ridge
    sss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=random_state)
    gs = GridSearchCV(Ridge(fit_intercept=True), 
                      cv=sss, 
                      param_grid={'alpha':lambda_vals},
                      scoring='neg_mean_absolute_error',
                      n_jobs=-1)
    gs.fit(train_in, train_out) 
    ridge_best = gs.best_estimator_
    
    # concatenate train and validation sets
    # train_in_complete  =  pd.concat([train_in, val_in],axis=0)
    # train_out_complete =  pd.concat([train_out, val_out],axis=0)
    # ridge_best.fit(train_in_complete, train_out_complete)
    
    #predict on test set
    test_pred = ridge_best.predict(test_in)
    return test_pred

    # output={}
    # output['mae'] = score_evaluation(test_out, test_pred, 'mae')
    # output['mse'] = score_evaluation(test_out, test_pred, 'mse')
    # output['r2']  = score_evaluation(test_out, test_pred, 'r2')
    # output['mre']  = score_evaluation(test_out, test_pred, 'mre')
    # freq_df = plots.elem_class_score(test_out,
    #                                   test_pred,
    #                                   formulae_train= formulae_train,
    #                                   formulae_test= formulae_test,
    #                                   metric= 'mae', web=False)
    # plots.plot_parity(test_out, test_pred, output['mae'])
    # return output, freq_df

def random_forest(train_in, train_out,
                  test_in, test_out,
                  val_metric='mae',
                  random_state=1234): 
    
    formulae_train = train_in.pop('formula')
    formulae_test  = test_in.pop('formula')
    
    rf = RandomForestRegressor(random_state=random_state)
    rf.fit(train_in, train_out)
    
    #predict on test set
    test_pred = rf.predict(test_in)
    return test_pred

    # output = {}
    # output['mae'] = score_evaluation(test_out, test_pred, 'mae')
    # output['mse'] = score_evaluation(test_out, test_pred, 'mse')
    # output['r2']  = score_evaluation(test_out, test_pred, 'r2')
    # output['mre']  = score_evaluation(test_out, test_pred, 'mre')
    # freq_df = plots.elem_class_score(test_out,
    #                                  test_pred,
    #                                  formulae_train= formulae_train,
    #                                  formulae_test= formulae_test,
    #                                  metric= 'mae', web=False)
    # # plots.plot_parity(test_out, test_pred, output['mae'])
    # return output, freq_df


def logistic_classification(train_in, train_out,
                            # val_in, val_out,
                            test_in, test_out,
                            C_vals=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            val_metric='acc',
                            random_state=1234): 
    
    formulae_train = train_in.pop('formula')
    formulae_test  = test_in.pop('formula')
    
    # cross validation to estimate C
    sss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=random_state)
    gs = GridSearchCV(LogisticRegression(penalty='l2', fit_intercept=True, 
                                         max_iter=3000, solver='lbfgs'), 
                      cv=sss, 
                      param_grid={'C':C_vals},
                      scoring='accuracy',
                      n_jobs=-1)
    gs.fit(train_in, train_out) 
    logistic_best = gs.best_estimator_
    
    # concatenate train and validation sets
    # train_in_complete  =  pd.concat([train_in, val_in],axis=0)
    # train_out_complete =  pd.concat([train_out, val_out],axis=0)
    # ridge_best.fit(train_in_complete, train_out_complete)
    
    #predict on test set
    test_pred = logistic_best.predict(test_in)
    return test_pred

    # output = {}
    # output['acc'] = score_evaluation(test_out, test_pred, 'acc')
    # return output

    
def crabnet(train_in, train_out,
            # val_in, val_out,
            test_in, test_out,
            transfer,
            prop,
            key_B,
            crabnet_kwargs:dict,
            classification=False,
            random_state:int = 1234):
    
    if classification:
        train_out.name = 'target'
        # val_out.name = 'target'
        test_out.name = 'target'
            
    train_df = pd.concat([train_in, train_out],axis=1)
    test_df = pd.concat([test_in, test_out],axis=1)
    
    train_formulae = train_df['formula']
    test_formulae  = test_df['formula']
    
    crabnet_model = Model(CrabNet(compute_device=device).to(device),
                          classification=False,
                          random_state=random_state,
                          verbose=crabnet_kwargs['verbose'],
                          discard_n=crabnet_kwargs['discard_n'])
    
    # little_val for early stopping
    little_val = train_df.sample(frac=0.10, random_state=random_state)
    train_df = train_df.drop(index=little_val.index)

    if transfer:
        crabnet_model.load_network(f'./transfer_models/crabnet_{prop}_{key_B}.pth')
        output_nn = crabnet_model.model.output_nn
        last_layer= list(output_nn.children())[-1]
        last_layer.reset_parameters()

    crabnet_model.load_data(train_df, batch_size=crabnet_kwargs['batch_size'], train=True)
    crabnet_model.load_data(little_val, batch_size=crabnet_kwargs['batch_size'], train=False)
    crabnet_model.fit(epochs = crabnet_kwargs['epochs'])
    
    #predict on test set
    
    # loading test dataset
    crabnet_model.load_data(test_df, batch_size=crabnet_kwargs['batch_size'], train=False)
    
    test_out, test_pred, _, _ = crabnet_model.predict(crabnet_model.data_loader)
    return test_pred

    # output = {}
    # if   classification==True:
    #     output['acc'] = score_evaluation(test_out, test_pred, 'acc')
        
    # elif classification==False:  
    #     output['mae'] = score_evaluation(test_out, test_pred, 'mae')
    #     output['mse'] = score_evaluation(test_out, test_pred, 'mse')
    #     output['r2']  = score_evaluation(test_out, test_pred, 'r2')
    #     output['mre']  = score_evaluation(test_out, test_pred, 'mre')
    #     # plots.plot_parity(test_out, test_pred, output['mae'])
    # return output
    
    
def roost(train_in, train_out,
        test_in, test_out,
        transfer,
        prop,
        key_B,
        roost_kwargs,
        random_state: int = 1234):
    
    train_df = pd.concat([train_in, train_out],axis=1)
    test_df = pd.concat([test_in, test_out],axis=1)
    
    roost_config['data_params']['batch_size'] = roost_kwargs['batch_size']

    if transfer:
        roost     = RoostLightning.load_from_checkpoint(f'./transfer_models/roost_{prop}_{key_B}.ckpt')
        output_nn = list(roost.model.children())[1]
        last_layer= list(output_nn.children())[-1]
        last_layer.reset_parameters()
    else:
        roost = RoostLightning(**roost_config)
    
    val_df    = train_df.sample(frac=0.10, random_state=random_state)
    train_df  = train_df.drop(index=val_df.index)
    
    roost.load_data(train_df, which='train')
    roost.load_data(val_df, which='val')
    roost.load_data(test_df, which='test')
    
    trainer = pl.Trainer(max_epochs=roost_kwargs['epochs'],
                         accelerator=device.type,
                         callbacks=[PrintRoostLoss(),
                                    EarlyStopping(monitor="val_loss", 
                                                  mode='min',
                                                  patience=20, 
                                                  verbose=True)
                                    ])
    trainer.fit(roost, 
                train_dataloaders=roost.train_loader,
                val_dataloaders  =roost.val_loader)
    
    preds = roost.predict(roost.test_loader)
    
    return preds


def apply_all_tasks(train,
                    # val,
                    test,
                    test_key,
                    tasks_list,
                    prop,
                    key_B,
                    crabnet_kwargs = {'epochs':100, 'batch_size':128},
                    roost_kwargs   = {'epochs':100, 'batch_size':128},
                    reg_metrics = ['mae','mse','r2','mape','mre'],
                    clas_metrics = ['acc'],
                    random_state = 1234,
                    verbose=False
                    ):
    """formula must be the first column, target must be the column after the features, extraords must follow"""
    # to store inputs and outputs
    d = {}
    
    # for every element in test, count occurences in train (and test)
    occ = utils.count_occurrences_traintest(train['formula'], test['formula'])
    true = test.loc[:, ['formula', 'target']]
    
    # to store mean accuracies
    scores = {}
    
    # loop over tasks
    for task in tasks_list:
        scores[task]={}
    ### prepare input ###
        if ('crabnet' in task) or ('roost' in task):
            d['train_in']  = train.loc[:,'formula']
            d['test_in']   = test.loc[:,'formula']
            # d['val_in']  = val.loc[:,'formula']
        else:
            d['train_in']  = train.loc[:,'formula':'target'].iloc[:,:-1]
            d['test_in']   = test.loc[:,'formula':'target'].iloc[:,:-1]
            # d['val_in']  = val.loc[:,'formula':'target'].iloc[:,1:-1]
            
    ### prepare output ###
        if 'regression' in task:
            metrics=reg_metrics
            true = test.loc[:, ['formula', 'target']]
            d['train_out'] = train.loc[:,'target']
            d['test_out']  = test.loc[:,'target']
            # d['val_out'] = val.loc[:,'target']
        elif 'classification' in task:
            metrics=clas_metrics
            true = test.loc[:,['formula', f'extraord|{test_key}']]
            d['train_out'] = train.loc[:,f'extraord|{test_key}']
            d['test_out']  = test.loc[:,f'extraord|{test_key}']
            # d['val_out'] = val.loc[:,f'extraord|{test_key}']
            
        
        ### apply task ###
        # regression
        if task=='linear_regression':
            pred = linear_regression(d['train_in'],d['train_out'],
                                    d['test_in'], d['test_out'],
                                    random_state=random_state)
            
        elif task=='random_forest_regression':
            pred = random_forest(d['train_in'],d['train_out'], 
                                d['test_in'], d['test_out'],
                                random_state=random_state)
            
        elif task=='crabnet_regression':
            pred = crabnet(d['train_in'],d['train_out'],
                        # d['val_in'],d['val_out'],
                        d['test_in'], d['test_out'],
                        transfer      = False,
                        prop          = prop,
                        key_B         = key_B,
                        classification=False,
                        crabnet_kwargs = crabnet_kwargs,
                        random_state=random_state)
    
        elif task == 'roost_regression':
            pred = roost(d['train_in'], d['train_out'],
                        d['test_in'], d['test_out'],
                        transfer     = False,
                        prop         = prop,
                        key_B        = key_B,
                        roost_kwargs = roost_kwargs,
                        random_state = random_state)
            
        elif task == 'transf_roost_regression':
            pred = roost(d['train_in'], d['train_out'],
                        d['test_in'], d['test_out'],
                        transfer     = True,
                        prop         = prop,
                        key_B        = key_B,
                        roost_kwargs = roost_kwargs,
                        random_state = random_state)
        
        elif task  == 'transf_crabnet_regression':
            pred = crabnet(d['train_in'],d['train_out'],
                            # d['val_in'],d['val_out'],
                            d['test_in'], d['test_out'],
                            transfer     = True,
                            prop         = prop,
                            key_B        = key_B,
                            classification=False,
                            crabnet_kwargs = crabnet_kwargs,
                            random_state=random_state)
            
        # extraord classification
        elif task=='logistic_classification':       
            pred = logistic_classification(d['train_in'],d['train_out'],
                                        d['test_in'], d['test_out'],
                                        random_state=random_state)

        elif task=='crabnet_classification':
            pred = crabnet(d['train_in'],d['train_out'],
                        # d['val_in'],d['val_out'],
                        d['test_in'], d['test_out'],
                        classification=True,
                        crabnet_kwargs = crabnet_kwargs,
                        random_state=random_state)
            
        
        ### accuracies ###
        output = {}
        true.columns = ['formula', 'target']
        pred = pd.DataFrame({'formula':true['formula'], 'target':pred})
        # add a column to occ with all accuracies with respect to the task
        occ = utils.add_accuracies_to_count(occ, true, pred, task, metrics)
        # global metrics
        for metric in metrics:
            scores[task][metric] = score_evaluation(true['target'], pred['target'], metric)
         
    return scores, occ
             
   


# def apply_all_tasks(train, # 1 dataset
#                     # val, # 1 dataset
#                     test,  # 1 dataset
#                     test_key,
#                     tasks_list,
#                     crabnet_kwargs = {'epochs':100},
#                     random_state=1234,
#                     verbose=False):
#     d = {}  
#     freq_df = None
#     output = {task:{} for task in tasks_list}
    
#     for task in tasks_list:
#         # prepare input
#           if 'crabnet' in task:
#               d['train_in']  = train.loc[:,'formula']
#               d['test_in']   = test.loc[:,'formula']
#               # d['val_in']  = val.loc[:,'formula']
#           else:
#               d['train_in']  = train.loc[:,'formula':'target'].iloc[:,:-1]
#               d['test_in']   = test.loc[:,'formula':'target'].iloc[:,:-1]
#               # d['val_in']  = val.loc[:,'formula':'target'].iloc[:,1:-1]
             
#         # prepare output
#           if 'regression' in task:
#               d['train_out'] = train.loc[:,'target']
#               d['test_out']  = test.loc[:,'target']
#               # d['val_out'] = val.loc[:,'target']
#           elif 'classification' in task:
#               d['train_out'] = train.loc[:,f'extraord|{test_key}']
#               d['test_out']  = test.loc[:,f'extraord|{test_key}']
#               # d['val_out'] = val.loc[:,f'extraord|{test_key}']
             
#           output[task][test_key], freq_df = apply_task(d, task, 
#                                                       crabnet_kwargs=crabnet_kwargs,
#                                                       random_state=random_state,
#                                                       verbose=verbose)      
                     
#     return output, freq_df



# def apply_task(d,    # dict of data
#                 task,
#                 crabnet_kwargs = {'epochs':100},
#                 random_state=1234,
#                 verbose=False):

#     freq_df = None    
#     if task=='linear_regression':
#         if verbose:
#             print('\t\t--linear regression')
#         output, freq_df = linear_regression(d['train_in'], d['train_out'],
#                                             # d['val_in'],   d['val_out'],
#                                             d['test_in'], d['test_out'],
#                                             random_state=random_state)
    
#     elif task=='logistic_classification':
#         if verbose:
#             print('\t\t--logistic classification')        
#         output = logistic_classification(d['train_in'], d['train_out'],
#                                           # d['val_in'],   d['val_out'],
#                                           d['test_in'], d['test_out'],
#                                           random_state=random_state)
        
#     elif task=='random_forest_regression':
#         output, freq_df = random_forest(d['train_in'], d['train_out'], 
#                                         d['test_in'], d['test_out'])
    
#     elif task=='crabnet_regression':
#         if verbose:
#             print('\t\t--crabnet regression')
#         output = crabnet(d['train_in'], d['train_out'],
#                           # d['val_in'],   d['val_out'],
#                           d['test_in'], d['test_out'],
#                           classification=False,
#                           crabnet_kwargs = crabnet_kwargs,
#                           random_state=random_state)
        
#     elif task=='crabnet_classification':
#         if verbose:
#             print('\t\t--crabnet classification') 
#         output = crabnet(d['train_in'], d['train_out'],
#                           # d['val_in'],   d['val_out'],
#                           d['test_in'], d['test_out'],
#                           classification=True,
#                           crabnet_kwargs = crabnet_kwargs,
#                           random_state=random_state)
        
#     return output, freq_df







