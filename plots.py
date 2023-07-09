# plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
# matplotlib
import matplotlib.pyplot as plt
import numpy as np
# imports
from chem_wasserstein.ElM2D_ import ElM2D
import umap
# from pymatviz.pymatviz.ptable import ptable_heatmap
import pandas as pd
from chem import _element_composition
from collections import Counter
from operator import attrgetter
from metrics import equitability_index
import pickle

pio.renderers.default="browser"    # 'svg' or 'browser'
pio.templates.default="simple_white"

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size']  = 16

def plot_self_augment(prop = 'bulkmodulus',
                      discotest=False):
    
    '''Plotting self augment result for RF & CrabNet'''
    
    if discotest:
        path = f'./results/results_6_discotest_{prop}.pkl'
    else:
        path = f'./results/results_6_{prop}.pkl'

    with open(f'{path}', 'rb') as handle:
        result = pickle.load(handle)
    
    result = result.dropna()
    result = result[['crabnet_regression','random_forest_regression']]
    crab_disco_avg = result.loc[:,('crabnet_regression','disco')].mean(axis=1)
    crab_disco_std = result.loc[:,('crabnet_regression','disco')].std(axis=1)
    crab_rnd_avg = result.loc[:,('crabnet_regression','random')].mean(axis=1)
    crab_rnd_std = result.loc[:,('crabnet_regression','random')].std(axis=1)
    rf_disco_avg = result.loc[:,('random_forest_regression','disco')].mean(axis=1)
    rf_disco_std = result.loc[:,('random_forest_regression','disco')].std(axis=1)
    rf_rnd_avg = result.loc[:,('random_forest_regression','random')].mean(axis=1)
    rf_rnd_std = result.loc[:,('random_forest_regression','random')].std(axis=1)
    
    # drop only the second last row in all datasets
    crab_disco_avg = crab_disco_avg.drop(crab_disco_avg.index[-2])
    crab_disco_std = crab_disco_std.drop(crab_disco_std.index[-2])
    crab_rnd_avg = crab_rnd_avg.drop(crab_rnd_avg.index[-2])
    crab_rnd_std = crab_rnd_std.drop(crab_rnd_std.index[-2])
    rf_disco_avg = rf_disco_avg.drop(rf_disco_avg.index[-2])
    rf_disco_std = rf_disco_std.drop(rf_disco_std.index[-2])
    rf_rnd_avg = rf_rnd_avg.drop(rf_rnd_avg.index[-2])
    rf_rnd_std = rf_rnd_std.drop(rf_rnd_std.index[-2])

    # bugghino fix
    crab_disco_avg[0] = crab_rnd_avg[0]
    
    fig, ax = plt.subplots(figsize=(12,6), nrows=1, ncols=2)
    
    x = np.arange(0.05, 1.05, 0.05)
    
    # xticks = range(0,len(result),4)
    ax[0].plot(x,
               crab_disco_avg,
               marker='o',
               markersize=5,
               linestyle='--',
               label='Disco')
    
    ax[0].fill_between(x,
                    crab_disco_avg-crab_disco_std, 
                    crab_disco_avg+crab_disco_std,
                    alpha=0.1)
    
    ax[0].plot(x,
               crab_rnd_avg,
               marker='o',
               markersize=5,
               linestyle='--',
               label ='Random')
    
    ax[0].fill_between(x,
                       crab_rnd_avg-crab_rnd_std, 
                       crab_rnd_avg+crab_rnd_std,
                       alpha=0.1)
    ax[0].grid()
    ax[0].set_xlabel('Train size (%)', labelpad=5)
    ax[0].set_ylabel('MAE', labelpad=5)
    
    ax[1].plot(x,
               rf_disco_avg,
               marker='o',
               markersize=5,
               linestyle='--',
               label='Disco')
    
    ax[1].fill_between(x,
                       rf_disco_avg-rf_disco_std, 
                       rf_disco_avg+rf_disco_std,
                       alpha=0.1)
    
    ax[1].plot(x,
               rf_rnd_avg,
               marker='o',
               markersize=5,
               linestyle='--',
               label ='Random')
    
    ax[1].fill_between(x,
                       rf_rnd_avg-rf_rnd_std, 
                       rf_rnd_avg+rf_rnd_std,
                       alpha=0.1)
    
    ax[1].grid()
    
    # xticklabels = [0.0,0.2,0.4,0.6,0.8,1.0]
    
    # ax[1].set_xticks(xticks)
    # ax[1].set_xticklabels(xticklabels)
    ax[1].set_xlabel('Train size (%)')
    plt.savefig('self_augment_bulkmodulus_standard.png')
    plt.legend()
        
        
plot_self_augment(prop = 'bulkmodulus',
                  discotest=False)
      
        

def add_prop_to_violins(fig, ind, dfs, prop, l):
    colors = {'japdata':'purple','citrine':'pink', 'mpds':'orange', 'te':'green', 
              'mp':'blue', 'aflow':'grey', 'zhuo':'red'}
    sides = ['negative', 'positive']  
    datasets = {'japdata':'MDR','citrine':'Citrine', 'mpds':'MPDS', 'te':'MRL',
                'mp':'Mat. Proj.', 'aflow':'AFLOW', 'zhuo':'Zhuo et al.'}
    titles = {'thermalcond': 'Thermal conductivity', 'superconT':'Supercon. Trans. Temp.',
            'seebeck':'Seebeck', 'rho':'Electrical resistivity', 
             'sigma':'Electrical conductivity', 'bandgap':'Bandgap', 
             'bulkmodulus':'Bulk modulus', 'shearmodulus':'Shear modulus'}  
    units = {'thermalcond': 'W/mK (log10)', 'superconT':'K',
            'seebeck':'μV / K', 'rho':'Ω · cm (log10)', 
             'sigma':'S/cm (log10)', 'bandgap':'eV', 
             'bulkmodulus':'GPa', 'shearmodulus':'GPa'}
    names = list(dfs.keys())
    row = (ind // 4) + 1  # Assign row based on the index
    col = (ind % 4) + 1  # Assign col cyclically from 1 to 4
    # row = 1 if ind < 4 else 2  # Assign row based on the index
    # col = ind+1 if ind < 4 else (ind-3)        # Assign col cyclically from 1 to 4
    for i, (key, df) in enumerate(dfs.items()):
        fig.add_trace(go.Violin(#x=pd.Series(data=[prop for i in range(len(df))]), 
                                y=df['target'], 
                                # box_visible=True,
                                meanline_visible=True, 
                                # side=sides[i],
                                # legendgroup=key, 
                                # offsetgroup=prop,
                                scalegroup=prop,
                                marker=dict(size=4, opacity=0.),
                                points='outliers',
                                scalemode='width',
                                # fillcolor=colors[i], 
                                line_color=colors[key],
                                opacity=0.6, 
                                name=datasets[key],
                                showlegend=True if key not in l else False,
                                # x0=prop+f'[{units[prop]}]',
                                x0 = 'A' if i==0 else 'B',
                                yaxis=f'y{ind+1}'
                                ),
                      row=row, 
                      col=col)
        l.append(key)
    fig.update_xaxes(title_text=titles[prop]+'<br>'+f'[{units[prop]}]', 
                     title_standoff=0,
                     tickvals=[],
                     ticktext=[],
                     row=row, 
                     col=col)
    return fig

def plot_violins(fig):
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=1.05,
        xanchor="left",
        x=0.02,
        font=dict(size=16)
    ))
    st = 0.
    tk=dict(size=10)
    tf=dict(size=15)
    # fig.update_layout(
    #     yaxis=dict(title_standoff=st,
    #                titlefont=tf,
    #                tickfont=tk),
    #     yaxis2=dict(
    #         overlaying="y",
    #         side="left",
    #         title_standoff=st,
    #         titlefont=tf,
    #         tickfont=tk
    #     ),
    #     yaxis3=dict(
    #         overlaying="y",
    #         side="left",
    #         title_standoff=st,
    #         titlefont=tf,
    #         tickfont=tk
    #     ),
    #     yaxis4=dict(
    #         overlaying="y",
    #         side="left",
    #         title_standoff=st,
    #         titlefont=tf,
    #         tickfont=tk
    #     ),
    #     yaxis5=dict(
    #         overlaying="y",
    #         side="left",
    #         title_standoff=st,
    #         titlefont=tf,
    #         tickfont=tk
    #     ),
    #     yaxis6=dict(
    #         overlaying="y",
    #         side="left",
    #         title_standoff=st,
    #         titlefont=tf,
    #         tickfont=tk
    #     ),
    #     yaxis7=dict(
    #         overlaying="y",
    #         side="left",
    #         title_standoff=st,
    #         titlefont=tf,
    #         tickfont=tk
    #     ),
    #     yaxis8=dict(
    #         overlaying="y",
    #         side="left",
    #         title_standoff=st,
    #         titlefont=tf,
    #         tickfont=tk
    #     )
    # )
    # fig.update_traces(meanline_visible=True)
    fig.update_layout(width=1000,  # 1500
                      height=800,  # 500
                      margin=dict(l=10, r=10, t=10, b=10)
                    #   violingap=0.05, 
                    #   violinmode='overlay',
                    #   showlegend=False
    )
    fig.show() 
    return fig
    


def plot_distinct_histos(dfs, bins, prop, extraord=True):
    n = len(dfs)
    fig = make_subplots(rows=n, cols=1)
    colors = ['blue', 'orange', 'green']
    text = ["extraord", 'extraord', 'extraord']
    
    high = max([max(df['target']) for df in dfs.values()])
    low  = min([min(df['target']) for df in dfs.values()])
    size = (high-low)/bins
    for i, (key, df) in enumerate(dfs.items()):
        fig.add_trace(go.Histogram(x=df['target'],
                             xbins=dict(
                                 start = low,
                                 end   = high,
                                 size = size), 
                             autobinx=False,
                             name = key),
                      row=i+1, col=1)
        if extraord:
            bar = df[df[f'extraord|{key}']==1].iloc[-1]['target']
            fig.add_vline(x=bar, 
                          line_width=3, 
                          line_dash="dash", 
                          line_color=colors[i],
                          annotation_text=text[i], 
                          annotation_position="top right",
                          row=i+1, col=1)
            
    fig.update_xaxes(range=[low, high])  
    fig.update_layout(title=prop, font=dict(size=15))
    
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=1.2,
        xanchor="right",
        x=1.02,
        font=dict(size=30)
    ))
    
    
    
    fig.show()   
    if extraord: title = f'plots/datasets/{prop}_distinct_extraord.png'
    else:        title = f'plots/datasets/{prop}_distinct.png'
    pio.write_image(fig, title, width=4*300, height=4*300, scale=1)
    
    
# def elem_class_score(targets,
#                      preds,
#                      formulae_train: pd.Series,
#                      formulae_test: pd.Series,
#                      metric: str = 'MAE',
#                      web = False
#                      ): 
    
    
#     train_elems_frequency = []
    
#     df = pd.DataFrame(None)
#     df['formula'] = formulae_test
#     df['targets'] = targets
#     df['preds'] = preds
    
#     test_dicts = formulae_test.apply(_element_composition)
#     test_list = [item for row in test_dicts for item in row.keys()]
#     test_counter = Counter(test_list)
    
#     freq_df = pd.DataFrame(None)
#     freq_df['test_elems'] = test_list
#     freq_df = freq_df.drop_duplicates('test_elems').reset_index(drop=True)
    
#     count_train, scores = [], []
    
#     dict_formulae_test = df['formula'].apply(_element_composition)
#     dict_formulae_train = formulae_train.apply(_element_composition)

#     for elem in freq_df['test_elems']:
        
#         good_idxs = [i for i,formula_dict in enumerate(dict_formulae_test)
#                      if elem in formula_dict.keys()]
        
#         temp = df.iloc[good_idxs]
#         score = tasks.score_evaluation(temp['targets'], temp['preds'], metric)
        
#         good_idxs = [i for i,formula_dict in enumerate(dict_formulae_train)
#                      if elem in formula_dict.keys()]
        
#         count_train.append(len(good_idxs))
#         scores.append(score)
                
#     freq_df[f'{metric}'] = scores
#     freq_df['count_train'] = count_train
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x= freq_df['count_train'],
#                              y= freq_df[f'{metric}'],
#                              mode='markers',
#                              name='',
#                              marker=dict(size=10),
#                              hovertext=freq_df['test_elems']
#                              ))
#     fig.update_xaxes(title='occurrences in train')
#     fig.update_yaxes(title='MAE')
#     fig.update_layout(title='mae vs occurrences plot')
#     # if web: fig.write_html('figure.html', auto_open=True)
#     # fig.show()   
    
#     return freq_df


def plot_elem_class_score_matplotlib(freq_df, task, metric, prop, web=True):
    
    plt.rcParams['font.size'] = 16
    
    fig, ax = plt.subplots(figsize=(14,8))
    
    ax.scatter(freq_df['occ_train'],
               freq_df[f'{task}_{metric}'],
               edgecolor='k',
               linewidth=0.5)
    
    lower_error = freq_df[f'{task}_{metric}'] - freq_df[f'{task}_{metric}_std']
    lower_error = lower_error.fillna(0)
    
    upper_error = freq_df[f'{task}_{metric}'] + freq_df[f'{task}_{metric}_std']
    upper_error = upper_error.fillna(0)
    
    ax.errorbar(freq_df['occ_train'], 
                freq_df[f'{task}_{metric}'], 
                yerr=freq_df[f'{task}_{metric}_std'],
                fmt='none', 
                ecolor='gray', 
                elinewidth=1, 
                alpha=0.4, 
                capsize=2)
    
    # ax.fill_between(freq_df['occ_train'],lower_error, upper_error)
    
    ax.set_xlabel('Train occurrences', labelpad=15)
    ax.set_ylabel('MAE', labelpad=15)

    # save the plot as a file in the plots/fig1 folder
    plt.savefig(f'plots/fig1_rf/{prop}_distinct.png', dpi=300)

    
        
    
def plot_elem_class_score(freq_df, task, metric, prop, web=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x= freq_df['occ_train'],
                             y= freq_df[f'{task}_{metric}'],
                             error_y=dict(
                                 type='data', # value of error bar given in data coordinates
                                 array=freq_df[f'{task}_{metric}_std'],
                                 visible=True),
                             error_x=dict(
                                 type='data', # value of error bar given in data coordinates
                                 array=freq_df['occ_train_std'],
                                 visible=True),
                             mode='markers',
                             name='',
                             marker=dict(size=10),
                             hovertext=list(freq_df.index)
                             ))
    fig.update_xaxes(title='occurrences in train')
    fig.update_yaxes(title='MAE')
    fig.update_layout(title='mae vs occurrences plot ')
    fig.add_annotation(
        x=0.8,
        y=1,
        xref="paper",
        yref="paper",
        text=f"{prop}",
        font=dict(
            family="Courier New, monospace",
            size=30,
            color="black"
            ),
        align="center",
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e"
        )
    # fig.update_xaxes(type="log")
    if web: fig.write_html('figure.html', auto_open=True)
    fig.show()       
    

def plot_super_histos(dfs, bins, prop, op1=0.65, op2=0.8, extraord=True):
    colors = ['blue', 'orange', 'green']
    text = ["extraord", '', '']    
    
    high = max([df['target'].max() for df in dfs.values()])
    low  = min([df['target'].min() for df in dfs.values()])
    size = (high-low)/bins
    fig = go.Figure()
    for i, (key, df) in enumerate(dfs.items()):
        fig.add_trace(go.Histogram(x=df['target'],
                                   xbins=dict(
                                       start = low,
                                       end   = high,
                                       size = size), 
                                   autobinx=False,
                                   opacity = op1,
                                   name = key,
                                   marker_color = colors[i]
                                   ))
        if extraord:
            bar = df[df[f'extraord|{key}']==1].iloc[-1]['target']
            fig.add_vline(x=bar, 
                          line_width=3, 
                          line_dash="dash", 
                          line_color=colors[i],
                          annotation_text=text[i], 
                          annotation_position="top right")
            
            fig.add_trace(go.Histogram(x=df[df[f'extraord|{key}']==1]['target'],
                                       xbins=dict(
                                           start = low,
                                           end   = high,
                                           size = size), 
                                       autobinx=False,
                                       opacity = op2,
                                       showlegend=False,
                                       marker_color = colors[i]
                                       ))
        # fig.add_vrect(x0=bar, x1=high+10, 
        #       annotation_text=text[i], annotation_position='top right',
        #       fillcolor=colors[i], opacity=0.15, line_width=0)
        
    fig.update_layout(barmode='overlay')
    fig.update_xaxes(range=[low, high])  
    # if prop in ['bandgap']: 
        # fig.update_yaxes(range=[0,500])  
        # fig.update_yaxes(type="log")
    fig.update_layout(title=prop, font=dict(size=20))
    
    
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=1.2,
        xanchor="right",
        x=1.02,
        font=dict(size=30)
    ))
    
    fig.show() 
    if extraord: title = f'plots/datasets/{prop}_extraord.png'
    else:        title = f'plots/datasets/{prop}.png'
    pio.write_image(fig, title, width=4*300, height=4*300, scale=1)


def plot_parity(truth, pred, score):
    fig = go.Figure()
    truth = np.array(truth)
    pred = np.array(pred)
    fig.add_trace(go.Scatter(x=truth,
                             y=pred,
                             mode='markers',
                             name = 'test',
                             ))
    
    fig.add_trace(go.Scatter(x=[max(min(truth), min(pred))-1, min(max(truth), max(pred))+1],
                             y=[max(min(truth), min(pred))-1, min(max(truth), max(pred))+1],
                             mode='lines',
                             line=dict(dash='dash', color='orange', width = 2),
                             showlegend=False,
                             ))
    
    fig.update_xaxes(range=[min(truth)-1, max(truth)+1], title='ground truth')
    fig.update_yaxes(range=[min(pred)-1, max(pred)+1], title='predictions')
    fig.update_layout(title=f'parity plot with MAE = {round(score,3)}')
    fig.show()     

def plot_umap(one_dataset, split):
    # plot original
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=one_dataset[:split,0],
                             y=one_dataset[:split,1],
                             mode='markers',
                             marker_color='blue',
                             name = 'acceptor',
                             opacity=0.5,
                             ))
    fig.add_trace(go.Scatter(x=one_dataset[split:,0],
                             y=one_dataset[split:,1],
                             mode='markers',
                             marker_color='red',
                             name = 'donor (mpds)',
                             opacity=0.5,
                             ))    
    fig.show()

def plot_umap_augmentation(datasets_list, random_state=1234):
    colors = ['#ffffcc',
              '#ffeda0',
              '#fed976',
              '#feb24c',
              '#fd8d3c',
              '#fc4e2a',
              '#e31a1c',
              '#bd0026',
              '#800026',
              '#993404',
              '#662506',
              '#41ab5d',
            ]
    colors = list(reversed(colors))
    first = datasets_list[0]
    last = datasets_list[-1]
    
    #compute distance matrix with all
    mapper = ElM2D(verbose=False)
    mapper.fit(last['formula'])
    dm = mapper.dm #distance matrix.
    # UMAP of all
    umap_trans = umap.UMAP(
        densmap=True,
        output_dens=True,
        dens_lambda=1.0,
        n_neighbors=10,
        min_dist=0,
        n_components=2,
        metric="precomputed",
        random_state=random_state,
        low_memory=False,
    ).fit(dm)
    umap_emb = attrgetter("embedding_")(umap_trans)

    # extract stuff for first dataset
    N_i = len(first)
    umap_first = umap_emb[:N_i,:]

    # plot original
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=umap_first[:,0],
                             y=umap_first[:,1],
                             mode='markers',
                             marker_color=colors[0],
                             name = 'orginal',
                             ))
    # plot the others
    for i, dataset in enumerate(datasets_list):
        # extract stuff for first dataset
        N_block = len(dataset) - N_i
        umap_i = umap_emb[N_i:N_i+N_block,:]
    
        # plot original
        fig.add_trace(go.Scatter(x=umap_i[:,0],
                                  y=umap_i[:,1],
                                  mode='markers',
                                  marker_color=colors[i+1],
                                  name = f'iteration {i+1}',
                                  ))     
        N_i = len(dataset)
        
    fig.show()

def periodic_table(train_list):
    
    before = train_list[0]    
    for i in range(len(train_list)-1):
        
        after = train_list[i+1].drop(index=train_list[0].index)
        
        ptable_heatmap(before['formula'],
                       after['formula'], 
                       cbar_title=f'El. count (Step {i+1})')

# def plot_augmentation(outs, train_list, test_key, task):
#     # extract accuracies from outs dictionary
#     l=[]
#     for out in outs:
#         if 'regression' in task:
#             l.append(out[task]['mae'])
#         if 'classification' in task:
#             l.append(out[task]['acc'])
            
#     fig = go.Figure()
#     N_first = len(train_list[0])
#     Ns = np.array([len(dataset) for dataset in train_list])
#     fig.add_trace(go.Scatter(x=Ns-N_first,
#                              y=l,
#                              mode='markers+lines',
#                              marker=dict(size=8),
#                              line=dict(width=1),
#                              marker_color='blue',
#                              name = 'accuracy',
#                              ))
    
    
#     fig.update_xaxes(title='N additional points')
#     fig.update_yaxes(title=f'accuracy on test ({test_key})')
#     fig.update_layout(title=f'accuracy plot for {test_key}')
#     fig.show() 
 
class plot_augmentation():    
    def __init__(self, test_key, task, prop):
        self.test_key = test_key
        self.task     = task
        self.prop     = prop
        
    def load_disco(self, outs_disco, train_list_disco):
        self.outs_disco         = outs_disco
        self.train_list_disco   = train_list_disco
        
    def load_rnd(self, outs_rnd, train_list_rnd):
        self.outs_rnd           = outs_rnd
        self.train_list_rnd     = train_list_rnd
        
    def plot_equitability(self):
        
        train_list_rnd = self.train_list_rnd
        train_list_disco = self.train_list_disco
        
        equit_indices_disco = []
        equit_indices_rnd   = []
        
        for df_rnd, df_disco in zip(train_list_rnd, train_list_disco):
            
            equit_df_rnd =  equitability_index(df_rnd)
            equit_df_disco =  equitability_index(df_disco)
            
            equit_indices_rnd.append(equit_df_rnd)
            equit_indices_disco.append(equit_df_disco)
            
        N_first = len(train_list_disco[0])
        Ns_disco = np.array([len(dataset) for dataset in train_list_disco])
        Ns_rnd = np.array([len(dataset) for dataset in train_list_rnd])
            
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=Ns_disco - N_first,
                                 y=equit_indices_disco,
                                 mode='markers+lines',
                                 marker=dict(size=8),
                                 line=dict(width=1),
                                 marker_color='blue',
                                 name = 'discover',
                                 ))  
        
        fig.add_trace(go.Scatter(x=Ns_rnd - N_first,
                                 y=equit_indices_rnd,
                                 mode='markers+lines',
                                 marker=dict(size=8),
                                 line=dict(width=1),
                                 marker_color='red',
                                 name = 'rnd',
                                 ))
        
        fig.update_xaxes(title='N additional points')
        fig.update_yaxes(title=f'Equitability on train ({self.test_key})')
        fig.update_layout(title=f'Equitability plot for {self.test_key} ({self.prop})')
        fig.show()
        
    def plot(self):
        outs_disco       = self.outs_disco
        outs_rnd         = self.outs_rnd
        train_list_rnd   = self.train_list_rnd
        train_list_disco = self.train_list_disco
        # extract accuracies from outs dictionary
        l_disco=[]
        for out in outs_disco:
            if 'regression' in self.task:
                l_disco.append(out[self.task]['mae'])
            if 'classification' in self.task:
                l_disco.append(out[self.task]['acc'])
    
        l_rnd=[]
        for out in outs_rnd:
            if 'regression' in self.task:
                l_rnd.append(out[self.task]['mae'])
            if 'classification' in self.task:
                l_rnd.append(out[self.task]['acc'])
                
        fig = go.Figure()
        N_first = len(train_list_disco[0])
        
        Ns_disco = np.array([len(dataset) for dataset in train_list_disco])
        Ns_rnd = np.array([len(dataset) for dataset in train_list_rnd])
        
        fig.add_trace(go.Scatter(x=Ns_disco-N_first,
                                 y=l_disco,
                                 mode='markers+lines',
                                 marker=dict(size=8),
                                 line=dict(width=1),
                                 marker_color='blue',
                                 name = 'discover',
                                 ))
        
        fig.add_trace(go.Scatter(x=Ns_rnd-N_first,
                                 y=l_rnd,
                                 mode='markers+lines',
                                 marker=dict(size=8),
                                 line=dict(width=1),
                                 marker_color='red',
                                 name = 'random',
                                 ))    
        
        fig.update_xaxes(title='N additional points')
        fig.update_yaxes(title=f'accuracy on test ({self.test_key})')
        fig.update_layout(title=f'accuracy plot for {self.test_key} ({self.prop})')
        fig.show()
        









        
