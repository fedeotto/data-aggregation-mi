import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.io as pio

from assets.plots import plot_super_histos, plot_distinct_histos, add_prop_to_violins, plot_violins
from assets.utils import load_dataset, print_info
from assets.preprocessing import preprocess_dataset, add_column
from settings import *
import kaleido

pio.renderers.default="svg"    # 'svg' or 'browser'
pio.templates.default="simple_white"

props_list = [  
                'rho',
                'sigma',
                'thermalcond',
                # 'superconT',
                'seebeck',
                'bandgap',
                'bulkmodulus',
                'shearmodulus'                
              ]

# settings imported from settings.py
def plot_all():    

    # violin plot
    violin_fig = make_subplots(rows=2, cols=4, 
                               start_cell="top-left",
                               horizontal_spacing=0.07,
                               vertical_spacing=0.15)
    l=[]

    # main loop
    for i, prop in enumerate(props_list):
        
        freq_df_complete = pd.DataFrame()        
        """LOADING"""
        # load datsets
        data_raw = load_dataset(prop)  
        keys_list = list(data_raw.keys())
        key_A = pairs[prop][0]; assert key_A != 'mpds'
        key_B = pairs[prop][1]
        data_raw = {key_A:data_raw[key_A], key_B:data_raw[key_B]}
        print_info(data_raw, prop)
        print('')
        
        """PREPROCESSING"""
        # preprocessing
        data_clean = preprocess_dataset(data_raw, prop, merging,
                                        epsilon_T, 
                                        med_sigma_multiplier,
                                        mult_outliers,
                                        ascending_setting[prop]) 
        print('')
        print_info(data_clean, prop)
        # plot_super_histos(data_clean, 60, prop, op1=0.6, extraord=False)
        
        violin_fig = add_prop_to_violins(violin_fig, i, data_clean, prop, l)
        # add extraord column to all datasets(0/1)
        # data_clean = add_column(data_clean, extraord_size, ascending_setting[prop])
        # plot_distinct_histos(data_clean, 60, prop, extraord=True)
        # plots.plot_super_histos(data_clean, 60, prop, op1=0.65, op2=0.8, extraord=True)
    
    # Set the target DPI (dots per inch)
    target_dpi = 600

    # Calculate the scale factor based on the target DPI
    scale_factor = target_dpi / 96.0  # 96 is the default DPI in Plotly

    fig = plot_violins(violin_fig)
    pio.write_image(fig, 'plots/fig2/violin.png', scale=scale_factor)


if __name__ == '__main__': plot_all()
