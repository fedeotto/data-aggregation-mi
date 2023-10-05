import numpy as np
import pandas as pd
from assets.chem import _element_composition_L, _element_composition
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy as Entropy
from collections import Counter

def equitability_index(df: pd.DataFrame):
    
    train_dicts = df['formula'].apply(_element_composition)
    train_list = [item for row in train_dicts for item in row.keys()]
    train_counter = Counter(train_list)
    trainc_df = pd.DataFrame.from_dict(train_counter, orient='index', columns=['count'])
    count_col = trainc_df['count']
    
    diversity = Entropy(count_col, base=2) / (np.log2(len(count_col)))
    
    return diversity
    

def discovery_yield(pred, test, target_range):
    """ calculate the discovery yield, i.e. the percentage of examples 
    correctly predicted in the target range
    
    Parameters:
        pred (arrray N_test):    predicted values
        test (array N_test):     testing values
        target_range (list 2):   property interval, i.e. [minimum, maximum]
    
    Returns:
        accuracy (float): % of examples correctly predicted in the target_range
    """
    inf, sup = target_range
    
    target_i = np.where(test<sup & test>inf)
    correct=0
    for i in target_i:
        if (pred[i]<sup and pred[i]>inf): correct+=1
    return correct/len(target_i)    
        
    
    
def plot_discovery_yield(_pred, _test, target_range):
    """ plot the 2D predictions plot. Hoping to identify a distinct region 
    in the top lright corner
    
    Parameters:
        pred (arrray N_test):    predicted values
        test (array N_test):     testing values
        target_range (list 2):   property interval, i.e. [minimum, maximum]
    """    
    inf, sup = target_range
    
    test = pd.Series(_test.copy())
    test_sorted = test.sort_values()
    indices = test_sorted.index.to_list()
    pred_sorted = [_pred[i] for i in indices]
    
    fig = plt.figure(figsize=[12,6], dpi=200)
    plt.scatter(test_sorted, pred_sorted)
    plt.axvline(inf)
    plt.axvline(sup)
    plt.axhline(inf)
    plt.axhline(sup)
    
    return 