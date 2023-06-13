import numpy as np
import pandas as pd
import torch
import os
import re
import json
import itertools
from chem import _fractional_composition

from torch_geometric.data import Data

# The task: write dataset class (as a list of Data objects) and a Dataloader
def proper_split(se):
    split=[]
    for i,symbol in enumerate(se):
        if(symbol=='[' or symbol==']'):
            split.append(symbol)
        elif(re.search('[A-Z]',symbol)):
            if(i<len(se)-1):
                if(re.search('[a-z]',se[i+1])):
                    split.append(se[i:i+2])
                else:
                    split.append(symbol)
            else:
                split.append(symbol)
        elif(re.search('[0-9]',symbol)):
            split.append(symbol)
        elif(re.search('\.',symbol)):
            split.append(symbol)
    return split

def join_numbers(s):
    join=np.zeros(len(s)-1)
    for i,symbol in enumerate(s[:-1]):
        if(re.search('[0-9]',symbol)):
            if(re.search('[0-9]',s[i+1])):
                join[i]=1
            elif(re.search('\.',s[i+1])):
                join[i]=1
        elif(re.search('\.',symbol)):
            if(re.search('[0-9]',s[i+1])):
                join[i]=1
    left=[]
    right=[]
    for i in range(len(join)-1):
        if(join[i]==0 and join[i+1]==1):
            left.append(i)
        elif(join[i]==1 and join[i+1]==0):
            right.append(i)
    if(len(right)<len(left)):
        right.append(len(s)-1)
    lenth=0
    for i in range(len(left)):
        s[left[i]+1-lenth:right[i]+2-lenth]=[''.join(s[left[i]+1-lenth:right[i]+2-lenth])]      
        lenth+=right[i]-left[i]
    
    return s

def composition_from_formula(s):
    a=proper_split(s)
    b=join_numbers(a)
    
    brakets=[1]
    extructed=[]
    for symbol in b:
        if(symbol=='[' or symbol==']'):
             extructed.append(symbol)
    for i in range(1,len(extructed)):
        if(extructed[i]=='[' and extructed[i-1]=='['):
            brakets.append(brakets[-1]+1)
        elif(extructed[i]=='[' and extructed[i-1]==']'):
            brakets.append(brakets[-1])
        elif(extructed[i]==']' and extructed[i-1]=='['):
            brakets.append(brakets[-1])
        elif(extructed[i]==']' and extructed[i-1]==']'):
            brakets.append(brakets[-1]-1)
            
    i=0
    for j,elem in enumerate(b):
        if(elem=='[' or elem==']'):
            b[j]=b[j]+str(brakets[i])
            i+=1
    
    n=max(brakets)
    braket_layers={}
    for order in range(1,n+1):
        left=[]
        right=[]
        for i,symi in enumerate(b):
            if(symi=='('+str(order)):
                left.append(i)
            elif(symi==')'+str(order)):
                right.append(i)
        pairs=[]
        for i in range(len(left)):
            pairs.append((left[i],right[i]))
        braket_layers[order]=pairs
        
    list_of_elements=[]
    
    for i,elem in enumerate(b):
        if(re.search('[A-Za-z]',elem)):
            if(i<len(b)-1):
                if(re.search('[0-9]',b[i+1]) and not re.search('\[|\]',b[i+1])):
                    num=float(b[i+1])
                else:
                    num=1
            else:
                num=1
            list_of_elements.append((elem,i,num))
        
            
    for order in range(1,n+1):
        for coord in braket_layers[order]:
            if(coord[1]<len(b)-1):
                if(re.search('[0-9]',b[coord[1]+1]) and not re.search('\[|\]',b[coord[1]+1])):
                    factor=float(b[coord[1]+1])
                else:
                    factor=1
            else:
                factor=1
            for k,elem in enumerate(list_of_elements):
                if(elem[1]>coord[0] and elem[1]<coord[1]):
                    new_value=elem[2]*factor
                    name=elem[0]
                    position=elem[1]
                    list_of_elements[k]=(name,position,new_value)
                    
                    
    names=[]
    for elem in list_of_elements:
        names.append(elem[0])
    names=list(set(names))
    
    composition={}
    for name in names:
        composition[name]=0
        for elem in list_of_elements:
            if(elem[0]==name):
                composition[name]+=elem[2]  
                
    norm=np.sum(list(composition.values()))
    if(norm>0):
        for key in composition.keys():
            composition[key]=composition[key]/norm          
                
    return composition


def data_from_composition(df,elem_prop='mat2vec'):
    normalised_compositions=[]
    elem_features = pd.read_csv(f'element_properties/{elem_prop}.csv',
                                index_col='element')
        
    for i,form in enumerate(df['formula'].values):
        comp= _fractional_composition(form)
        # comp=composition_from_formula(form)
        normalised_compositions.append(comp)

    df['comp_dict']=normalised_compositions

    data_list=[] 
    for i in range(len(df)):
        a=df.iloc[i]
        y=torch.Tensor([a['target']])
        x_init=[]
        nodes=[]
        weights=[]
        for j,key in enumerate(a['comp_dict'].keys()):
            x_init.append(elem_features.loc[key])
            weights.append(a['comp_dict'][key])
            nodes.append(j)
        x=torch.Tensor(x_init)
        weights=torch.Tensor(weights)
        edge_index=[]
        for j in nodes:
            for k in nodes:
                edge_index.append([j,k])
        edge_index=torch.tensor(edge_index,dtype=torch.long)
        data_list.append(Data(x=x,edge_index=edge_index.t().contiguous(),pos=weights,y=y))   
    return data_list



