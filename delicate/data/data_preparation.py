# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 12:01:14 2022

@author: Yi Yu
"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
import collections
from sklearn.model_selection import train_test_split

def data_load(dataset):
    
    dataset_file = dataset + '.csv'
    data = './data/' + dataset_file
    df1 = pd.read_csv(data)
    
    def clean(a):
        try:
            mol = Chem.MolFromSmiles(a)
            Chem.RemoveStereochemistry(mol)
            Chem.RemoveHs(mol)
            remover = SaltRemover()
            mol1 = remover.StripMol(mol)
            aa = Chem.MolToSmiles(mol1)
            return aa
        except:
            return ''
        
    print('number of molecules before cleaning',len(df1))
    df1['smilessd'] = df1['smiles'].map(clean)
    num = np.where(df1["smiles"] == df1["smilessd"], True, False)
    print('number of molecules smiles changed', collections.Counter(num))  
    df1 = df1.dropna()
    df1['smiles'] = df1['smilessd']
    df1 = df1.drop('smilessd', 1)
    df1.reset_index(level=0, inplace=True)
    print('number of molecules after cleaning',len(df1))
    
    X = df1['index'].to_numpy()
    y = df1['label'].to_numpy()
    
    if df1['label'].isin([0,1]).all() == False:
    
        n_splits = 5
        SKF = KFold(n_splits=5, shuffle=True, random_state=0)
        List1 = []
        List2 = []
                    
        for train_index, val_index in SKF.split(X):
            
            list_t = train_index.tolist()
            
            X_val,X_test,y_val,y_test = train_test_split(val_index, y[val_index],
                                                         test_size=0.5, random_state = 42)
            
            
            List_v_1 = X_val.tolist()
            
            
            List_v_2 = X_test.tolist()
            
            List1.append(list_t)
            List1.append(List_v_1)
            List1.append(List_v_2)
            
            List2.append(List1)
            List1 = []
            
        List3 = List2[:3]
            
    else:
        n_splits = 5
        SKF = StratifiedKFold(n_splits=n_splits, shuffle = True,random_state=0)
        List1 = []
        List2 = []
        for train_index, val_index in SKF.split(X,y):
            
            list_t = train_index.tolist()
            
            X_val,X_test,y_val,y_test = train_test_split(val_index, y[val_index],
                                                         test_size=0.5, random_state = 42)
            
            
            List_v_1 = X_val.tolist()
            
            
            List_v_2 = X_test.tolist()
            
            List1.append(list_t)
            List1.append(List_v_1)
            List1.append(List_v_2)
            
            List2.append(List1)
            List1 = []
            
        List3 = List2[:3]
   
    df1 = df1.drop(['index'],axis = 1)
    return df1,List3


