#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 09:14:25 2022

@author: etienne
"""



# import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler

class series_import:
    
    
    def __init__(self, path_to_csv_file = None, scale=True):
        
        self.path_to_csv_file = path_to_csv_file
        self.data = pd.read_csv(path_to_csv_file, index_col=0)
        self.evolving_index = pd.to_datetime(self.data.index)
        self.series_name = ['S'+str(i+1) for i in range(len(list(self.data)))]
        self.data.columns = self.series_name
        if scale:
            X = self.data.values
            self.data = pd.DataFrame(MinMaxScaler().fit_transform(X))
            self.data.columns = self.series_name
            self.data.index = self.evolving_index
        
        
    
        