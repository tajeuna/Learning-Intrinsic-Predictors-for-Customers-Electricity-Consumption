#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 07:52:26 2022

@author: etienne
"""

# import pandas as pd
import torch as th
from torch import nn
from dgl.nn import GraphConv 
import torch.nn.functional as F

# device = 'cuda:0'
device = 'cpu'
dtype = th.double


class predictor_neural_net(nn.Module):
    
    
    def __init__(self, frame, output_, predictor_layers, regressor_layers):
        
        
        self.output_ = output_
        self.frame = frame
        self.predictor_layers = predictor_layers
        self.in_feats = len(frame)
        self.regressor_layers = regressor_layers
        super(predictor_neural_net, self).__init__()
        
        # Encoder net
        
        self.conv_layers = nn.ModuleList()

        self.conv_layers.append(
            GraphConv(len(frame), predictor_layers[0], allow_zero_in_degree=True).to(device, dtype)
            )
        
        for i in range(1, len(predictor_layers)):
            self.conv_layers.append(
                GraphConv(predictor_layers[i-1],
                          predictor_layers[i], 
                          allow_zero_in_degree=True).to(device, dtype)
                )
            

        self.conv_layers.append(
            GraphConv(predictor_layers[-1], 
                      output_, 
                      allow_zero_in_degree=True).to(device, dtype)
            )
        
        
        
        
        # Decoder net
        
        self.regres_layers = nn.Sequential()

        self.regres_layers.add_module(
            'first layer regressor',
            nn.Linear(output_, regressor_layers[0]).to(device, dtype)
            )
        self.regres_layers.add_module(
            'first regressor activation ', nn.Tanh().to(device, dtype)
            )

        for i in range(1, len(regressor_layers)):
            self.regres_layers.add_module(
                'regressor layer '+str(i+2),
                nn.Linear(regressor_layers[i-1], 
                          regressor_layers[i]).to(device, dtype)
                )
            self.regres_layers.add_module(
            'regressor activation '+str(i+2), nn.Tanh().to(device, dtype)
            )

        self.regres_layers.add_module(
            'latest layer regressor',
            nn.Linear(regressor_layers[-1], 
                      len(frame)).to(device, dtype)
            )
        self.regres_layers.add_module(
            'out activation',
            nn.Softmax().to(device, dtype)
            )
    
    def forward(self, g, in_feat):
        
        h = in_feat
        
        for i, layer in enumerate(self.conv_layers):
            h = layer(g, h)
            if i < len(self.conv_layers)-1:
                h = F.tanh(h)
            else:
                h = F.softmax(h, dim=1)
        r = self.regres_layers(h)

        

        return h, r
