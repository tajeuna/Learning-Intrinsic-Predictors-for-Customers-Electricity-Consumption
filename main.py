#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 09:32:13 2022

@author: etienne
"""


from series_loader import series_import
from plotting import view
from series_decomposition import decomposition
from pattern_identification import graph_representation
from graph_properties import predictor_neural_net as pnn
import torch as th
import dgl
import numpy as np
import json
import os


def tensor_graph(nx_graph, frame):
        nodes = list(frame)
        source, destination = [], []
        for u, v in nx_graph.edges():
            source.append(nodes.index(u))
            destination.append(nodes.index(v))
        source = th.tensor(source)
        destination = th.tensor(destination)
        g = dgl.graph((source, destination)).to(device)
        features = np.zeros((len(nodes), len(frame)))
        for i in range(len(nodes)):
            features[i, :] = frame[nodes[i]].values.flatten()
        g.ndata['features'] = th.from_numpy(features).to(device, dtype)
        
        
        return dgl.add_self_loop(g).to(device)
    
def save_graph(nx_graph, time_interval, name_file, path_to_save, threshold_used):
    dico = {}
    dico['threshold'] = threshold_used
    dico['source'] = []
    dico['target'] = []
    dico['time_interval'] = time_interval
    dico['nodes'] = list(nx_graph.nodes())
    dico['class'] = []
    [dico['class'].append(nx_graph.nodes[n]['class']) for n in nx_graph.nodes()]
    dico['features'] = []
    [dico['features'].append(nx_graph.nodes[n]['features'].tolist()) for n in nx_graph.nodes()]
    dico['color'] = []
    [dico['color'].append(nx_graph.nodes[n]['color']) for n in nx_graph.nodes()]
    for u, v in nx_graph.edges():
        dico['source'].append(u)
        dico['target'].append(v)
    with open(path_to_save+name_file, 'w') as jasonfile:
        json.dump(dico, jasonfile, indent=4)
        
data_name = 'apartment.csv'
# data_name = 'ELF_data.csv'
# data_name = 'enernoc.csv'
# data_name = 'gps_data.csv'
# data_name = 'passengers.csv'
# data_name = 'departement_1_emails.csv'
# data_name = 'departement_2_emails.csv'
# data_name = 'trips.csv'
path_to_csv_file = None
path_to_graph = None
data_directory = '/home/etienne/LIPMAC/Data sets/'
if data_name not in os.listdir(data_directory): print('  Data is not existing, nothing can be done ...')
else: path_to_csv_file = data_directory+data_name
if data_name.split('.')[0] not in os.listdir(data_directory+'stored_graph/'):
    os.mkdir(data_directory+'stored_graph/'+data_name.split('.')[0])
path_to_graph = data_directory+'stored_graph/'+data_name.split('.')[0]+'/'


period = 'day'
if path_to_graph+period not in os.listdir(path_to_graph):
    os.mkdir(path_to_graph+period)
path_to_graph_period = path_to_graph+period+'/'

clustering_method = 'Markov'
if path_to_graph_period+clustering_method not in os.listdir(path_to_graph_period):
    os.mkdir(path_to_graph_period+clustering_method)
path_to_graph_period_clustering_method = path_to_graph_period+clustering_method+'/'

nbre_component = 5
if path_to_graph_period_clustering_method+str(nbre_component) not in os.listdir(path_to_graph_period_clustering_method):
    os.mkdir(path_to_graph_period_clustering_method+str(nbre_component))
path_to_graph_period_clustering_method_nbre_component = path_to_graph_period_clustering_method+str(nbre_component)+'/'
hidden_layers_pred = [24, 8]
hidden_layers_reg = [8, 24]


device = 'cpu'
dtype = th.double




if __name__ == '__main__':
    
    load_series = series_import(path_to_csv_file = path_to_csv_file)
    data = load_series.data
    time = load_series.evolving_index
    
    categorizer = decomposition(data,components=nbre_component)
    
    # sequences = categorizer.categorize_series(list(data['S1'].values))
    all_sequences = categorizer.categorize_co_evolving_series()
    members = categorizer.get_membership_matrix(all_sequences)
    
    graph_rep = graph_representation(data, members, period=period, clustering_method = clustering_method)
    frames = graph_rep.split_series_members()
    all_graphs, all_clusters, all_thresholds = [], [], []
    interval = 1
    if len(os.listdir(path_to_graph_period_clustering_method_nbre_component)) == 0:
        for frame in frames:
            final_g, clusters, thresh = graph_rep.get_graph(frame)
            all_graphs.append(final_g)
            all_clusters.append(clusters)
            all_thresholds.append(thresh)
            name_file = data_name.split('.')[0]+'_graph_'+str(interval)+'.json'
            save_graph(final_g, 
                       'T'+str(interval), 
                       name_file, 
                       path_to_graph_period_clustering_method_nbre_component,
                       thresh)
            interval += 1
    
#     representatives = graph_rep.track_patterns(all_graphs, all_clusters)
    
    
    
    # properties = pnn(frame = data.loc[frames[0].index, :], 
    #                  output_ = 5, 
    #                  predictor_layers=hidden_layers_pred,
    #                  regressor_layers= hidden_layers_reg)
    
    
    
    # tn_g = tensor_graph(final_g, data.loc[frames[0].index, :])
    
    
    
    # h, r = properties(tn_g, tn_g.ndata['features'])
    
    # visualize = view(data, frame=frames[0], graph=final_g, clusters=clusters,embedding=h)
    # visualize.clusters_()
    # visualize.embeddings()
    