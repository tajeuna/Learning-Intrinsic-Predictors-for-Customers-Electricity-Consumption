#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 07:34:07 2022

@author: etienne
"""


import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from itertools import combinations
from matplotlib import colors as mcolors


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

color_names = [name for name, color in colors.items()]
# by_hsv = (tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
#                 for name, color in colors.items()
# sorted_names = [name for hsv, name in by_hsv]




class graph_representation:
    
    
    def __init__(self, original, memberships, period='month', clustering_method = 'Markov'):
        
        # period can be: day, week, month, term, semester, year
        
        self.memberships = memberships
        self.period = period
        self.clustering_method = clustering_method
        self.original = original
        
        
    def split_series_members(self):
        
        frames = []
        if self.period == 'day':
            for fr in self.memberships.resample('1D'):
                frames.append(fr[1])
                
        elif self.period == 'week':
            for fr in self.memberships.resample('1W'):
                frames.append(fr[1])
                
        elif self.period == 'month':
            for fr in self.memberships.resample('1M'):
                frames.append(fr[1])
                
        elif self.period == 'term':
            for fr in self.memberships.resample('3M'):
                frames.append(fr[1])
                
        elif self.period == 'semester':
            for fr in self.memberships.resample('6M'):
                frames.append(fr[1])
                
        elif self.period == 'year':
            for fr in self.memberships.resample('1Y'):
                frames.append(fr[1])
                
        
        return frames
        
        
    def split_series_members_in_one(self):
        
        frames = []
        headers = []
        
        if self.period == 'day':
            for fr in self.memberships.resample('1D'):
                frames.append(fr[1])
        elif self.period == 'week':
            for fr in self.memberships.resample('1W'):
                test = [sb[1] for sb in fr[1].resample('1D')]
                if len(test) == 7:
                    frames.append(fr[1])
        elif self.period == 'month':
            for fr in self.memberships.resample('1M'):
                test = [sb[1] for sb in fr[1].resample('1D')]
                if len(test) >= 28:
                    frames.append(pd.concat(test[0:28]))
        elif self.period == 'term':
            for fr in self.memberships.resample('3M'):
                toconcat = []
                test1 = [sb[1] for sb in fr[1].resample('1M')]
                if len(test1) == 3:
                    for fr2 in test1:
                       test2 = [sb[1] for sb in fr2.resample('1D')]
                       if len(test2) >= 28:
                            toconcat.append(pd.concat(test2[0:28])) 
                    frames.append(pd.concat(toconcat))
        elif self.period == 'semester':
            for fr in self.memberships.resample('6M'):
                toconcat = []
                test1 = [sb[1] for sb in fr[1].resample('1M')]
                if len(test1) == 6:
                    for fr2 in test1:
                       test2 = [sb[1] for sb in fr2.resample('1D')]
                       if len(test2) >= 28:
                            toconcat.append(pd.concat(test2[0:28])) 
                    frames.append(pd.concat(toconcat))
        elif self.period == 'year':
            frames = [fr[1] for fr in self.memberships.resample('1Y')]
        
        arrays = []
        for i in range(len(frames)):
            fr = frames[i]
            header = [s+'_'+str(i+1) for s in list(fr)]
            headers.extend(header)
            arrays.append(fr.values)
            
        
        
        return headers, np.concatenate(arrays, axis=1)
    
    
    def get_graph(self, frame):
        mat = frame.values
        headers = list(frame)
        weights = np.dot(mat.T, mat)
        thresh = weights[0,0]/2
        g, g2 = None, None
        final_g, final_g2 = None, None
        edges = []
        w_attr = []
        print('Size of series '+str(weights[0,0])+' testing with threshold ')
        while True:
            print(thresh, end=',')
            g = nx.Graph()
            g2 = ig.Graph()
            g.add_nodes_from(list(frame))
            g2.add_vertices(np.arange(frame.shape[1]))
            
            for n1, n2 in combinations(headers, 2):
                i, j = headers.index(n1), headers.index(n2)
                if weights[i, j] > thresh:
                    g.add_edge(n1, n2)
                    g[n1][n2]['weight'] = weights[i, j]
                    g2.add_edges([(i,j)])
                    w_attr.append(weights[i, j])
            g.add_edges_from(edges)
            if nx.is_connected(g):
                thresh += 1
                final_g, final_g2 = g, g2
            else:
                if final_g is None:
                    thresh -= 1
                    edges = []
                    w_attr = []
                else:
                    break
        clusters = []
        if self.clustering_method == 'Markov':
            for cl in final_g2.community_walktrap(weights=w_attr).as_clustering():
                clusters.append([list(frame)[el] for el in cl])
        if self.clustering_method == 'Infomax':
            for cl in final_g2.community_infomap(w_attr):
                clusters.append([list(frame)[el] for el in cl])
                
        
        print('graph build with '+str(thresh)+' threshold')
        print(nx.info(final_g))
        
        for i in range(len(clusters)):
            cluster = clusters[i]
            for n in cluster:
                final_g.nodes[n]['class'] = 'C'+str(i+1)
                final_g.nodes[n]['color'] = color_names[i]
                final_g.nodes[n]['features'] = self.original.loc[frame.index, [n]].values.flatten()
        return final_g, clusters, thresh
    
    def track_patterns(self, all_graphs, all_clusters):
        def get_representative(graph, cluster):
            degree = dict(graph.degree(cluster))
            rep = []
            for n, deg in degree.items():
                if deg == max(degree.values()):
                    rep.append(n)
            return rep
        representatives = []
        for g, clusters in zip(all_graphs, all_clusters):
            tmp = []
            for cluster in clusters:
                tmp.append(get_representative(g, cluster))
            representatives.append(tmp)
        return representatives
                