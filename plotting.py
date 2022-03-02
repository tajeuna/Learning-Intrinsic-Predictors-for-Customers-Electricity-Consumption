#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 10:31:28 2022

@author: etienne
"""



from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



class view:
    
    def __init__(self, original, frame=None, graph=None, clusters = None, embedding=None):
        
        self.frame = frame
        self.graph = graph
        self.clusters = clusters
        self.original = original
        self.embedding = embedding
        
        
    def embeddings(self,method='pca'):
        mat = self.embedding.cpu().detach().numpy()
        
        model_view = None
        if method == 'pca':
            model_view = PCA(n_components=2)
        elif method == 'tsne':
            model_view = TSNE(n_components=2)
            
        data = model_view.fit_transform(mat)
            
        fig = plt.figure(figsize=(8, 8), dpi=175)
        axe = fig.add_subplot(111)
        
        
        for i in range(len(mat)):
            n = list(self.original)[i]
            axe.plot(data[i, 0], data[i, 1], '*', color=self.graph.nodes[n]['color'])
                
        axe.set_title(method+' projection')
        plt.show()
        
    def series(self, series_name, range_=None):
        fig = plt.figure(figsize=(12,8),dpi=90)
        axe = fig.add_subplot(111)
        if range_ is None:
            self.frame[series_name].plot(ax=axe)
        else:
            self.frame.loc[range_, series_name].plot(ax=axe)
        axe.grid(True, which='both',axis='both', animated=True)
        plt.show()
        
    def clusters_(self):
        def get_representative(cluster):
            degree = dict(self.graph.degree(cluster))
            rep = []
            for n, deg in degree.items():
                if deg == max(degree.values()):
                    rep.append(n)
            return rep
        
        
        if len(self.clusters) == 1:
            rep = get_representative(self.clusters[0])
            fig = plt.figure(figsize=(12, 8), dpi=175)
            axe = fig.add_subplot(111)
            self.original.loc[self.frame.index, self.clusters[0]].plot(color='k',lw=.85,alpha=.5, ax=axe,legend=False)
            self.original.loc[self.frame.index, rep].mean(axis=1).plot(color='r', lw=1.5,alpha=1,ax=axe,legend=False)
            axe.grid(True, which='both',axis='both', animated=True)
            axe.set_title('Cluster with '+str(len(self.clusters[0]))+' time series')
            plt.show()
        else:
            if len(self.clusters) % 2 == 0:
                if len(self.clusters) / 2 == 1:
                    fig = plt.figure(figsize=(12, 8), dpi=175)
                    for i in range(len(self.clusters)):
                        rep = get_representative(self.clusters[i])
                        axe = fig.add_subplot(1, 2, i+1)
                        self.original.loc[self.frame.index, self.clusters[i]].plot(color = self.graph.nodes[self.clusters[i][0]]['color'],lw=.85,alpha=.5, ax=axe,legend=False)
                        self.original.loc[self.frame.index, rep].mean(axis=1).plot(color = 'k', lw=1.5,alpha=1, ax=axe,legend=False)
                        axe.grid(True, which='both',axis='both', animated=True)
                        axe.set_title('Cluster with '+str(len(self.clusters[i]))+' time series')
                    plt.show()
                else:
                    fig = plt.figure(figsize=(12, 12), dpi=175)
                    for i in range(len(self.clusters)):
                        rep = get_representative(self.clusters[i])
                        axe = fig.add_subplot(int(len(self.clusters) / 2), 2, i+1)
                        self.original.loc[self.frame.index, self.clusters[i]].plot(color = self.graph.nodes[self.clusters[i][0]]['color'],lw=.85,alpha=.5, ax=axe,legend=False)
                        self.original.loc[self.frame.index, rep].mean(axis=1).plot(color = 'k', lw=1.5,alpha=1, ax=axe,legend=False)
                        axe.grid(True, which='both',axis='both', animated=True)
                        axe.set_title('Cluster with '+str(len(self.clusters[i]))+' time series')
                    plt.show()
                    