import numpy as np
import pandas as pd

class decomposition:


    def __init__(self, co_evolving_timeseries, components=10):

        self.co_evolving_timeseries = co_evolving_timeseries
        self.intervals = {}
        self.components = components
        min_val =0
        cpt = 1
        while min_val < 1.:
            self.intervals['g'+str(cpt)] = [min_val, min_val+(1./components)]
            min_val += 1./components
            cpt += 1
        
    def categorize_series(self, series):
        sequence = []
        for val in series:
            placed = False
            for k, interval in self.intervals.items():
                if min(interval) <= val < max(interval):
                        sequence.append(k)
                        placed = True
            if placed == False:
                sequence.append('g'+str(self.components))
        return sequence
    
    def categorize_co_evolving_series(self):
        header = list(self.co_evolving_timeseries)
        
        frame_label = self.co_evolving_timeseries.copy()
        for s in header: 
            frame_label[s] = self.categorize_series(self.co_evolving_timeseries[s])
        
        
        return frame_label

    def get_membership_matrix(self,frame_label):

        n = frame_label.shape[0]*self.components
        m = frame_label.shape[1]
        headers = list(frame_label)
        labels = ['g'+str(cpt) for cpt in range(1, self.components+1)]
        matrix = np.zeros((n,m))
        instant = []
        for i in range(len(frame_label)):
            instant.extend([i+1]*self.components)

        for h in list(frame_label):
            s = frame_label[h].values
            step = 0
            for i in range(len(s)):
                posi = labels.index(s[i])+step
                matrix[posi][headers.index(h)] = 1

                step += self.components

        members = pd.DataFrame(matrix)
        members.columns = headers
        members['instant'] = [self.co_evolving_timeseries.index[ind-1] for ind in instant]
        members.set_index('instant', inplace=True)
        return members
    
    
                                   
        



