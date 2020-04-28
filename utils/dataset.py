#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd

from IPython.core.debugger import set_trace


# In[2]:


class Feat3Dataset(Dataset):
    
    def __init__(self, adj_path, da_path, transform=None):
        """ adj_path(string): path to the adjacency matrix
            da_path(string): path to da results, each file in the path has four columns:
                A, probability of feasibility, avg and std of objective energy
            transform(callable, optional): Optional transform to be applied
                on a sample.
        """
        self.groups = []
        da_files = os.listdir(da_path)
        adj_files = os.listdir(adj_path)

        self.data = []
        self.transform = transform
        
        for entry in adj_files:
            if not entry+'.csv' in da_files:
                continue

            adj_data = adj_file_handler(os.path.join(adj_path, entry))
#             da_data = da_file_handler(os.path.join(da_path, entry))
            da_data = da_csv_file_handler(os.path.join(da_path, entry+'.csv'))

            for inst in da_data:
                row = (np.hstack((adj_data, inst[0])), inst[1:])
                self.data.append(row)
                self.groups.append(entry)
            
    def __len__(self):
        return len(self.groups)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        elif isinstance(idx, int):
            idx = [idx]
        
        _input = np.asarray([self.data[i][0] for i in idx])
        _target = np.asarray([self.data[i][1] for i in idx])
        batch = {'input':_input, 'target':_target}
        
        if self.transform:
            batch = self.transform(batch)
            
        return batch
    
    def bestSet(self, pf):
        
        _dataset = [d for d in self]
        graphs = set(self.getGroup())

        for g in graphs:
            subset = [_dataset[i] for i, group in enumerate(self.getGroup()) if group == g]
            subset = [data for data in subset if data['target'][0][0]>=pf]
            bestInst = self._bestA(subset)
            yield bestInst
    
    def _bestA(self, subset):
        bestInst = None
        
        for data in subset:
            if bestInst is None:
                bestInst = data
            elif data['target'][0][1] <= bestInst['target'][0][1]:
                bestInst = data

        return bestInst
    
    def getGroup(self):
        return self.groups


# In[3]:


class Feat300Dataset(Feat3Dataset):
    
    def __init__(self, feat_path, adj_path, da_path, transform=None):
        """ feat_path(string): path to the features, each file in the path has 300 features 
                for each edge
            adj_path(string): path to the adjacency matrix
            da_path(string): path to da results, each file in the path has four columns:
                A, probability of feasibility, avg and std of objective energy
            transform(callable, optional): Optional transform to be applied
                on a sample.
        """
        self.groups = []
        feat_files = os.listdir(feat_path)
        da_files = os.listdir(da_path)
        adj_files = os.listdir(adj_path)

        self.data = []
        self.transform = transform
        
        for entry in adj_files:
            if not entry+'.npy' in feat_files:
                continue
            if not entry+'.csv' in da_files:
                continue

            adj_data = adj_file_handler(os.path.join(adj_path, entry))
            feat_data = feat_file_handler(os.path.join(feat_path, entry+'.npy'))
#             da_data = da_file_handler(os.path.join(da_path, entry))
            da_data = da_csv_file_handler(os.path.join(da_path, entry+'.csv'))

            for inst in da_data:
                row = (np.hstack((feat_data, adj_data, inst[0])), inst[1:])
                self.data.append(row)
                self.groups.append(entry)


# In[4]:


def adj_file_handler(adj_filename):
    adj = np.loadtxt(adj_filename)
    elements = adj[np.tril_indices(adj.shape[0])]
    avg = np.average(elements)
    std = np.std(elements)
    dim = adj.shape[0]
    
    return np.asarray([avg, std, dim])

def feat_file_handler(feat_filename):
    feat = np.load(feat_filename).mean(axis=(0,1))
    return feat

def da_file_handler(da_filename):
    da = np.loadtxt(da_filename, delimiter=',')
    return da

def da_csv_file_handler(da_filename):
    da_data = pd.read_csv(da_filename)
    da_data.columns = [col.strip() for col in da_data.columns]
    data = da_data[['A', 'da_distance', 'objective_energy']].to_numpy()
    ret = []
    a_set = np.unique(data[:,0])
    for a in a_set:
        subset = data[data[:,0]==a, :]
        pf = 1-(subset[:,1]==-1).sum()/subset.shape[0]
        avg = subset[:,2].mean()
        std = subset[:,2].std()
        row = [a, pf, avg, std]
        ret.append(row)
    return np.asarray(ret)


# In[5]:


class FeatTransformer:
    """Transform a batch of data to mean of zero and std of one. PF is left untouched.

    Args:
        scalerx (StandardScaler): for input feature normalization, feature + A
        scalery (StandardScaler): for target feature normalization, PF, Eavg and Estd
    """

    def __init__(self, scalerx, scalery):
        self.scalerx = scalerx
        self.scalery = scalery
        
    def __call__(self, sample):
        _input = self.scalerx.transform(sample['input'])
        _target = self.scalery.transform(sample['target'])
        _target[:,0] = sample['target'][:,0]
        
        return {'input': _input, 'target': _target}
    
    def inverse_transform_Y(self, Y):
        Y_orig = self.scalery.inverse_transform(Y)
        Y_orig[:,0] = Y[:,0]
        return Y_orig


# In[9]:


if __name__ == '__main__':
    
    # Demo the usage
    
    adj_path = '../dataset/round3/adj/train'
    da_path = '../dataset/round3/da/train'
    dataset = Feat3Dataset(adj_path=adj_path, da_path=da_path)
    X = np.asarray([d['input'][0] for d in dataset])
    Y = np.asarray([d['target'][0] for d in dataset])
    scalerx = StandardScaler().fit(X)
    scalery = StandardScaler().fit(Y)
    transform = FeatTransformer(scalerx, scalery)
    dataset = Feat3Dataset(adj_path=adj_path, da_path=da_path, transform=transform)
    
    for inst in dataset.bestSet(0.5):
        print(inst)

    print()
        
    feat_path = '../dataset/round3/feat300/test'
    adj_path = '../dataset/round3/adj/test'
    da_path = '../dataset/round3/da/test'
    dataset = Feat300Dataset(feat_path=feat_path, adj_path=adj_path, da_path=da_path)
    X = np.asarray([d['input'][0] for d in dataset])
    Y = np.asarray([d['target'][0] for d in dataset])
    scalerx = StandardScaler().fit(X)
    scalery = StandardScaler().fit(Y)
    transform = FeatTransformer(scalerx, scalery)
    dataset = Feat300Dataset(feat_path=feat_path, adj_path=adj_path, da_path=da_path, transform=transform)
    
    for inst in dataset:
        print('Input: ', inst['input'])
        print('Target: ', inst['target'])
        print('Target (inverse transformed): ', transform.inverse_transform_Y(inst['target']))
        break
        

