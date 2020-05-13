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

        da_files = os.listdir(da_path)
        adj_files = os.listdir(adj_path)

        self.df = None
        self.transform = transform

        for entry in adj_files:
            if not entry+'.csv' in da_files:
                continue

            adj_data = adj_file_handler(os.path.join(adj_path, entry))
            da_data = da_pd_file_handler(os.path.join(da_path, entry+'.csv'))

            da_data['graph'] = entry
            da_data['feat3'] = None
            da_data['feat3'] = da_data['feat3'].apply(lambda x: adj_data)

            if self.df is None:
                self.df = da_data
            else:
                self.df = self.df.append(da_data, ignore_index=True)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        elif isinstance(idx, (int, np.integer)):
            idx = [idx]
        if any([i not in range(len(self)) for i in idx]):
            raise IndexError

        _input = self.df.loc[idx, ['feat3','A']].apply(lambda x: np.hstack([x['feat3'],[x['A']]]), axis=1)
        _target = self.df.loc[idx, ['PF', 'Eavg', 'Estd']]
        _graph = self.df.loc[idx, 'graph']
        _dmin = self.df.loc[idx, 'dists'].apply(lambda x: np.nan if len(x)==0 else x.min())

        batch = {'input':np.vstack(_input.values),
                 'target':np.vstack(_target.values),
                 'graph':np.vstack(_graph.values),
                 'dmin':np.vstack(_dmin.values),}

        if self.transform:
            batch = self.transform(batch)

        if batch['input'].shape[0]==1:
            batch['input'] = batch['input'][0]
            batch['target'] = batch['target'][0]
            batch['graph'] = batch['graph'][0]
            batch['dmin'] = batch['dmin'][0]

        return batch

    def bestSet(self, pf):
        groups = self.df.groupby('graph')
        for name, group in groups:
            idx = group[group['PF']>=pf]['Eavg'].idxmin(axis=0)
            yield self[idx]

    def getGroup(self):
        return self.df['graph'].tolist()


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
        feat_files = os.listdir(feat_path)
        da_files = os.listdir(da_path)
        adj_files = os.listdir(adj_path)

        self.df = None
        self.transform = transform

        for entry in adj_files:
            if not entry+'.npy' in feat_files:
                continue
            if not entry+'.csv' in da_files:
                continue

            adj_data = adj_file_handler(os.path.join(adj_path, entry))
            feat_data = feat_file_handler(os.path.join(feat_path, entry+'.npy'))
            da_data = da_pd_file_handler(os.path.join(da_path, entry+'.csv'))

            da_data['graph'] = entry
            da_data['feat3'] = None
            da_data['feat3'] = da_data['feat3'].apply(lambda x: adj_data)
            da_data['feat300'] = None
            da_data['feat300'] = da_data['feat3'].apply(lambda x: feat_data)

            if self.df is None:
                self.df = da_data
            else:
                self.df = self.df.append(da_data, ignore_index=True)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        elif isinstance(idx, (int, np.integer)):
            idx = [idx]
        if any([i not in range(len(self)) for i in idx]):
            raise IndexError

        _input = self.df.loc[idx, ['feat300','feat3','A']].apply(lambda x: np.hstack([x['feat300'],x['feat3'],[x['A']]]), axis=1)
        _target = self.df.loc[idx, ['PF', 'Eavg', 'Estd']]
        _graph = self.df.loc[idx, 'graph']
        _dmin = self.df.loc[idx, 'dists'].apply(lambda x: np.nan if len(x)==0 else x.min())

        batch = {'input':np.vstack(_input.values),
                 'target':np.vstack(_target.values),
                 'graph':np.vstack(_graph.values),
                 'dmin':np.vstack(_dmin.values),}

        if self.transform:
            batch = self.transform(batch)

        if batch['input'].shape[0]==1:
            batch['input'] = batch['input'][0]
            batch['target'] = batch['target'][0]
            batch['graph'] = batch['graph'][0]
            batch['dmin'] = batch['dmin'][0]

        return batch


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
    set_trace()
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

def da_pd_file_handler(da_filename):
    data = pd.read_csv(da_filename)
    data['feasibility'] = data['da_distance']!=-1
    groups = data.groupby('A')
    eavg = groups['objective_energy'].mean()
    estd = groups['objective_energy'].std()
    pf = groups['feasibility'].apply(lambda x: x.sum()/x.shape[0])
    dists = groups['da_distance'].apply(lambda x: x[x>0].values)
    return pd.concat([pf, eavg, estd, dists], axis=1, keys=['PF', 'Eavg', 'Estd', 'dists']).reset_index()



# In[5]:


class FeatTransformer:
    """Transform a batch of data to mean of zero and std of one. PF is left untouched.

    Args:
        scalerx (StandardScaler): for input feature normalization, feature + A
        scalery (StandardScaler): for target feature normalization, PF, Eavg and Estd and Dmin
    """

    def __init__(self, scalerx, scalery):
        self.scalerx = scalerx
        self.scalery = scalery

    def __call__(self, sample):

        sample['input'] = self.scalerx.transform(sample['input'])

        _target = self.scalery.transform(sample['target'])
        _target[:, 0] = sample['target'][:, 0]
        sample['target'] = _target

        _dummyY = np.zeros_like(sample['target'])
        _dummyY[:, 1] = sample['dmin'][:, 0]
        sample['dmin'] = self.scalery.transform(_dummyY)[:, 1]

        return sample

    def inverse_transform_Y(self, Y):
        Y_orig = self.scalery.inverse_transform(Y)
        Y_orig[:,0] = Y[:,0]
        return Y_orig


# In[6]:


if __name__ == '__main__':

    # Demo the usage

    adj_path = '../dataset/round3/adj/'
    da_path = '../dataset/round3/da/'
    dataset = Feat3Dataset(adj_path=adj_path, da_path=da_path)
    X = np.asarray([d['input'] for d in dataset])
    Y = np.asarray([d['target'] for d in dataset])
    scalerx = StandardScaler().fit(X)
    scalery = StandardScaler().fit(Y)
    transform = FeatTransformer(scalerx, scalery)
    dataset = Feat3Dataset(adj_path=adj_path, da_path=da_path, transform=transform)

    for inst in dataset:
        print(inst)
        break


#     print()

#     feat_path = '../dataset/round3/feat300/'
#     adj_path = '../dataset/round3/adj/'
#     da_path = '../dataset/round3/da/'
#     dataset = Feat300Dataset(feat_path=feat_path, adj_path=adj_path, da_path=da_path)
#     X = np.asarray([d['input'] for d in dataset])
#     Y = np.asarray([d['target'] for d in dataset])
#     scalerx = StandardScaler().fit(X)
#     scalery = StandardScaler().fit(Y)
#     transform = FeatTransformer(scalerx, scalery)
#     dataset = Feat300Dataset(feat_path=feat_path, adj_path=adj_path, da_path=da_path, transform=transform)

#     for inst in dataset.bestSet(0.5):
#         print(inst)
#         break

