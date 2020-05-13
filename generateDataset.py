#!/usr/bin/env python
# coding: utf-8

# ## Generate Dataset

# ### Generate Graph

# In[1]:


from dummyAPI import api as smu_api
import os
import time
import numpy as np
import argparse
import pandas as pd


# In[2]:


parser = argparse.ArgumentParser(description='Sampling As for a folder of graphs')
parser.add_argument('--graph_path', type=str, required=True)
parser.add_argument('--da_path', type=str, default='log')
parser.add_argument('--a_per_graph', type=int, default=100)
parser.add_argument('--extend', action='store_true', help='Extend dataset')


# In[10]:


def main():
    global args
    
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.da_path):
        os.mkdir(args.da_path)
        
    if args.extend:
        # The coverage of A is not good enough for digiplex hyper-parameter tuning
        # The goal of the snippet of code is to extend the dataset a little
        
        for entry in os.listdir(args.graph_path):

            daFilename = os.path.join(args.da_path, entry)+'.csv'
            gFilename = os.path.join(args.graph_path, entry)

            if not os.path.exists(daFilename):
                print('Skip {} because no da file is found'.format(gFilename), flush=True)
                continue

            a_list = generate_a(daFilename)
            
            if len(a_list)==0:
                print('Skip {} because it has already been extended'.format(gFilename), flush=True)
                continue

            func = lambda A: evaluate(gFilename, A, daDir=args.da_path)[0]
            
            print('Extending {}'.format(gFilename), end='', flush=True)
            for a in generate_a(daFilename):
                func(a)
                print('.', end='', flush=True)
            print('done', flush=True)

    else:
        
        for entry in os.listdir(args.graph_path):
            daFilename = os.path.join(args.da_path, entry)+'.csv'
            gFilename = os.path.join(args.graph_path, entry)
            if os.path.exists(daFilename):
                print('Skip {}'.format(gFilename))
                continue
            evaluateGraph(gFilename, numSample=args.a_per_graph, daDir=args.da_path)



# ### API related stuffs

# In[1]:


def evaluateGraphAPair(gFilename, A, timeout=1200):
    """ Given graph in np.loadtxt compatible format, an A
        returns the da data including 
         A  iterationnumber  da_distance  objective_energy
         objective_energy_unperturbed  column_constraint_energy
         row_constraint_energy  
    """
    

    waitTotal = 0
    waitInterval = 5
    
    while True:
        try:
            status_code, reason, text = smu_api.post(gFilename, A)
            break
        except:
            print('Submit job ({},{}) failed, retry'.format(gFilename, A))
            time.sleep(waitInterval)


    ret = 'init'

    while type(ret) == str:
        time.sleep(waitInterval)
#         ret = smu_api.get(text)      
        while True:
            try:
                ret = smu_api.get(text)
                break
            except:
                print('Retrieve job ({},{}) failed, retry'.format(gFilename, A))
                time.sleep(waitInterval)
        
        waitTotal += waitInterval
        if waitTotal>timeout:
            return None

    return ret

def getStatistics(da_data):
    """ Given da_data in pandas dataframe format
        returns the probability of feasibility
        mean of objective_energy
        std of objective_energy
        
    """
    pf = (da_data[['da_distance']]!=-1).to_numpy().sum() / da_data.shape[0]
    avg_obj = da_data[['objective_energy']].mean().to_numpy()[0]
    std_obj = da_data[['objective_energy']].std().to_numpy()[0]
    return pf, avg_obj, std_obj

def saveProgress(entry, da_data, daDir='log/'):
    if not os.path.exists(daDir):
        os.mkdir(daDir)

    da_filename = os.path.join(daDir, entry)+'.csv'
    
    if not os.path.isfile(da_filename):
        da_data.to_csv(da_filename, header=True, index=False, mode='a+')
    else:
        da_data.to_csv(da_filename, header=False, index=False, mode='a+')

def evaluate(gFilename, A, daDir='log/'):
    
    da_data = None
    RETRY = 3
    nTry = 0

    while da_data is None and nTry < RETRY: 

        # evaluate the given graph A pair
        da_data = evaluateGraphAPair(gFilename, A)
        
        nTry += 1
    
    if da_data is None:
        print('Failed on graph, A: {}, {}'.format(gFilename, A))
        return None

    # calculate the statistics of the da_data
    pf, avg_obj, std_obj = getStatistics(da_data)
    
    # save statistics for future use
    baseName = os.path.basename(gFilename)
    saveProgress(baseName, da_data, daDir=daDir)
    
    return pf, avg_obj, std_obj


# ### Search Range of A

# In[5]:


def guessA(gFilename):
    """ Initial guess about A
        A good start could be the Average of edge weight
    """
    graph = np.loadtxt(gFilename)
    return graph.max()

def findACoarseRange(A, func):
    """ Given a random guess of A, find Amin and Amax exponentially
        Amin is the bigest A that leads to PF=0
        Amax is the smallest A that leads to PF=1
        It's OK if the Amin and Amax we find is far away from ground truth
        We will carry further search to approach the ground truth
    """
    p = func(A)
    if p<=0:
        # when the guess leads to p==0
        Amin = A
        Amax = Amin * 2
        while p<1:
            p=func(Amax)
            if p<=0:
                Amin = Amax
                Amax = Amin * 2
            elif p>0 and p<1:
                Amax += Amax - Amin
    elif p>0 and p<1:
        # when the guess leads to p in (0,1)
        Amin = A
        Amax = A
        while p>0:
            p=func(Amin)
            if p>0:
                Amin /= 2
        while p<1:
            p=func(Amax)
            if p<1:
                Amax += Amax-Amin
    else:
        # when the guess leads to p==1
        Amax = A
        Amin = Amax / 2
        while p>0:
            p=func(Amin)
            if p>=1:
                Amax = Amin
                Amin = Amax / 2
            elif p>0 and p<1:
                Amin /= 2
                
    return Amin, Amax


# In[6]:


def findAFineRange(Amin, Amax, func):
    """ Given coarse range of A, find fine range of A using iterative grid search
    """
    betterAmin, betterAmax = Amin, Amax
    flagUpdated = True
    while flagUpdated:
        flagUpdated = False
        As = np.linspace(betterAmin, betterAmax, num=5)[1:-1]
        pfs = np.asarray([func(a) for a in As])

        indices = np.where(pfs<=0)[0]
        if len(indices)>0:
            betterAmin = As[indices[-1]]
#             print('Amin updated to {}'.format(betterAmin))
            flagUpdated = True
        indices = np.where(pfs>=1)[0]
        if len(indices)>0:
            betterAmax = As[indices[0]]
#             print('Amax updated to {}'.format(betterAmax))
            flagUpdated = True
    
    return betterAmin, betterAmax


# In[7]:


def uniformSample(Amin, Amax, func, num=100, report_interval=10):
    """ Sample As uniformly within given range
    """
    assert Amax>Amin
    assert Amax>0 and Amin>0
    As = np.linspace(Amin, Amax, num=num+2)[1:-1]

    for i, a in enumerate(As):
        func(a)
        if i % report_interval == report_interval-1:
            print('.', end='')

    print('done')
    
def gaussianSample(avg, std, func, num=100, report_interval=10):
    """ Sample gaussianly with given mean and std
    """
    
    As = np.random.normal(avg, std, num)
    
    for i, a in enumerate(As):
        if a<0:
            func(-a)
        else:
            func(a)
        if i % report_interval == report_interval-1:
            print('.', end='')

    print('done')


# In[8]:


def generate_a(da_filename, enlarge=5, num_bin=20, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
        
    da_data = pd.read_csv(da_filename)
    
    da_pf = da_data.groupby('A').agg(pf=('da_distance', lambda x: (x>0).values.sum()/x.size))
    da_pf.sort_index(inplace=True)
    a_pf_0 = da_pf[da_pf['pf']>0].head(1).index.values[0]
    a_pf_1 = da_pf[da_pf['pf']<1].tail(1).index.values[0]
    a_pf_range = a_pf_1-a_pf_0
    
    half_side = (enlarge-1)/2

    a_min = max(1, a_pf_0 - a_pf_range * half_side)
    a_max = a_pf_1 + a_pf_range * half_side
    bins = np.linspace(a_min, a_max, num=num_bin)

    hist, bins = np.histogram(da_pf.index.values, bins=bins)

    random_factor = np.random.rand((hist==0).sum()) * (a_max-a_min) / (num_bin-1)
    return bins[:-1][hist==0] + random_factor


# In[9]:


def evaluateGraph(gFilename, numSample=100, daDir='log/'):
    func = lambda A: evaluate(gFilename, A, daDir=daDir)[0]
    print('Evaluating As for graph {}'.format(gFilename))
    initA = guessA(gFilename)
    print('\tInitial guess of A is {}'.format(initA))
    Amin, Amax = findACoarseRange(initA, func)
    print('\tFind coarse range of A: [{},{}]'.format(Amin, Amax))
    Amin, Amax = findAFineRange(Amin, Amax, func)
    print('\tFind fine range of A: [{},{}]'.format(Amin, Amax))
    print('\tSweeping A ranging [{},{}]'.format(Amin, Amax), end='')
    uniformSample(Amin, Amax, func, num=numSample//5*4)
    print('\tGaussian sampling A with (avg, std): ({},{})'.format((Amax+Amin)/2, (Amax-Amin)/2), end='')
    gaussianSample((Amax+Amin)/2, (Amax-Amin)/2, func, num=numSample//5)
    


# In[ ]:


if __name__ == '__main__':
    main()

