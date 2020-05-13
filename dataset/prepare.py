#!/usr/bin/env python
# coding: utf-8

# In[36]:


import os
import time
import argparse
import glob
import random
import shutil


# In[7]:


parser = argparse.ArgumentParser(description='Sampling As for a folder of graphs')
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--test_size', type=float, default=0.1)
parser.add_argument('--data_path', type=str, required=True)


# In[34]:


def main():
    global args
    
    #cmd = '--data_path ./'
    #cmd += ' --random_seed 1'
    #cmd += ' --test_size 0.1'
    #args = parser.parse_args(cmd.split(' '))

    args = parser.parse_args()
    
    random.seed(args.random_seed)
    
    shuffle_orders = None

    for folder in [f for f in glob.glob(args.data_path+'/*') if os.path.isdir(f)]:
        train_folder = os.path.join(folder, 'train')
        test_folder = os.path.join(folder, 'test')
        
        files = [f for f in glob.glob(folder+'/*') if os.path.isfile(f)]
        files.sort()
        
        if shuffle_orders is None:
            shuffle_orders = [i for i in range(len(files))]
            random.shuffle(shuffle_orders)
            sep = int(len(files) * args.test_size)
            shuffle_orders[:sep]
            shuffle_orders[sep:]
        assert len(shuffle_orders)==len(files)
            
        train_files = [files[idx] for idx in shuffle_orders[sep:]]    
        test_files = [files[idx] for idx in shuffle_orders[:sep]]
        
        os.mkdir(train_folder)
        os.mkdir(test_folder)
        
        for file in train_files:
            shutil.move(file, train_folder)
        for file in test_files:
            shutil.move(file, test_folder)


# In[35]:


if __name__ == '__main__':
    main()

