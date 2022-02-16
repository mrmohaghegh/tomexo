#! /usr/bin/env python3

import numpy as np
import pandas as pd
import os
import pickle
from oncotree import OncoTree, OncoNode
from anytree import RenderTree

#-------------------------------------------#
#----- GENERATIVE MODEL SPECIFICATIONS -----#
#-------------------------------------------#
nodes_dict = {}
### Linear Structure ###
root = OncoNode(genes=[], f=1)
c0 = OncoNode(parent=root, genes=[0,1,2,3], f=0.9)
c1 = OncoNode(parent=c0, genes=[4,5,6,7], f=0.7)
c2 = OncoNode(parent=c1, genes=[8,9,10,11], f=0.5)
c3 = OncoNode(parent=c2, genes=[12,13,14,15], f=0.3)
ps = OncoNode(genes=[16,17,18,19], f=0)
nodes_dict['linear'] = [root, c0, c1, c2, c3, ps]
### Tree Structure ###
root = OncoNode(genes=[], f=1)
c0 = OncoNode(parent=root, genes=[0,1,2], f=0.9)
c1 = OncoNode(parent=c0, genes=[3,4,5], f=0.7)
c2 = OncoNode(parent=c0, genes=[6], f=0.6)
c3 = OncoNode(parent=c0, genes=[7,8], f=0.3)
c1_1 = OncoNode(parent=c1, genes=[9,10,11], f=0.8)
c1_2 = OncoNode(parent=c1, genes=[12,13], f=0.5)
c2_1 = OncoNode(parent=c2, genes=[14,15], f=0.3)
ps = OncoNode(genes=[16,17,18,19], f=0)
nodes_dict['tree'] = [root, c0, c1, c2, c3, c1_1, c1_2, c2_1, ps]
### Mixture Structure ###
root = OncoNode(genes=[], f=1)
c0 = OncoNode(parent=root, genes=[0,1,2], f=0.8)
c1 = OncoNode(parent=root, genes=[3,4,5], f=0.9)
c2 = OncoNode(parent=root, genes=[6], f=0.5)
c1_1 = OncoNode(parent=c1, genes=[7,8], f=0.3)
c1_2 = OncoNode(parent=c1, genes=[9,10,11], f=0.7)
c2_1 = OncoNode(parent=c2, genes=[12,13], f=0.8)
c2_1_1 = OncoNode(parent=c2_1, genes=[14,15], f=0.4)
ps = OncoNode(genes=[16,17,18,19], f=0)
nodes_dict['mixture'] = [root, c0, c1, c2, c1_1, c1_2, c2_1, c2_1_1, ps]
####----------------####------------------####

#-------------------------------------------#
#-------- Constructing the datasets --------#
#-------------------------------------------#

n_list = [50, 100, 200, 500]
e_list = [0.001, 0.01, 0.05, 0.1]

if 'synthetic' not in os.listdir('data'):
    os.mkdir('data/synthetic')

for mode in ['linear', 'tree', 'mixture']:
    np.random.seed(1)
    if mode not in os.listdir('data/synthetic'):
        os.mkdir('data/synthetic/{}'.format(mode))

    for e in e_list:
        pkl_file = 'data/synthetic/{}/{}.pkl'.format(mode, e)
        gen_progmo = OncoTree(nodes_dict[mode], pfp=e, pfn=e)
        gene_names = ['g{}'.format(i) for i in gen_progmo.genes]
        with open(pkl_file, 'wb') as f:
            pickle.dump(gen_progmo, f)
        for n in n_list:
            df_file = 'data/synthetic/{}/{}-{}.csv'.format(mode, n, e)
            dataset, _ =gen_progmo.draw_sample(n)
            tumor_ids = ['tumor{}'.format(i) for i in range(n)]
            df = pd.DataFrame(data=np.array(dataset, dtype=int), columns=gene_names, index=tumor_ids)
            ## Comment lines for csv file:
            comment = '# Synthetic dataset  \n'
            comment += '# Number of tumors: %i \n' % n
            comment += '# Number of mutations: %i \n' % gen_progmo.n_genes
            comment += '# driver tree: \n'
            for pre, fill, node in RenderTree(gen_progmo.root):
                comment += '# %s%s\n' % (pre, node.genes)
            comment += '# set of passengers: \n'
            for pre, fill, node in RenderTree(gen_progmo.ps):
                comment += '# %s%s\n' % (pre, node.genes)
            comment += '# p(FP): %.4f \n' % e
            comment += '# p(FN): %.4f \n' % e
            comment += '# ------------------------------------ #\n'
            with open(df_file, 'w') as f:
                f.write(comment)
            ## Done with comments
            with open(df_file, 'a') as f:          
                df.to_csv(f, sep=',')
