#! /usr/bin/env python3

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from oncotree import OncoTree

sample_file = 'result/GDAC/coadread/progmo.pkl'
data_file = 'data/AllIntogen/gdac_firehose_coadread.csv'
pdf_dir = 'result/sandbox/coadread.pdf'

prior_coeff = 5

df_input = pd.read_csv(data_file, delimiter=',', index_col=None, comment='#')
if df_input.iloc[0,0] not in [0,1]:
    # The csv does have index column
    # RELOADING
    df_input = pd.read_csv(data_file, delimiter=',', index_col=0, comment='#')
gene_names = list(df_input.columns)
dataset = np.array(df_input, dtype=bool)
with open(sample_file, 'rb') as f:
    sample = pickle.load(f)
print('Number of tumors: %i\nNumber of genes: %i'%(dataset.shape[0], dataset.shape[1]))
sample_post = sample.likelihood(dataset) + sample.prior(prior_coeff)
print('Log-posterior of the output: %.2f'%sample_post)
print('Epsilon: %.3f \nDelta: %.3f'%(sample.pfp, sample.pfn))
star_tree = OncoTree.star_from_dataset(
    dataset,
    single_error=False,
    error_estimation = True
    )
star_tree = star_tree.assign_error_values(dataset)
star_tree = star_tree.assign_f_values(dataset, fine_tuning=True)
star_tree,llhs = star_tree.fit_error_params(dataset)
star_llh = star_tree.likelihood(dataset)
star_post = star_llh + star_tree.prior(prior_coeff)
print('Star tree log-posterior: %.2f'%star_post)
print('llh improvement ratio %.2e'%np.exp(sample_post-star_post))
print('per-tumor-llh improvement ratio %.4f'%np.exp((sample_post-star_post)/dataset.shape[0]))

sample.to_dot(dataset, gene_names=gene_names, show_passengers="True", fig_file=pdf_dir)