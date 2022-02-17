#! /usr/bin/env python3

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from oncotree import OncoTree

sample_file = 'output/progmo.pkl'
data_file = 'gdac_firehose_GBM.csv'
pdf_dir = 'gbm.pdf'

df_input = pd.read_csv(data_file, delimiter=',', index_col=0, comment='#')
gene_names = list(df_input.columns)
dataset = np.array(df_input, dtype=bool)
with open(sample_file, 'rb') as f:
    sample = pickle.load(f)
print(dataset.shape)
print('llh BEFORE fine-tuning the params: %.2f'%sample.likelihood(dataset))
sample = sample.assign_f_values(dataset, fine_tuning=True)
sample,llhs=sample.fit_error_params(dataset)
sample_llh = sample.likelihood(dataset)
print('llh AFTER fine-tuning the params: %.2f'%sample_llh)
print('Epsilon: %.3f \nDelta: %.3f'%(sample.pfp, sample.pfn))
star_tree = OncoTree.star_from_dataset(
    dataset,
    single_error=False,
    error_estimation = True
    )
star_tree = star_tree.assign_f_values(dataset, fine_tuning=True)
star_tree,llhs=star_tree.fit_error_params(dataset)
star_llh = star_tree.likelihood(dataset)
print('star tree llh (after fine tuning): %.2f'%star_llh)
print('llh improvement ratio %.2e'%np.exp((sample_llh-star_llh)))
print('per-tumor-llh improvement ratio %.4f'%np.exp((sample_llh-star_llh)/dataset.shape[0]))

sample.to_dot(dataset, gene_names=gene_names, show_passengers="True", fig_file=pdf_dir)