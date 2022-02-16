#! /usr/bin/env python3

import numpy as np
import pandas as pd
import os


filter_silents = True
hm_coeff = 0.02 # Threshold of mutation rate of genes to be considered

input_dir = 'data/gdac.broadinstitute.org_GBM.Mutation_Packager_Calls.Level_3.2016012800.0.0'
df_full_dir = 'data/processed/gdac_firehose_gbm_full.csv' # can be set to None
intogen_genes_dir = 'IntOGen/IntOGen-DriverGenes_GBM_TCGA.tsv'
df_filtered_dir = 'HMIntogen/gdac_firehose_GBM.csv'

# ----------------------------------------- #
# ------- Builing the full DataFrame ------ #
# ----------------------------------------- #

files_list = []
for file_name in os.listdir(input_dir):
    if file_name.startswith('TCGA'):
        files_list.append(file_name)
n_patients = len(files_list)
df_full = pd.DataFrame()
for i, file_name in enumerate(files_list):
    file_address = os.path.join(input_dir, file_name)
    df_input = pd.read_csv(file_address, sep='\t', comment='#')
    if filter_silents:
        df_input = df_input[~df_input.Variant_Classification.isin(['Silent', 'RNA'])]
    for index, row in df_input.iterrows():
        df_full.at[row.Tumor_Sample_Barcode, row.Hugo_Symbol] = True
df_full = df_full.fillna(False).astype(int)
df_full = df_full.sort_index(axis='index')
df_full = df_full.sort_index(axis='columns')
if df_full_dir is not None:
    df_full.to_csv(output_file)

# ----------------------------------------- #
# -------------- Filtering ---------------- #
# ----------------------------------------- #

intogen_genes_list = list(pd.read_csv(intogen_genes_dir, sep='\t', comment='#').Symbol)
n_tumors, n_genes = df_full.shape
# finding intogen genes:
gene_names = list(df_full.columns)
intogen_list = []
for gene in intogen_genes_list:
    if gene in gene_names:
        intogen_list.append(gene_names.index(gene))
intogen_list = np.array(intogen_list)
# finding highly mutated genes:
th_hm = 0.02*n_tumors
hm_list = np.where(df_full.sum()>=th_hm)[0]
# Constucting the list of genes based on the filters:
genes_to_keep = np.intersect1d(intogen_list, hm_list)
# Filtering and saving the resulting df:
filtered_df = df_full.iloc[:, genes_to_keep]
## Comment lines for csv file:
comment = '# GDAC firehose dataset  \n'
comment += '# Number of tumors: %i \n' % n_tumors
comment += '# Number of genes before filtering: %i \n' % n_genes
comment += '# Number of genes after filtering: %i \n' % len(genes_to_keep)
with open(df_filtered_dir, 'w') as f:
    f.write(comment)
with open(df_filtered_dir, 'a') as f:          
    filtered_df.to_csv(f, sep=',')