#! /usr/bin/env python3

# --------------------------------------- #
# Example:
# python tomexo.py -i data/HMIntogen/gdac_firehose_thym.csv -o result/HMIntogen_thym --n_chains 2 --n_mixing 2 --n_samples 2000
# --------------------------------------- #

import argparse
import numpy as np
import pandas as pd
import os
import time
from itertools import product
from multiprocessing import Pool
from copy import deepcopy

from oncotree import OncoTree
from util import save_result, Geweke, Gelman_Rubin

def initialize(dataset, coeff):
    if coeff<0:
        sample = OncoTree.star_from_dataset(
            dataset,
            single_error=False,
            error_estimation = True
            )
    else:
        sample = OncoTree.from_dataset(
            dataset,
            coeff=coeff,
            passenger_threshold=0.0001,
            single_error=False,
            error_estimation = True
            )
    p = sample.likelihood(dataset) + sample.prior()
    return(sample, p)

def run_race(args):
    sample = args[0]
    seed = args[1]
    if len(args)==3:
        current_posterior=args[2]
    else:
        current_posterior=None
    
    return(sample.fast_training_iteration(
        dataset,
        n_iters=n2,
        seed=seed,
        current_posterior=current_posterior
        )
    )

def fit_oncotree(dataset, print_running_status=False, include_linear_initializations=False):
    st = time.time()
    if include_linear_initializations:
        coeffs = [-1]
        coeffs.extend(list(np.linspace(0, 1, n0-1)))
    else:
        coeffs = [-1 for _i in range(n0)]
    random_seeds = np.arange(1, n0+1)
    best_raw_samples = []
    sample_list = []
    p_list = []
    for coeff in coeffs:
        _sample, _p = initialize(dataset, coeff)
        sample_list.append(_sample)
        p_list.append(_p)
    overall_posterior_array = np.empty(shape=(n0, 0))
    overall_updates_array = np.empty(shape=(n0, 0))
    details_dict = {} # Keeping number of proposed samples, novel samples and accepted samples, per move type
    for _training_race in range(n1+1):
        args_list = [(sample_list[i], random_seeds[i], p_list[i]) for i in range(len(sample_list))]
        with Pool() as p:
            results = p.map(run_race, args_list)
        if print_running_status:
            print("\nMixing point %i (out of %i) \nTime spent so far: %.1f seconds"%(_training_race,n1,time.time()-st))
            print("Number of accepted moves per chain (out of %i proposed moves):"%n2)
            print([result_tuple[5] for result_tuple in results])
        race_posterior_array = np.array([result_tuple[4] for result_tuple in results])
        overall_posterior_array = np.concatenate((overall_posterior_array, race_posterior_array), axis=1)
        race_updates_array = np.array([[result_tuple[5] for result_tuple in results]]).T
        overall_updates_array = np.concatenate((overall_updates_array, race_updates_array), axis=1)
        details_dict[_training_race] = [result_tuple[6] for result_tuple in results]
        # Initializing the next race
        # Format of the result tuples: (sample, current_posterior, best_sample, best_posterior, posteriors_list, n_updates)
        sample_list = []
        p_list = []
        # In previous versions, we wanted to give the currently best chain (best on their last state)
        # ... to continue. The other chains were to pick the best sample ever encountered and continue
        ###
        # current_posteriors = [result_tuple[1] for result_tuple in results]
        # bcp_idx = np.argmax(current_posteriors) # Best Current Posterior
        # sample_list.append(deepcopy(results[bcp_idx][0]))
        # p_list.append(deepcopy(results[bcp_idx][1]))
        # best_posteriors = [result_tuple[3] for result_tuple in results]
        # bop_idx = np.argmax(best_posteriors) # Best Overall Posterior
        # for _i in range(0, n0-1):
        #     sample_list.append(deepcopy(results[bop_idx][2]))
        #     p_list.append(deepcopy(results[bop_idx][3]))
        ###
        # In the new version, we make a list of best samples encountered by individual chains, prune them,
        # ... look at the likelihoods of the pruned trees and all chains continue from the best pruned tree
        #best_raw_llhs = [raw_sample.likelihood(dataset)+raw_sample.prior() for raw_sample in best_raw_samples]
        best_pruned_samples = [result_tuple[2].prune(dataset) for result_tuple in results]
        best_pruned_llhs = [pruned_sample.likelihood(dataset)+pruned_sample.prior() for pruned_sample in best_pruned_samples]
        best_index = np.argmax(best_pruned_llhs)
        sample_list = [best_pruned_samples[best_index] for _i in range(n0)]
        p_list = [best_pruned_llhs[best_index] for _i in range(n0)]
        # done!
        best_raw_samples.append([result_tuple[2] for result_tuple in results])
    return(best_pruned_samples[best_index], overall_posterior_array, overall_updates_array, details_dict, best_raw_samples)

#--------------------------------------
#---- Parsing command line input ------
#--------------------------------------

parser = argparse.ArgumentParser(
    description='SGClone v0.5'
    )

parser.add_argument('-i', '--input', help='input csv file')
parser.add_argument('-o', '--output', help='output directory')
parser.add_argument('--n_chains', help='number of chains', default=10, type=int)
parser.add_argument('--n_mixing', help='number of chain mixing events', default=0, type=int)
parser.add_argument('--n_samples', help='number of samples between mixing events', default=100000, type=int)

args = parser.parse_args()

n0 = args.n_chains
n1 = args.n_mixing
n2 = args.n_samples
input_file = args.input
output_folder = args.output

#--------------------------------------
#------ Running the algorithm ---------
#--------------------------------------

df_input = pd.read_csv(input_file, delimiter=',', index_col=None, comment='#')
if df_input.iloc[0,0] not in [0,1]:
    # The csv does have index column
    # RELOADING
    df_input = pd.read_csv(input_file, delimiter=',', index_col=0, comment='#')
gene_names = list(df_input.columns)
dataset = np.array(df_input, dtype=bool)
start_time = time.time()
progmo, posterior_array, updates_array, details_dict, best_raw_samples = fit_oncotree(dataset)
spent_time = time.time()-start_time

#--------------------------------------
#--------- Post-Processing ------------
#--------------------------------------

save_result(os.path.join(output_folder, 'progmo.pkl'), progmo)
save_result(os.path.join(output_folder, 'raw_samples.pkl'), best_raw_samples)

df_posterior_array = pd.DataFrame(posterior_array)
df_posterior_array.to_csv(os.path.join(output_folder, 'posteriors.csv'))

report = '# Dataset %s \n' % input_file
report += '# Number of tumors: %i \n'  %dataset.shape[0]
report += '# Number of genes: %i \n'  %dataset.shape[1]
report += '# Analysis folder %s \n' % output_folder
report += '# Analysis parameters: %i chains, %i mixings, %i samples between mixings \n' %(n0,n1,n2)
report += '# Analysis time: %i seconds \n'%spent_time
report += '# ----------Convergence analysis------------ # \n'
gs_vector = posterior_array[np.argmax(np.max(posterior_array[:, -n2:], axis=1)),-n2:]
(_b, _GS) = Geweke(gs_vector)
if _b:
    report += '# Best chain has converged (Geweke z score: %.3f) \n' %_GS
else:
    report += '# Best chain has NOT converged (Geweke z score: %.3f) \n' %_GS

if n0>1: # if there is more than one chain
    set_of_posts = [posterior_array[_i,-n2:] for _i in range(posterior_array.shape[0])]
    (_b, _GR) =Gelman_Rubin(set_of_posts)
    if _b:
        report += '# Gelman_Rubin convergence IS achieved (GR score: %.3f) \n' %_GR
    else:
        report += '# Gelman_Rubin convergence IS NOT achieved (GR score: %.3f) \n' %_GR

report += '# ----------Result analysis------------ # \n'

star_tree = OncoTree.star_from_dataset(
    dataset,
    single_error=False,
    error_estimation = True
    )

# report += '# BEFORE FINE-TUNING the error parameters:\n'

# star_llh = star_tree.likelihood(dataset)
# report += '# Star tree log likelihood: %.4f \n' %star_llh
# star_pri = star_tree.prior()
# report += '# Star tree log prior: %.4f \n' %star_pri
# report += '# Star tree log posterior: %.4f \n' %(star_pri+star_llh)

# best_llh = progmo.likelihood(dataset)
# report += '# Best sample log likelihood: %.4f \n' %best_llh
# best_pri = progmo.prior()
# report += '# Best sample log prior: %.4f \n' %best_pri
# report += '# Best sample log posterior: %.4f \n' %(best_pri+best_llh)
# report += '# Best sample epsilon: %.5f \n' %(progmo.pfp)
# report += '# Best sample delta: %.5f \n' %(progmo.pfn)

report += '# AFTER FINE-TUNING the error parameters:\n'

star_tree = star_tree.assign_error_values(dataset)
star_tree = star_tree.assign_f_values(dataset, fine_tuning=True)
star_tree, _ = star_tree.fit_error_params(dataset)
star_llh = star_tree.likelihood(dataset)
report += '# Star tree log likelihood: %.4f \n' %star_llh
star_pri = star_tree.prior()
report += '# Star tree log prior: %.4f \n' %star_pri
report += '# Star tree log posterior: %.4f \n' %(star_pri+star_llh)

# progmo, _ = progmo.fit_error_params(dataset)
best_llh = progmo.likelihood(dataset)
report += '# Best sample log likelihood: %.4f \n' %best_llh
best_pri = progmo.prior()
report += '# Best sample log prior: %.4f \n' %best_pri
report += '# Best sample log posterior: %.4f \n' %(best_pri+best_llh)
report += '# Best sample epsilon: %.5f \n' %(progmo.pfp)
report += '# Best sample delta: %.5f \n' %(progmo.pfn)
report += '# Move type specific statistics:\n'

details_file = os.path.join(output_folder, 'analysis_details.csv')
with open(details_file, 'w') as f:
    f.write(report)

pds = []
for _race_idx in range(n1+1):
    race_details_proposed = [details_dict[_race_idx][_chain_idx]['n_proposed'] for _chain_idx in range(n0)]
    pds.append(pd.DataFrame(race_details_proposed, dtype=int, index = ['n_proposed(r{}-c{})'.format(_race_idx,_chain_idx) for _chain_idx in range(n0)]))
    race_details_novel = [details_dict[_race_idx][_chain_idx]['n_novel'] for _chain_idx in range(n0)]
    pds.append(pd.DataFrame(race_details_novel, dtype=int, index = ['n_novel(r{}-c{})'.format(_race_idx,_chain_idx) for _chain_idx in range(n0)]))
    race_details_accepted = [details_dict[_race_idx][_chain_idx]['n_accepted'] for _chain_idx in range(n0)]
    pds.append(pd.DataFrame(race_details_accepted, dtype=int, index = ['n_accepted(r{}-c{})'.format(_race_idx,_chain_idx) for _chain_idx in range(n0)]))
pd_to_save = pd.concat(pds)
pd_to_save.to_csv(details_file, mode='a')