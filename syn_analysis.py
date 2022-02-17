#! /usr/bin/env python3

import numpy as np
import pandas as pd
import os
import time
import pickle
import matplotlib.pyplot as plt
from oncotree import OncoTree, OncoNode
from util import save_result, Geweke, Gelman_Rubin
from itertools import product
from multiprocessing import Pool
from copy import deepcopy

def initialize(dataset, coeff=-1):
    if coeff<0:
        sample = OncoTree.star_from_dataset(
            dataset,
            single_error=single_error,
            error_estimation = True
            )
    else:
        sample = OncoTree.from_dataset(
            dataset,
            coeff=coeff,
            passenger_threshold=0.0001,
            single_error=single_error,
            error_estimation = True
            )
    p = sample.likelihood(dataset) + sample.prior()
    return(sample, p)

def run_race(args):
    dataset = args[-1]
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

def fit_oncotree(dataset, print_running_status=False):
    st = time.time()
    random_seeds = np.arange(1, n0+1)
    sample_list = []
    p_list = []
    for _i in range(n0):
        _sample, _p = initialize(dataset)
        sample_list.append(_sample)
        p_list.append(_p)
    overall_posterior_array = np.empty(shape=(n0, 0))
    overall_updates_array = np.empty(shape=(n0, 0))
    details_dict = {} #Keeping number of proposed samples, novel samples and accepted samples, per move type
    for _training_race in range(n1+1):
        args_list = [(sample_list[i], random_seeds[i], p_list[i], dataset) for i in range(len(sample_list))]
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
        sample_list = []
        p_list = []
        current_posteriors = [result_tuple[1] for result_tuple in results]
        bcp_idx = np.argmax(current_posteriors) # Best Current Posterior
        sample_list.append(deepcopy(results[bcp_idx][0]))
        p_list.append(deepcopy(results[bcp_idx][1]))
        best_posteriors = [result_tuple[3] for result_tuple in results]
        bop_idx = np.argmax(best_posteriors) # Best Overall Posterior
        for _i in range(0, n0-1):
            sample_list.append(deepcopy(results[bop_idx][2]))
            p_list.append(deepcopy(results[bcp_idx][3]))
        #sample, current_posterior, best_sample, best_posterior, posteriors_list, n_updates
    return(results[bop_idx][2], overall_posterior_array, overall_updates_array, details_dict)

#--------------------------------------
#-------- Setting the params ----------
#--------------------------------------
n0 = 10 #n_chains
n1 = 0 #n_mixing
n2 = 100000 #n_samples
single_error = False
#--------------------------------------
#-------- Running the chains ----------
#--------------------------------------

if 'synthetic' not in os.listdir('result'):
    os.mkdir('result/synthetic')
if single_error:
    res_dir = 'result/synthetic/one_error'
    if 'one_error' not in os.listdir('result/synthetic'):
        os.mkdir('result/synthetic/one_error')
else:
    res_dir = 'result/synthetic/two_errors'
    if 'two_errors' not in os.listdir('result/synthetic'):
        os.mkdir('result/synthetic/two_errors')

modes = ['linear', 'tree', 'mixture']
#modes = ['linear']
#modes = ['tree']
#modes = ['mixture']

for mode in modes:
    if mode not in os.listdir(res_dir):
        os.mkdir(os.path.join(res_dir, mode))

    dataset_list = [filename for filename in os.listdir('data/synthetic/{}'.format(mode)) if filename.endswith('.csv')]
    datasets = [
        np.array(pd.read_csv(os.path.join('data/synthetic/{}'.format(mode), filename), delimiter=',', index_col=0, comment='#'), dtype=bool) for filename in dataset_list
    ]

    gene_names = list(pd.read_csv(os.path.join('data/synthetic/{}'.format(mode), dataset_list[0]), delimiter=',', index_col=0, comment='#').columns)
    coeffs = [-1 for _i in range(n0)]

    result_list = [os.path.join('{}/{}'.format(res_dir, mode), filename[:-4]) for filename in dataset_list]

    for i, dataset in enumerate(datasets):
        output_folder = result_list[i]
        if dataset_list[i][:-4] not in os.listdir('{}/{}'.format(res_dir, mode)):
            os.mkdir(output_folder)
        start_time = time.time()
        progmo, posterior_array, updates_array, details_dict = fit_oncotree(dataset)
        spent_time = time.time()-start_time
        save_result(os.path.join(output_folder, 'progmo.pkl'), progmo)
        df_posterior_array = pd.DataFrame(posterior_array)
        df_posterior_array.to_csv(os.path.join(output_folder, 'posteriors.csv'))

        report = '# Dataset %s \n' % dataset_list[i]
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

        report += '# BEFORE FINE-TUNING the error parameters:\n'

        star_llh = star_tree.likelihood(dataset)
        report += '# Star tree log likelihood: %.4f \n' %star_llh
        # star_pri = star_tree.prior()
        # report += '# Star tree log prior: %.4f \n' %star_pri
        # report += '# Star tree log posterior: %.4f \n' %(star_pri+star_llh)

        best_llh = progmo.likelihood(dataset)
        report += '# Best sample log likelihood: %.4f \n' %best_llh
        # best_pri = progmo.prior()
        # report += '# Best sample log prior: %.4f \n' %best_pri
        # report += '# Best sample log posterior: %.4f \n' %(best_pri+best_llh)
        report += '# Best sample epsilon: %.5f \n' %(progmo.pfp)
        report += '# Best sample delta: %.5f \n' %(progmo.pfn)

        report += '# AFTER FINE-TUNING the error parameters:\n'

        star_tree, _ = star_tree.fit_error_params(dataset)
        star_llh = star_tree.likelihood(dataset)
        report += '# Star tree log likelihood: %.4f \n' %star_llh
        # star_pri = star_tree.prior()
        # report += '# Star tree log prior: %.4f \n' %star_pri
        # report += '# Star tree log posterior: %.4f \n' %(star_pri+star_llh)

        progmo, _ = progmo.fit_error_params(dataset)
        best_llh = progmo.likelihood(dataset)
        report += '# Best sample log likelihood: %.4f \n' %best_llh
        # best_pri = progmo.prior()
        # report += '# Best sample log prior: %.4f \n' %best_pri
        # report += '# Best sample log posterior: %.4f \n' %(best_pri+best_llh)
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
        
        print('done with {} - {}'.format(mode, dataset_list[i]))