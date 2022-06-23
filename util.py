#! /usr/bin/env python3

import numpy as np
import pandas as pd
import pickle
import os
from scipy.special import logsumexp
import matplotlib.pyplot as plt

####### ------------------------------------------- #######
####### ----- Main functions, frequently used ----- #######
####### ------------------------------------------- #######

def save_result(file_path, obj):
    # sfp = file_path.split('/')
    # adrs = '.'
    # for i in sfp[:-1]:
    #     if i not in os.listdir(adrs):
    #         os.mkdir(os.path.join(adrs, i))
    #     adrs = os.path.join(adrs, i)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_result(result_file, data_file, gen_file=None):
    df_input = pd.read_csv(data_file, delimiter=',', index_col=0, comment='#')
    gene_names = list(df_input.columns)
    dataset = np.array(df_input, dtype=bool)
    with open(result_file, 'rb') as f:
        samples = pickle.load(f)
    if gen_file is not None:
        with open(gen_file, 'rb') as f:
            gen_progmo = pickle.load(f)
    else:
        gen_progmo = None
    return(dataset, gene_names, samples, gen_progmo)

def log_nCr(n, m):
    result = 0
    for i in range(m):
        result = result + np.log(n-i) - np.log(m-i)
    return(result)

def perfect_ME(node, dataset):

    # Returns True if the node does not violate ME
    n = len(node.genes)
    ME = True
    if n>1:
        for _idx1 in range(n-1):
            for _idx2 in range(_idx1+1, n):
                if np.sum(dataset[:,node.genes[_idx1]]*dataset[:,node.genes[_idx2]])>0:
                    ME = False
    return(ME)

def perfect_PR(node, dataset):

    # Returns True if the node does not violate PR
    PR = True
    if not(node.is_root) and not(node.is_ps):
        if not(node.parent.is_root):
            mutated_in_node = np.sum(dataset[:,node.genes], axis=1)>0
            mutated_in_parent = np.sum(dataset[:,node.parent.genes], axis=1)>0
            mutated_only_in_node = mutated_in_node * (1-mutated_in_parent)
            if np.sum(mutated_only_in_node)>0:
                PR = False
    return(PR)

def ME_test(node, dataset):

    # Averaged over pairs of genes (X,Y):
    # Assuming n_X >= n_Y, we go over the tumors with mutated Y and calculate
    # the ratio A/B, where:
    # A:    probability of this many or FEWER mutations in X under Null hypothesis
    # B:    probability of this many or MORE mutations in X under Null hypothesis
    n = len(node.genes)
    n_tumors = dataset.shape[0]
    if n > 1:
        mi_log_p_values = []
        me_log_p_values = []
        for _idx1 in range(n-1):
            for _idx2 in range(_idx1+1, n):
                pair = [node.genes[_idx1], node.genes[_idx2]]
                n_1, n_2 = np.sort([np.sum(dataset[:,pair[0]]),np.sum(dataset[:,pair[1]])])
                f_1 = n_1/n_tumors
                f_2 = n_2/n_tumors
                n_p = np.sum(dataset[:,pair[0]]*dataset[:,pair[1]])
                f_p = n_p/n_tumors
                if n_1 == 0: # n_1 is the smaller number
                    me_log_p = 0 # set the p_value to 1
                    mi_log_p = 0
                else:
                    log_p_i = np.zeros(n_1+1)
                    for i in range(n_1+1):
                        log_p_i[i] = log_nCr(n_1, i)+(i*np.log(f_2))+((n_1-i)*np.log(1-f_2))
                    me_log_p = logsumexp(log_p_i[:n_p+1])
                    mi_log_p = logsumexp(log_p_i[n_p:])
                me_log_p_values.append(me_log_p)
                mi_log_p_values.append(mi_log_p)
        ME_score = np.mean(np.exp(me_log_p_values))/np.mean(np.exp(mi_log_p_values))
        ME_p = np.mean(np.exp(me_log_p_values))
        return(ME_score, ME_p)
    else:
        return()

def PR_test(node, dataset):
    # the ratio A/B, where: (note that it's NOT in log-scale)
    # A:    probability of this many or FEWER mutations in the tumors with
    #       healthy parents under Null hypothesis
    # B:    probability of this many or MORE mutations in the tumors with
    #       healthy parents under Null hypothesis
    n_tumors = dataset.shape[0]
    mutated_in_node = np.sum(dataset[:,node.genes], axis=1)>0
    mutated_in_parent = np.sum(dataset[:,node.parent.genes], axis=1)>0
    mutated_only_in_node = mutated_in_node * (1-mutated_in_parent)
    mutated_only_in_parent = (1-mutated_in_node) * (mutated_in_parent)

    n_parent = np.sum(mutated_in_parent)
    n_node = np.sum(mutated_in_node)
    n_only_node = np.sum(mutated_only_in_node)
    n_only_parent = np.sum(mutated_only_in_parent)

    m_forward = n_tumors-n_parent
    p_forward = n_node/n_tumors
    log_p_i_forward = np.zeros(n_only_node+1)

    for i in range(n_only_node+1):
        log_p_i_forward[i] = log_nCr(m_forward, i)+(i*np.log(p_forward))+((m_forward-i)*np.log(1-p_forward))

    log_p_forward = logsumexp(log_p_i_forward)

    m_backward = n_tumors-n_node
    p_backward = n_parent/n_tumors
    log_p_i_backward = np.zeros(n_only_parent+1)
    for i in range(n_only_parent+1):
        log_p_i_backward[i] = log_nCr(m_backward, i)+(i*np.log(p_backward))+((m_backward-i)*np.log(1-p_backward))
    log_p_backward = logsumexp(log_p_i_backward)

    # old test
    # if n_only_node == 0:
    #         PR_score = 0
    # else:
    #     PR_score = (n_tumors*n_only_node)/(n_node*(n_tumors-n_parent))
    # if PR_score<1 and n_node>0:
    #     p = n_node/n_tumors
    #     m = n_tumors-n_parent
    #     log_p_i = np.zeros(n_only_node+1)
    #     for i in range(n_only_node+1):
    #         log_p_i[i] = log_nCr(m, i)+(i*np.log(p))+((m-i)*np.log(1-p))
    #     log_p = logsumexp(log_p_i)
    # else:
    #     log_p = 0
    # return(PR_score, np.exp(log_p))
    #

    PR_score = np.exp(log_p_forward-log_p_backward)
    PR_p = np.exp(log_p_forward)
    return(PR_score, PR_p)

def Geweke(chain, first_proportion=0.1, second_proporiton=0.5, threshold=2):
    ''' The convergence is achieved if Z score is below threshold (the standard value is 2) '''
    n_samples = len(chain)
    if n_samples < 100:
        return(False, np.inf)
    else:
        a = chain[:int(first_proportion*n_samples)]
        b = chain[int(second_proporiton*n_samples):]
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        var_a = np.var(a)
        var_b = np.var(b)
        z_score = np.abs((mean_a-mean_b)/(np.sqrt(var_a+var_b)))
        result = z_score<threshold
        return(result, z_score)

def Gelman_Rubin(set_of_chains, burn_in=0.5, threshhold=1.2):
    ''' The convergence is achieved if Potential Scale Reduction Factor is close to 1, e.g., below 1.2 or 1.1 '''
    idx = int(len(set_of_chains[0])*(1-burn_in))
    means = np.zeros(len(set_of_chains))
    variances = np.zeros(len(set_of_chains))
    N = len(set_of_chains[0])-idx
    for _idx, chain in enumerate(set_of_chains):
        means[_idx] = np.mean(chain[idx:])
        variances[_idx] = np.var(chain[idx:],ddof=1)
    overall_mean = np.mean(means)
    N = len(set_of_chains[0])-idx
    M = len(set_of_chains)
    B = (N/(M-1))*np.sum((means-overall_mean)**2)
    W = np.mean(variances)
    V_hat = ((N-1)/N)*W + ((M+1)/(M*N))*B
    PSRF = V_hat/W
    result = PSRF < threshhold
    return(result, PSRF)

####### ------------------------------------------- #######
####### ---- Functions used for postprocessing ---- #######
####### ------------------------------------------- #######

def dataset_visualization(input, output):
    # input: df or path to csv
    # output: path to pdf file

    def fig_generator(ax, data, gene_names, bar_color='black', text_color='black', print_names=False):
        
        n_genes = data.shape[1]
        n_patients = data.shape[0]
        freqs = np.mean(data, axis=0)
        indices = np.argsort(freqs)[::-1]
        sorted_names = [gene_names[i] for i in indices]
        sorted_data = data[:, indices]
        
        ax = iterative_plotter(ax, sorted_data, bar_color, left=0)
        ax.set_xlim(0, n_patients)
        ax.set_xticks([])
        ax.set_ylim(0, n_genes)

        if print_names:
            ax.set_yticks([])
            for gene_idx in range(n_genes):
                ax.text(n_patients*0.5, gene_idx+0.5, gene_names[gene_idx], c=text_color, va='center', ha='center')
        else:
            ax.set_yticks(np.arange(n_genes)+0.5)
            #ax.set_yticklabels(sorted_names[::-1], fontsize=4*len(gene_names))
            ax.set_yticklabels(sorted_names[::-1], fontsize=30)
            #ax.set_yticklabels(sorted_names[::-1])
        
            
        return(ax)

    def iterative_plotter(ax, sorted_data, bar_color, left=0):
        
        run_for_green = False
        run_for_blue = False
        
        n_patients = sorted_data.shape[0]
        y = sorted_data.shape[1]

        black_width = np.sum(sorted_data, axis=0)[0]
        #print(black_width)
        white_width = n_patients - black_width
        #print(white_width)
        
        ax.barh(y-1, black_width, left=left, height=1, align='edge', color=bar_color)
        
        ax.barh(y-1, white_width, left=left+black_width, height=1, align='edge', color='white')
        
        #ax.barh(0, black_width, left=left, height=1*(y-1), align='edge', color='g')

        #ax.barh(0, white_width, left=left+black_width, height=1*(y-1), align='edge', color='b')

        if y>1:
            _g = sorted_data[:, 0] == 1
            if np.sum(_g)>0:
                run_for_green = True
            _b = sorted_data[:, 0] == 0
            if np.sum(_b)>0:
                run_for_blue = True
        
        if run_for_green:
            green_data = sorted_data[_g, 1:]
            ax = iterative_plotter(ax, green_data, bar_color, left=left)
        
        if run_for_blue:
            blue_data = sorted_data[_b, 1:]
            ax = iterative_plotter(ax, blue_data, bar_color, left=left+black_width)
        
        return(ax)

    if type(input) == str:
        df_input = pd.read_csv(input, delimiter=',', index_col=0, comment='#')
    elif type(input) == pd.core.frame.DataFrame:
        df_input = input
    else:
        print('The input has to be a pandas DataFrame or the path to a csv file including the DataFrame!')
        return()
    gene_names = list(df_input.columns)
    data = np.array(df_input)

    fig, ax = plt.subplots(figsize=(len(gene_names), len(gene_names)))
    ax = fig_generator(ax, data, gene_names, bar_color='black', print_names=False)
    plt.tight_layout()
    plt.savefig(output)

def PR_test_gbg(node, dataset, gene_names=None, mode='classic'):
    # PR_score performed gene-by-gene (similar to ME_test)
    if gene_names is None:
        gene_names = [str(i) for i in range(dataset.shape[1])]
    n_tumors = dataset.shape[0]
    mut_freqs = []
    df_indices = []
    n = len(node.genes)
    m = len(node.parent.genes)
    if mode=='classic':
        for _idx1 in range(n):
            for _idx2 in range(m):
                mutated_in_gene = dataset[:,node.genes[_idx1]]
                mutated_in_parent = dataset[:,node.parent.genes[_idx2]]
                mutated_only_in_node = mutated_in_gene * (1-mutated_in_parent)
                n_parent = np.sum(mutated_in_parent)
                n_node = np.sum(mutated_in_gene)
                n_only_node = np.sum(mutated_only_in_node)
                if n_only_node == 0:
                    f_1 = 0
                    f_2 = n_node/n_tumors
                    PR_score = 0
                else:
                    f_1 = n_only_node/(n_tumors-n_parent)
                    f_2 = n_node/n_tumors
                    PR_score = f_1/f_2
                if PR_score<1 and n_node>0:
                # Calculating P-value:
                    p = n_node/n_tumors
                    m = n_tumors-n_parent
                    log_p_i = np.zeros(n_only_node+1)
                    for i in range(n_only_node+1):
                        log_p_i[i] = log_nCr(m, i)+(i*np.log(p))+((m-i)*np.log(1-p))
                    log_p = logsumexp(log_p_i)
                else:
                    log_p = 0
                mut_freqs.append([f_1, f_2, PR_score, log_p])
                df_indices.append('{} to {}'.format(gene_names[node.parent.genes[_idx2]], gene_names[node.genes[_idx1]]))
    elif mode=='new':
        for _idx1 in range(n):
            for _idx2 in range(m):
                mutated_in_gene = dataset[:,node.genes[_idx1]]
                mutated_in_parent = dataset[:,node.parent.genes[_idx2]]
                mutated_only_in_node = mutated_in_gene * (1-mutated_in_parent)
                n_parent = np.sum(mutated_in_parent)
                n_node = np.sum(mutated_in_gene)
                n_only_node = np.sum(mutated_only_in_node)
                p = n_node/n_tumors
                m = n_tumors-n_parent
                log_p_i = np.zeros(m+1)
                for i in range(m+1):
                    log_p_i[i] = log_nCr(m, i)+(i*np.log(p))+((m-i)*np.log(1-p))
                PR_score = np.exp(logsumexp(log_p_i[:n_only_node+1])-logsumexp(log_p_i[n_only_node:]))
                mut_freqs.append([n_only_node/(n_tumors-n_parent), n_node/n_tumors, PR_score, PR_score])
                df_indices.append('{} to {}'.format(gene_names[node.parent.genes[_idx2]], gene_names[node.genes[_idx1]]))
    ######
    mutated_in_node = np.sum(dataset[:,node.genes], axis=1)>0
    mutated_in_parent = np.sum(dataset[:,node.parent.genes], axis=1)>0
    mutated_only_in_node = mutated_in_node * (1-mutated_in_parent)
    
    n_parent = np.sum(mutated_in_parent)
    n_node = np.sum(mutated_in_node)
    n_only_node = np.sum(mutated_only_in_node)
    if mode=='classic':
        if n_only_node == 0:
            PR_score = 0
            f_1 = 0
            f_2 = n_node/n_tumors
        else:
            f_1 = n_only_node/(n_tumors-n_parent)
            f_2 = n_node/n_tumors
            PR_score = f_1/f_2
        if PR_score<1 and n_node>0:
            p = n_node/n_tumors
            m = n_tumors-n_parent
            log_p_i = np.zeros(n_only_node+1)
            for i in range(n_only_node+1):
                log_p_i[i] = log_nCr(m, i)+(i*np.log(p))+((m-i)*np.log(1-p))
            log_p = logsumexp(log_p_i)
        else:
            log_p = 0
        mut_freqs.append([f_1, f_2, PR_score, log_p])
        df_indices.append('Complete node')
    elif mode=='new':
        m = n_tumors-n_parent
        p = n_node/n_tumors
        log_p_i = np.zeros(m+1)
        for i in range(m+1):
            log_p_i[i] = log_nCr(m, i)+(i*np.log(p))+((m-i)*np.log(1-p))
        PR_score = np.exp(logsumexp(log_p_i[:n_only_node+1])-logsumexp(log_p_i[n_only_node:]))
        mut_freqs.append([n_only_node/(n_tumors-n_parent), n_node/n_tumors, PR_score, PR_score])
        df_indices.append('Complete node')
    #######
    df = pd.DataFrame(
        data=mut_freqs,
        columns=[
            'Child_Mut_Freq_in_Healthy_Parent',
            'Child_Mut_Freq',
            'PR_score',
            'Log_p_value'
            ],
        index=df_indices)
    return(df, np.mean(df.iloc[:-1, 2]), np.mean(np.exp(df.iloc[:-1,3])))

def minimum_mutation_freq(n_tumors, n_max=None, p_th=0.5, mode='PR-with-HighlyMutated'):
    if mode == 'PR-with-HighlyMutated':
        # If all the tumors with mutation in THE GENE also have mutation in the most-
        # -highly mutated gene (perfect progression), the p_value of PR should be less -
        # - than the threshold (of around 0.5)
        n_min = int(np.ceil(n_tumors*(1-(p_th)**(1/(n_tumors-n_max)))))
    elif mode == 'PR-with-eachother':
        # If two genes are always mutated together (perfect progression), the p_value of -
        # - PR should be less than the threshold (of around 0.02)
        for i in range(n_tumors+1):
            if (n_tumors-i)*np.log(1-(i/n_tumors))<np.log(p_th):
                n_min = i
                break
    elif mode == 'ME-with-HighlyMutated':
        # If we have perfect ME with the most-highly mutated gene (perfect ME), the -
        # - ME signal's p_value should be less than the threshold (of around  )
        n_min = int(np.ceil((np.log(p_th))/(np.log(1-(n_max/n_tumors)))))
    elif mode == 'ME-with-eachother':
        # If two genes are never mutated together (perfect ME), the p_value of -
        # - ME should be less than the threshold (of around )
        for i in range(n_tumors+1):
            if i*np.log(1-(i/n_tumors)) <= np.log(p_th):
                n_min = i
                break
    return(n_min)

def to_json(node, dataset, gene_names):
    output = {}
    output['genes'] = ','.join(gene_names[i] for i in node.genes)
    output['l'] = -np.log(node.f)
    if len(node.genes)>1:
        _, ME_score, avg_p = ME_test(node, dataset, gene_names)
        output['ME_score'] = float("%.3f" %ME_score)
        output['ME_p'] = float("%.4f" %avg_p)
    if node.parent is not None and not(node.parent.is_root):
        if np.sum(np.sum(dataset[:,node.genes], axis=1)>0)<dataset.shape[0]:
            PR_score, p_value = PR_test(node, dataset, gene_names)
            output['PR_score'] = float("%.3f" %PR_score)
            output['PR_p'] = float("%.4f" %p_value)
    if len(node.children)>0:
        output['children'] = [to_json(child, dataset, gene_names) for child in node.children]
    return(output)

def to_newick(node, dataset=None, gene_names=None):
    def _parse_json(js):
        try:
            newick = '-'.join(js['genes'].split(','))
            if len(newick)==0:
                newick = 'Root'
        except KeyError:
            newick = ''
        if 'l' in js:
            newick += ':' + '%.3f'%js['l']
        if len(js.keys())>3:
            newick += '[&&NHX'
            for _key in js.keys():
                if _key not in ['genes','l','children']:
                    newick += ':' + _key + '=' + str(js[_key])
            newick += ']'
        if 'children' in js:
            info = []
            for child in js['children']:
                info.append(_parse_json(child))
            info = ','.join(info)
            newick = '(' + info + ')' + newick
        return(newick)
    if type(node) == dict:
        # Used to convert json to newick
        js = node
    else:
        js = to_json(node, dataset, gene_names)
    return(_parse_json(js)+';')

def SR_analysis(node, dataset, gene_names, log_p_th = 0):
    # Analysis of possible dependencies among the children of a node
    ME_rep = []
    MI_rep = []
    n_tumors = dataset.shape[0]
    n_children = len(node.children)
    for i in range(n_children):
        node1 = node.children[i]
        mutated_in_node1 = np.sum(dataset[:,node1.genes], axis=1)>0
        n_1 = np.sum(mutated_in_node1)
        f_1 = n_1/n_tumors
        for j in range(i+1, n_children):
            node2 = node.children[j]
            mutated_in_node2 = np.sum(dataset[:,node2.genes], axis=1)>0
            n_2 = np.sum(mutated_in_node2)
            f_2 = n_2/n_tumors
            mutated_in_both = mutated_in_node1 * mutated_in_node2
            n_p = np.sum(mutated_in_both)
            f_p = n_p/n_tumors
            if f_p <= f_1*f_2:
                # There is a chance of mutual exclusivity
                log_p_i = np.zeros(n_p+1)
                if n_1 <= n_2:
                    for i in range(n_p+1):
                        log_p_i[i] = log_nCr(n_1, i)+(i*np.log(f_2))+((n_1-i)*np.log(1-f_2))
                else:
                    for i in range(n_p+1):
                        log_p_i[i] = log_nCr(n_2, i)+(i*np.log(f_1))+((n_2-i)*np.log(1-f_1))
                log_p = logsumexp(log_p_i)
                if log_p <= log_p_th:
                    print('Significant Mutual Exclusivity found, with log_p %.3f'%log_p)
                    ME_rep.append({
                        'node1': ','.join(gene_names[_i] for _i in node1.genes),
                        'node2': ','.join(gene_names[_i] for _i in node2.genes),
                        'f_1': f_1,
                        'f_2': f_2,
                        'f_p': f_p,
                        'log_p': log_p
                        })
            else:
                # There is a chance of mutual inclusivity
                log_p_i = []
                if n_1 <= n_2:
                    for i in range(n_p, n_1+1):
                        log_p_i.append(log_nCr(n_1, i)+(i*np.log(f_2))+((n_1-i)*np.log(1-f_2)))
                else:
                    for i in range(n_p, n_2+1):
                        log_p_i.append(log_nCr(n_2, i)+(i*np.log(f_1))+((n_2-i)*np.log(1-f_1)))
                log_p = logsumexp(log_p_i)
                if log_p <= log_p_th:
                    print('Significant Mutual Inclusivity found, with log_p %.3f'%log_p)
                    MI_rep.append({
                        'node1': ','.join(gene_names[_i] for _i in node1.genes),
                        'node2': ','.join(gene_names[_i] for _i in node2.genes),
                        'f_1': f_1,
                        'f_2': f_2,
                        'f_p': f_p,
                        'log_p': log_p
                        })
    return(ME_rep, MI_rep)