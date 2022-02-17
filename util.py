#! /usr/bin/env python3

import numpy as np
import pandas as pd
import pickle
import os
from scipy.special import logsumexp
import matplotlib.pyplot as plt

def save_result(file_path, obj):
    sfp = file_path.split('/')
    adrs = '.'
    for i in sfp[:-1]:
        if i not in os.listdir(adrs):
            os.mkdir(os.path.join(adrs, i))
        adrs = os.path.join(adrs, i)
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

def ME_test(node, dataset, gene_names):
    # Outputs:
    #       df:         rows are pairs of genes,
    #                   first column is the freq. of mut. in gene1
    #                   second column is the freq. of mut. in gene2
    #                   third column is the freq. of mut. in both genes
    #                   forth column is log_p_value of the obs. if the genes are not correlated
    #       ME_score:   Averaged ME_score ( ME = f(1&2)/(f(1)*f(2)) )
    #       avg_p:      Mean of the p_values
    n = len(node.genes)
    n_tumors = dataset.shape[0]
    mut_freqs = []
    if n > 1:
        df_indices=[]
        for _idx1 in range(n-1):
            for _idx2 in range(_idx1+1, n):
                pair = [node.genes[_idx1], node.genes[_idx2]]
                n_1 = np.sum(dataset[:,pair[0]])
                f_1 = n_1/n_tumors
                n_2 = np.sum(dataset[:,pair[1]])
                f_2 = n_2/n_tumors
                n_p = np.sum(dataset[:,pair[0]]*dataset[:,pair[1]])
                f_p = n_p/n_tumors
                log_p_i = np.zeros(n_p+1)
                if f_1*f_2 == 0:
                    log_p = 0
                else:
                    if n_1 <= n_2:
                        for i in range(n_p+1):
                            log_p_i[i] = log_nCr(n_1, i)+(i*np.log(f_2))+((n_1-i)*np.log(1-f_2))
                    else:
                        for i in range(n_p+1):
                            log_p_i[i] = log_nCr(n_2, i)+(i*np.log(f_1))+((n_2-i)*np.log(1-f_1))
                    log_p = logsumexp(log_p_i)
                mut_freqs.append([f_1, f_2, f_p, log_p])
                df_indices.append('{}, {}'.format(gene_names[pair[0]], gene_names[pair[1]]))
        df = pd.DataFrame(
            data=mut_freqs,
            columns=[
                'Gene1_Mut_Freq',
                'Gene2_Mut_Freq',
                'Sim_Mut_Freq',
                'Log_p_value'
                ],
            index=df_indices)
        ME_scores = []
        for i in mut_freqs:
            if i[1]*i[0]>0:
                ME_scores.append(i[2]/(i[1]*i[0]))
        if len(ME_scores)>0:
            ME_score = np.mean(ME_scores)
        else:
            ME_score = 0
        mut_freqs = np.array(mut_freqs)
        avg_p = np.mean(np.exp(mut_freqs[:,3]))
        return(df, ME_score, avg_p)
    else:
        return()

def PR_test(node, dataset, gene_names=None):
    # PR_score should be smaller than 1, ideally 0
    n_tumors = dataset.shape[0]
    mutated_in_node = np.sum(dataset[:,node.genes], axis=1)>0
    mutated_in_parent = np.sum(dataset[:,node.parent.genes], axis=1)>0
    mutated_only_in_node = mutated_in_node * (1-mutated_in_parent)
    
    n_parent = np.sum(mutated_in_parent)
    n_node = np.sum(mutated_in_node)
    if n_parent == n_tumors:
        return(None)
    elif n_node == 0:
        return(1, 0)
    else:
        n_only_node = np.sum(mutated_only_in_node)
        PR_score = (n_tumors*n_only_node)/(n_node*(n_tumors-n_parent))
        if PR_score<1:
            # Calculating P-value:
            p = n_node/n_tumors
            m = n_tumors-n_parent
            log_p_i = np.zeros(n_only_node+1)
            for i in range(n_only_node+1):
                log_p_i[i] = log_nCr(m, i)+(i*np.log(p))+((m-i)*np.log(1-p))
            log_p = logsumexp(log_p_i)
        else:
            log_p = 0
        return(PR_score, np.exp(log_p))

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