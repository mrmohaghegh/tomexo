import numpy as np
import pandas as pd
from subprocess import check_call
from scipy.special import logsumexp

def log_nCr(n, m):
    result = 0
    for i in range(m):
        result = result + np.log(n-i) - np.log(m-i)
    return(result)

def plot_annotated_mhn(df, annotation_df, dot_file='tmp.dot', fig_file='tmp.png'):
    gene_names = list(df.index)
    txt = 'digraph tree {\n'
    for i, gene in enumerate(gene_names):
        label = '<%s>'%(gene)
        txt += '    Node%i [label=%s, shape=box, style=\"rounded, filled\", fillcolor=grey95, color=black];\n'%(i, label)
    for i, gene_1 in enumerate(gene_names):
        for k, gene_2 in enumerate(gene_names[i+1:]):
            j = k+i+1
            if np.round(df.loc[gene_1,gene_2], decimals=1)!=1:
                if np.round(df.loc[gene_2,gene_1], decimals=1)!=1:
                    # bi-directional relation!
                    if df.loc[gene_1,gene_2]<1:
                        arrow_type= 'oinv'
                    else:
                        arrow_type = 'normal'
                    txt+='    Node%i -> Node%i [arrowhead=%s, arrowtail=%s, dir=both];\n'%(j, i, arrow_type, arrow_type)
                else:
                    # single-directional!
                    if df.loc[gene_1,gene_2]<1:
                        arrow_type= 'oinv'
                    else:
                        arrow_type = 'normal'
                    if annotation_df.loc[gene_1,gene_2]==0:
                        edge_color = 'red'
                    elif annotation_df.loc[gene_1,gene_2]==2:
                        edge_color = 'black'
                    elif annotation_df.loc[gene_1,gene_2]==1:
                        edge_color = 'black'
                    else:
                        edge_color = 'yellow'
                    txt+='    Node%i -> Node%i [arrowhead=%s, color=%s];\n'%(j, i, arrow_type, edge_color)
            else:
                if np.round(df.loc[gene_2,gene_1], decimals=1)!=1:
                    #single-directional but in the other way!
                    if df.loc[gene_2,gene_1]<1:
                        arrow_type= 'oinv'
                    else:
                        arrow_type = 'normal'
                    if annotation_df.loc[gene_2,gene_1]==0:
                        edge_color = 'red'
                    elif annotation_df.loc[gene_2,gene_1]==2:
                        edge_color = 'black'
                    elif annotation_df.loc[gene_2,gene_1]==1:
                        edge_color = 'black'
                    else:
                        edge_color = 'yellow'
                    txt+='    Node%i -> Node%i [arrowhead=%s, color=%s];\n'%(i, j, arrow_type, edge_color)

    txt += '}'
    with open(dot_file, 'w') as f:
        f.write(txt)
    if fig_file.endswith('.pdf'):
        check_call(['dot', '-Kcirco', '-Tpdf',dot_file,'-o',fig_file])
    else:
        check_call(['dot', '-Kcirco', '-Tpng',dot_file,'-o',fig_file])
    return(txt)

def plot_mhn(df, dot_file='tmp.dot', fig_file='tmp.png'):
    gene_names = list(df.index)
    txt = 'digraph tree {\n'
    for i, gene in enumerate(gene_names):
        label = '<%s<br/><font color=\'Blue\' POINT-SIZE=\'12\'> %.2f </font>>'%(gene,df.loc[gene,gene])
        txt += '    Node%i [label=%s, shape=box, style=\"rounded, filled\", fillcolor=grey95, color=black];\n'%(i, label)
    for i, gene_1 in enumerate(gene_names):
        for k, gene_2 in enumerate(gene_names[i+1:]):
            j = k+i+1
            if np.round(df.loc[gene_1,gene_2], decimals=1)!=1:
                if np.round(df.loc[gene_2,gene_1], decimals=1)!=1:
                    # bi-directional relation!
                    if df.loc[gene_1,gene_2]<1:
                        arrow_type= 'oinv'
                        text_color = '\'Red\''
                    else:
                        arrow_type = 'onormal'
                        text_color = '\'Green\''
                    txt+='    Node%i -> Node%i [headlabel=<<font color=%s POINT-SIZE=\'12\'> %.1f </font>>, taillabel=<<font color=\'Red\' POINT-SIZE=\'12\'> %.1f </font>>, arrowhead=%s, arrowtail=%s, dir=both];\n'%(j, i, text_color,  df.loc[gene_1,gene_2], df.loc[gene_2,gene_1], arrow_type, arrow_type)
                else:
                    # single-directional!
                    if df.loc[gene_1,gene_2]<1:
                        arrow_type= 'oinv'
                        text_color = '\'Red\''
                    else:
                        arrow_type = 'onormal'
                        text_color = '\'Green\''
                    txt+='    Node%i -> Node%i [headlabel=<<font color=%s POINT-SIZE=\'12\'> %.1f </font>>, arrowhead=%s];\n'%(j, i, text_color, df.loc[gene_1,gene_2], arrow_type)
            else:
                if np.round(df.loc[gene_2,gene_1], decimals=1)!=1:
                    #single-directional but in the other way!
                    if df.loc[gene_2,gene_1]<1:
                        arrow_type= 'oinv'
                        text_color = '\'Red\''
                    else:
                        arrow_type = 'onormal'
                        text_color = '\'Green\''
                    txt+='    Node%i -> Node%i [headlabel=<<font color=%s POINT-SIZE=\'12\'> %.1f </font>>, arrowhead=%s];\n'%(i, j, text_color, df.loc[gene_2,gene_1], arrow_type)

    txt += '}'
    with open(dot_file, 'w') as f:
        f.write(txt)
    if fig_file.endswith('.pdf'):
        check_call(['dot', '-Kcirco', '-Tpdf',dot_file,'-o',fig_file])
    else:
        check_call(['dot', '-Kcirco', '-Tpng',dot_file,'-o',fig_file])
    return(txt)

def ME(df, gene_set):
    n_tumors, n_genes = df.shape
    genes = df.columns
    n = len(gene_set)
    me_log_p_values = []
    me_scores = []
    for _idx1 in range(n-1):
        for _idx2 in range(_idx1+1, n):
            pair = [gene_set[_idx1], gene_set[_idx2]]
            n_1, n_2 = np.sort(df.loc[:,pair].sum(axis=0))
            f_1 = n_1/n_tumors
            f_2 = n_2/n_tumors
            n_p = (df.loc[:,pair[0]]*df.loc[:,pair[1]]).sum()
            f_p = n_p/n_tumors
            if n_1 == 0: # n_1 is the smaller number
                me_log_p = 0 # set the p_value to 1
                me_score = 1
            else:
                me_score = f_p/(f_1*f_2)
                log_p_i = np.zeros(n_1+1)
                for i in range(n_1+1):
                    log_p_i[i] = log_nCr(n_1, i)+(i*np.log(f_2))+((n_1-i)*np.log(1-f_2))
                me_log_p = logsumexp(log_p_i[:n_p+1])
            me_log_p_values.append(me_log_p)
            me_scores.append(me_score)
    ME_p = np.mean(np.exp(me_log_p_values))
    ME_s = np.mean(me_scores)
    return(ME_s, ME_p)

def PR(df, gene_set_parent, gene_set_child):
    n_tumors, n_genes = df.shape
    genes = df.columns
    mutated_in_node = df.loc[:,gene_set_child].sum(axis=1)>0
    mutated_in_parent = df.loc[:,gene_set_parent].sum(axis=1)>0
    mutated_only_in_node = mutated_in_node * (1-mutated_in_parent)
    mutated_only_in_parent = (1-mutated_in_node) * (mutated_in_parent)

    n_parent = np.sum(mutated_in_parent)
    n_node = np.sum(mutated_in_node)
    n_only_node = np.sum(mutated_only_in_node)
    n_only_parent = np.sum(mutated_only_in_parent)

    if n_node == 0:
        log_p_forward = 0
    else:
        m_forward = n_tumors-n_parent
        p_forward = n_node/n_tumors
        log_p_i_forward = np.zeros(n_only_node+1)
        for i in range(n_only_node+1):
            log_p_i_forward[i] = log_nCr(m_forward, i)+(i*np.log(p_forward))+((m_forward-i)*np.log(1-p_forward))
        log_p_forward = logsumexp(log_p_i_forward)

    if n_parent == 0:
        log_p_backward = 0
    else:
        m_backward = n_tumors-n_node
        p_backward = n_parent/n_tumors
        log_p_i_backward = np.zeros(n_only_parent+1)
        for i in range(n_only_parent+1):
            log_p_i_backward[i] = log_nCr(m_backward, i)+(i*np.log(p_backward))+((m_backward-i)*np.log(1-p_backward))
        log_p_backward = logsumexp(log_p_i_backward)

    forward_PR_s = (n_tumors*n_only_node)/(n_node*(n_tumors-n_parent))
    forward_PR_p = np.exp(log_p_forward)
    backward_PR_s = (n_tumors*n_only_parent)/(n_node*(n_tumors-n_node))
    backward_PR_p = np.exp(log_p_backward)
    F2B_p = np.exp(log_p_forward-log_p_backward)
    return(forward_PR_s, forward_PR_p, backward_PR_s, backward_PR_p, F2B_p)

def Complex_PR(df, list_of_gene_set_parent, gene_set_child):
    n_tumors, n_genes = df.shape
    genes = df.columns
    mutated_in_node = df.loc[:,gene_set_child].sum(axis=1)>0
    mutated_in_parent_list = [df.loc[:,gene_set_parent].sum(axis=1)>0 for gene_set_parent in list_of_gene_set_parent]
    mutated_in_parent = mutated_in_parent_list[0]
    i = 1
    while i<len(mutated_in_parent_list):
        mutated_in_parent = mutated_in_parent*mutated_in_parent_list[i]
        i = i+1
    mutated_only_in_node = mutated_in_node * (1-mutated_in_parent)
    mutated_only_in_parent = (1-mutated_in_node) * (mutated_in_parent)

    n_parent = np.sum(mutated_in_parent)
    n_node = np.sum(mutated_in_node)
    n_only_node = np.sum(mutated_only_in_node)
    n_only_parent = np.sum(mutated_only_in_parent)

    if n_node == 0:
        log_p_forward = 0
    else:
        m_forward = n_tumors-n_parent
        p_forward = n_node/n_tumors
        log_p_i_forward = np.zeros(n_only_node+1)
        for i in range(n_only_node+1):
            log_p_i_forward[i] = log_nCr(m_forward, i)+(i*np.log(p_forward))+((m_forward-i)*np.log(1-p_forward))
        log_p_forward = logsumexp(log_p_i_forward)

    if n_parent == 0:
        log_p_backward = 0
    else:
        m_backward = n_tumors-n_node
        p_backward = n_parent/n_tumors
        log_p_i_backward = np.zeros(n_only_parent+1)
        for i in range(n_only_parent+1):
            log_p_i_backward[i] = log_nCr(m_backward, i)+(i*np.log(p_backward))+((m_backward-i)*np.log(1-p_backward))
        log_p_backward = logsumexp(log_p_i_backward)

    forward_PR_s = (n_tumors*n_only_node)/(n_node*(n_tumors-n_parent))
    forward_PR_p = np.exp(log_p_forward)
    backward_PR_s = (n_tumors*n_only_parent)/(n_node*(n_tumors-n_node))
    backward_PR_p = np.exp(log_p_backward)
    F2B_p = np.exp(log_p_forward-log_p_backward)
    return(forward_PR_s, forward_PR_p, backward_PR_s, backward_PR_p, F2B_p)

def plot_pathtimex(df, sets, edges, dot_file='tmp.dot', fig_file='tmp.png'):
    gene_names = list(df.index)
    txt = 'digraph tree {\n'
    for i, genes in enumerate(sets):
        genes_list = ','.join(item for item in genes)
        if len(genes)==1:
            label = '<%s>'%genes_list
            txt += '    Node%i [label=%s, peripheries=1, shape=box, style=\"rounded, filled\", fillcolor=grey95, color=black];\n'%(i, label)
        else:
            peripheries = 1
            ME_score, ME_p,_ = CME(df, genes)
            if ME_p<0.01: # significant mutual exclusivity!
                #fillcolor = 'mistyrose'
                #bordercolor = 'red'
                fillcolor = 'grey95'
                bordercolor = 'black'
            else:
                fillcolor = 'grey95'
                bordercolor = 'black'
            label = '<%s<br/><font color=\'Blue\' POINT-SIZE=\'12\'> %.2f </font><br/><font color=\'ForestGreen\' POINT-SIZE=\'12\'> (%.2e) </font>>'%(genes_list,ME_score,ME_p)
            txt += '    Node%i [label=%s, peripheries=%i, shape=box, style=\"rounded, filled\", fillcolor=%s, color=%s];\n'%(i, label, peripheries, fillcolor, bordercolor)
    for edge in edges:
        i = sets.index(edge[0])
        j = sets.index(edge[1])

        details = CPR(df, [sets[i]], sets[j])
        if details['score_ratio']>1 or details['PR_forward']<0:
            arrow_c = 'red'
        else:
            arrow_c = 'black'
        label = '<<font color=\'Blue\' POINT-SIZE=\'12\'> %.2f </font><br/><font color=\'ForestGreen\' POINT-SIZE=\'12\'> (%.2e) </font>>'%(details['PR_forward'], details['F_p'])
        txt += '    Node%i -> Node%i [style=solid, color=%s, label=%s];\n' %(i, j, arrow_c, label)

    txt += '}'
    with open(dot_file, 'w') as f:
        f.write(txt)
    if fig_file.endswith('.pdf'):
        check_call(['dot','-Tpdf',dot_file,'-o',fig_file])
    else:
        check_call(['dot','-Tpng',dot_file,'-o',fig_file])
    return(txt)

############# G2G for mhn (needs to be investigated) #############

def plot_mhn_old(df, dot_file='tmp.dot', fig_file='tmp.png'):
    gene_names = list(df.index)
    txt = 'digraph tree {\n'
    for i, gene in enumerate(gene_names):
        label = '<%s<br/><font color=\'Blue\' POINT-SIZE=\'12\'> %.2f </font>>'%(gene,df.loc[gene,gene])
        txt += '    Node%i [label=%s, shape=box, style=\"rounded, filled\", fillcolor=grey95, color=black];\n'%(i, label)
    for i, gene_1 in enumerate(gene_names):
        for j, gene_2 in enumerate(gene_names):
            if gene_2!=gene_1:
                if np.round(df.loc[gene_1,gene_2], decimals=1)!=1:
                    if df.loc[gene_1,gene_2]<1:
                        arrow_type= 'obox'
                    else:
                        arrow_type = 'normal'
                    txt+='    Node%i -> Node%i [label=< %.1f >, arrowhead=%s];\n'%(j, i, df.loc[gene_1,gene_2], arrow_type)
    txt += '}'
    with open(dot_file, 'w') as f:
        f.write(txt)
    if fig_file.endswith('.pdf'):
        check_call(['dot','-Tpdf',dot_file,'-o',fig_file])
    else:
        check_call(['dot','-Tpng',dot_file,'-o',fig_file])
    return(txt)

def G2G(df, gene1, gene2, print_on=False):
    # checking effect of gene1 on gene2
    n_tumors, n_genes = df.shape
    n2 = df.loc[:,gene2].sum()
    n1 = df.loc[:,gene1].sum()
    n1and2 = (df.loc[:,gene2]*df.loc[:,gene1]).sum()
    f2 = n2/n_tumors
    f2_m1 = n1and2/n1
    f2_h1 = (n2-n1and2)/(n_tumors-n1)
    if print_on:
        print('Number of mutations in %s, %s, and both: %i, %i, and %i'%(gene1, gene2, n1, n2, n1and2))
    if f2_m1 < f2:
        if print_on:
            print('Mutation in %s REDUCES probability of mutation in %s'%(gene1, gene2))
        # checking p_value of mutation in gene1 reducing chance of mutation in gene2
        log_p_i = np.zeros(n1and2+1)
        for i in range(n1and2+1):
            log_p_i[i] = log_nCr(n1, i) + (i*np.log(f2)) + ((n1-i)*np.log(1-f2))
        p_value = np.exp(logsumexp(log_p_i))
        if print_on:
            print('P-value: %.2e'%(p_value))
    elif f2_m1 > f2:
        if print_on:
            print('Mutation in %s INCREASES probability of mutation in %s'%(gene1, gene2))
        # checking p_value:
        log_p_i = np.zeros(n1and2)
        for i in range(n1and2):
            log_p_i[i] = log_nCr(n1, i) + (i*np.log(f2)) + ((n1-i)*np.log(1-f2))
        p_value = 1-np.exp(logsumexp(log_p_i))
        if print_on:
            print('P-value: %.2e'%(p_value))
    output = {
        'prob. of %s'%(gene2): f2,
        'prob. of %s, given %s'%(gene2,gene1): f2_m1,
        'prob. of %s, given not(%s)'%(gene2,gene1): f2_h1,
        'p_value': p_value
    }
    return (output)

def CG2G(df, gene1, gene2, print_on=False):
    # checking effect of gene1 on gene2
    n_tumors, n_genes = df.shape
    n2 = df.loc[:,gene2].sum()
    n1 = df.loc[:,gene1].sum()
    n1and2 = (df.loc[:,gene2]*df.loc[:,gene1]).sum()
    f2 = n2/n_tumors
    f2_m1 = n1and2/n1
    f2_h1 = (n2-n1and2)/(n_tumors-n1)
    if print_on:
        print('Number of mutations in %s, %s, and both: %i, %i, and %i'%(gene1, gene2, n1, n2, n1and2))
    if f2_m1 < f2:
        if print_on:
            print('Mutation in %s REDUCES probability of mutation in %s'%(gene1, gene2))
        # checking p_value of mutation in gene1 reducing chance of mutation in gene2
        log_p_i = np.zeros(n1and2+1)
        for i in range(n1and2+1):
            log_p_i[i] = log_nCr(n1, i) + (i*np.log(f2)) + ((n1-i)*np.log(1-f2))
        p_value = np.exp(logsumexp(log_p_i))
        if print_on:
            print('P-value: %.2e'%(p_value))
        if print_on:
            print('Healthy %s INCREASES probability of mutation in %s'%(gene1, gene2))
        log_p_i = np.zeros(n2-n1and2)
        for i in range(n2-n1and2):
            log_p_i[i] = log_nCr(n_tumors-n1, i) + (i*np.log(f2)) + ((n_tumors-n1-i)*np.log(1-f2))
        p_value_2 = 1-np.exp(logsumexp(log_p_i))
        if print_on:
            print('P-value: %.2e'%(p_value_2))
    elif f2_m1 > f2:
        if print_on:
            print('Mutation in %s INCREASES probability of mutation in %s'%(gene1, gene2))
        # checking p_value:
        log_p_i = np.zeros(n1and2)
        for i in range(n1and2):
            log_p_i[i] = log_nCr(n1, i) + (i*np.log(f2)) + ((n1-i)*np.log(1-f2))
        p_value = 1-np.exp(logsumexp(log_p_i))
        if print_on:
            print('P-value: %.2e'%(p_value))
        if print_on:
            print('Healthy %s REDUCES probability of mutation in %s'%(gene1, gene2))
        log_p_i = np.zeros(n2-n1and2+1)
        for i in range(n2-n1and2+1):
            log_p_i[i] = log_nCr(n_tumors-n1, i) + (i*np.log(f2)) + ((n_tumors-n1-i)*np.log(1-f2))
        p_value_2 = np.exp(logsumexp(log_p_i))
        if print_on:
            print('P-value: %.2e'%(p_value_2))
    output = {
        'prob. of %s'%(gene2): f2,
        'prob. of %s, given %s'%(gene2,gene1): f2_m1,
        'prob. of %s, given not(%s)'%(gene2,gene1): f2_h1,
        'p_value': p_value,
        'p_value_of_obs_with_healthy_%s'%gene1: p_value_2
    }
    return (output)

def SOS(df, gene1, gene2):
    # Second Order Statistics
    n_tumors, n_genes = df.shape
    
    n = {
        '1': df.loc[:,gene1].sum(),
        '2': df.loc[:,gene2].sum(),
        '1&2': (df.loc[:,gene2]*df.loc[:,gene1]).sum()
    }
    n['1-2'] = n['1']-n['1&2']
    n['2-1'] = n['2']-n['1&2']
    n['1||2'] = n['1']+n['2']-n['1&2']
    
    f = {
        '1': n['1']/n_tumors,
        '2': n['2']/n_tumors,
        '1&2': n['1&2']/n_tumors
    }
    return(n,f)

def CME(df, gene_set):
    n = len(gene_set)
    n_tumors, n_genes = df.shape
    details = {}
    if n > 1:
        me_log_p_values = []
        ME_score_values = []
        for _idx1 in range(n):
            for _idx2 in range(n):
                if _idx2 != _idx1:
                    mutated_in_1 = df.loc[:,gene_set[_idx1]]
                    mutated_in_2 = df.loc[:,gene_set[_idx2]]
                    mutated_in_both = mutated_in_1 * mutated_in_2
                    n_1 = np.sum(mutated_in_1)
                    n_2 = np.sum(mutated_in_2)
                    n_both = np.sum(mutated_in_both)

                    if n_1 == 0:
                        ME_1_to_2 = 0
                        log_p = 0
                    else:
                        p_1_given_2 = n_both/n_2
                        p_1_given_not_2 = (n_1-n_both)/(n_tumors-n_2)
                        ME_1_to_2 = (p_1_given_not_2-p_1_given_2)/(p_1_given_not_2+p_1_given_2)

                        log_p_i = []
                        for i in range(n_both+1):
                            log_p_i.append(
                                log_nCr(n_1, i)+(i*np.log(n_2/n_tumors))+((n_1-i)*np.log(1-n_2/n_tumors))
                            )
                        log_p = logsumexp(log_p_i)
                    ME_score_values.append(ME_1_to_2)
                    me_log_p_values.append(log_p)
                    details['%s-to-%s'%(gene_set[_idx1],gene_set[_idx2])] = (ME_1_to_2, np.exp(log_p))

        overall_ME_score = np.mean(ME_score_values)       
        ME_p = np.mean(np.exp(me_log_p_values))
        return(overall_ME_score, ME_p, details)

def CPR(df, set_of_parent_sets, child_set):
    # CASUAL PROGRESSION RELATION
    # EQUIVALENT OF PR_TEST USED IN TOMEXO
    # Example set_of_parents_sets : [[1,2], [3]]
    # Example child_set : [4]
    n_tumors, n_genes = df.shape

    mutated_in_child = df.loc[:,child_set].sum(axis=1)>0
    mutated_in_parent_list = [df.loc[:,gene_set_parent].sum(axis=1)>0 for gene_set_parent in set_of_parent_sets]
    mutated_in_parent = mutated_in_parent_list[0]
    i = 1
    while i<len(mutated_in_parent_list):
        mutated_in_parent = mutated_in_parent & mutated_in_parent_list[i]
        i = i+1
    mutated_only_in_child = mutated_in_child & (1-mutated_in_parent)
    mutated_only_in_parent = (1-mutated_in_child) & (mutated_in_parent)
    mutated_in_both = mutated_in_child & mutated_in_parent

    n_b = np.sum(mutated_in_both)
    n_p = np.sum(mutated_in_parent)
    n_c = np.sum(mutated_in_child)
    n_n = n_tumors - n_p - n_c + n_b

    p_child_given_parent = n_b/n_p
    p_child_given_not_parent = (n_c-n_b)/(n_tumors-n_p)
    PR_forward = (p_child_given_parent-p_child_given_not_parent)/(p_child_given_parent+p_child_given_not_parent)

    log_p_i_forward = []
    for i in range(n_b, n_p+1):
        log_p_i_forward.append(
            log_nCr(n_p, i)+(i*np.log(n_c/n_tumors))+((n_p-i)*np.log(1-n_c/n_tumors))
        )
    log_p_forward = logsumexp(log_p_i_forward)
    
    F_p = np.exp(log_p_forward)

    p_parent_given_child = n_b/n_c
    p_parent_given_not_child = (n_p-n_b)/(n_tumors-n_c)
    PR_backward = (p_parent_given_child-p_parent_given_not_child)/(p_parent_given_child+p_parent_given_not_child)

    log_p_i_backward = []
    for i in range(n_b, n_c+1):
        log_p_i_backward.append(
            log_nCr(n_c, i)+(i*np.log(n_p/n_tumors))+((n_c-i)*np.log(1-n_p/n_tumors))
        )
    log_p_backward = logsumexp(log_p_i_backward)

    B_p = np.exp(log_p_backward)

    score_ratio = (PR_backward+0.0001)/(PR_forward+0.0001)

    details = {
        'n_b': n_b,
        'n_p': n_p,
        'n_c': n_c,
        'n_n': n_n,
        'p_child_given_parent': p_child_given_parent,
        'p_child_given_not_parent': p_child_given_not_parent,
        'p_parent_given_child': p_parent_given_child,
        'p_parent_given_not_child': p_parent_given_not_child,
        'PR_forward': PR_forward,
        'F_p': F_p,
        'PR_backward': PR_backward,
        'B_p': B_p,
        'score_ratio': score_ratio
    }
    return(details)

def Desper_test(df, gene):
    n_tumors, n_genes = df.shape
    pg = np.sum(df.loc[:,gene])/n_tumors
    desper_weight = {}
    for x in df.columns:
        if x != gene:
            px = np.sum(df.loc[:,x])/n_tumors
            if px>=pg:
                pxg = np.sum(df.loc[:,x] * df.loc[:,gene])/n_tumors
                desper_weight[x] = ((px)/(px+pg))*((pxg)/(px*pg))
    desper_weight['LHS'] = 1/(1+pg)
    return(desper_weight)
