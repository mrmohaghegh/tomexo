#! /usr/bin/env python3

import numpy as np
import time
from scipy.special import logsumexp

from anytree import NodeMixin, PostOrderIter
from anytree.exporter import DotExporter
from IPython.display import Image
from copy import deepcopy
import matplotlib.pyplot as plt
from util import ME_test, PR_test, perfect_ME, perfect_PR
from subprocess import check_call

class OncoNode(NodeMixin):

    def __init__(self, genes=[], f=1, parent=None, children=[]):
        super().__init__()
        self.genes = genes
        self.f = f
        self.parent = parent
        self.children = children
    
    @property
    def s(self):
        return(len(self.genes))

    @property
    def is_root(self):
        # Is it the root?
        if self.parent is None and not(self.is_ps):
            return(True)
        else:
            return(False)
    
    @property
    def is_ps(self):
        # Is it the set of passengers?
        if self.parent is None and self.f==0:
            return(True)
        else:
            return(False)
    
    @property
    def is_leaf(self):
        # Is it a leaf node?
        if len(self.children)==0:
            return(True)
        else:
            return(False)

    @property
    def is_simple(self):
        # Is it a simple node (including a single gene, in the first layer, with no child)
        if self.parent is None:
            return(False)
        elif len(self.children)==0 and self.parent.is_root and len(self.genes)==1:
            return(True)
        else:
            return(False)


class OncoTree():

    def __init__(self, nodes, pfp, pfn, single_error=False, mut_rates=None):
        self.nodes = nodes
        tmp = []
        for node in nodes:
            tmp.extend(node.genes)
        self.genes = tmp
        self.n_genes = len(tmp)
        self.ps = None
        for node in self.nodes:
            if node.is_root:
                self.root = node
            elif node.is_ps:
                self.ps = node
        self.pfp = pfp
        self.pfn = pfn
        self.single_error = single_error
        if mut_rates is None:
            self.mut_rates = np.ones(self.n_genes, dtype=int)
        else:
            self.mut_rates = mut_rates

    @classmethod
    def from_dataset(cls, dataset, coeff=0.5, passenger_threshold=0.001, pfp=0.001, pfn=0.001, single_error=False, error_estimation=False):
        # --- Initial tree adapted to the dataset ---
        # Genes with mut freq lower than the threshold are put into the set of passengers
        # The driver sets have a linear structure where the avg mut freq of each driver set -
        #   - is lower than its parent by a factor less than "coeff"
        n_muts = np.sum(dataset, axis=0)
        mmg = np.argsort(n_muts)[::-1] # Most frequently Mutated Genes, MMG
        root_node = OncoNode(genes=[], f=1)
        nodes=[root_node, OncoNode(genes=[mmg[0]], f=1, parent=root_node)]
        psnode = OncoNode(genes=[], f=0)
        for i in range(1, len(n_muts)):
            if n_muts[mmg[i]]<passenger_threshold*dataset.shape[0]:
                psnode.genes.append(mmg[i])
            elif n_muts[mmg[i]]<coeff*np.mean(n_muts[nodes[-1].genes]):
                nodes.append(OncoNode(genes=[mmg[i]], parent=nodes[-1], f=1))
            else:
                nodes[-1].genes.append(mmg[i])
        nodes.append(psnode)
        prog_mo = cls(nodes, pfp=pfp, pfn=pfn, single_error=single_error, mut_rates=np.sum(dataset,axis=0))
        # for the case of uniform rate of mutation for the genes of each node, use:
        # prog_mo = cls(nodes, pfp=pfp, pfn=pfn, single_error=single_error)
        prog_mo = prog_mo.assign_f_values(dataset)
        if error_estimation:
            prog_mo = prog_mo.assign_error_values(dataset)
        return(prog_mo)
    
    @classmethod
    def star_from_dataset(cls, dataset, pfp=0.001, pfn=0.001, single_error=False, error_estimation=False):
        # --- Star tree adapted to the dataset ---
        root_node = OncoNode(genes=[], f=1)
        nodes = [root_node]
        for gene in range(dataset.shape[1]):
            nodes.append(OncoNode(parent=root_node, genes=[gene], f=0.5))
        psnode = OncoNode(genes=[], f=0)
        nodes.append(psnode)
        prog_mo = cls(nodes, pfp=pfp, pfn=pfn, single_error=single_error, mut_rates=np.sum(dataset,axis=0))
        # for the case of uniform rate of mutation for the genes of each node, use:
        # prog_mo = cls(nodes, pfp=pfp, pfn=pfn, single_error=single_error)
        prog_mo = prog_mo.assign_f_values(dataset)
        if error_estimation:
            prog_mo = prog_mo.assign_error_values(dataset)
        return(prog_mo)

    def to_matrix(self):
        # matrix[i,i]=0
        # matrix[i,j]=1, iff, i and j are in the same node
        # matrix[i,j]=2, iff, the node including i is a descendant of the node including j
        # matrix[i,j]=3, iff, the node including i is an ancestor of the node including j
        matrix = np.zeros(shape=(self.n_genes, self.n_genes), dtype=int)
        for node in PostOrderIter(self.root):
            for i in node.genes:
                for j in node.genes:
                    matrix[i, j] = 1
                for anc_node in node.ancestors:
                    for j in anc_node.genes:
                        matrix[i, j] = 2
                        matrix[j, i] = 3
        for i in range(self.n_genes):
            matrix[i, i] = 0
        return(matrix)
    
    def compare(self, ref_mat):
        # Compares self with a reference matrix or OncoTree object
        self_mat = self.to_matrix()
        if type(ref_mat) == OncoTree:
            ref_mat = ref_mat.to_matrix()
        prog_TP = np.sum((ref_mat==2)*(self_mat==2))
        prog_FP = np.sum((ref_mat!=2)*(self_mat==2))
        prog_FN = np.sum((ref_mat==2)*(self_mat!=2))
        if prog_TP == 0:
            prog_precision = 0
            prog_recall = 0
            prog_f = 0
        else:
            prog_precision = prog_TP/(prog_TP+prog_FP)
            prog_recall = prog_TP/(prog_TP+prog_FN)
            prog_f=(2*prog_precision*prog_recall)/(prog_precision+prog_recall)
        mx_TP = int(0.5*np.sum((ref_mat==1)*(self_mat==1)))
        mx_FP = int(0.5*np.sum((ref_mat!=1)*(self_mat==1)))
        mx_FN = int(0.5*np.sum((ref_mat==1)*(self_mat!=1)))
        if mx_TP == 0:
            mx_precision = 0
            mx_recall = 0
            mx_f = 0
        else:
            mx_precision = mx_TP/(mx_TP+mx_FP)
            mx_recall = mx_TP/(mx_TP+mx_FN)
            mx_f=(2*mx_precision*mx_recall)/(mx_precision+mx_recall)
        return(prog_precision, prog_recall, prog_f, mx_precision, mx_recall, mx_f)
      
    def remove_subtree(self, node, mode="into_simple_nodes"):
        # modes: "into_simple_nodes", "into_passengers", "spr_into_root", "spr_into_grandparent", "break_leaf"
        if mode == "into_simple_nodes":
            for _node in PostOrderIter(node):
                for gene in _node.genes:
                    self.nodes.append(OncoNode(genes=[gene], f=0.5, parent=self.root))
                self.nodes.remove(_node)
            node.parent = None
        elif mode == "into_passengers":
            for _node in PostOrderIter(node):
                self.ps.genes.extend(_node.genes)
                self.nodes.remove(_node)
            node.parent = None
        elif mode == "spr_into_root":
            node.parent = self.root
        elif mode == "spr_into_grandparent":
            if not(node.is_root):
                if not(node.parent.is_root):
                    node.parent = node.parent.parent
        elif mode == "break_leaf":
            if node.is_leaf:
                for gene in node.genes:
                    self.nodes.append(OncoNode(genes=[gene], f=0.5, parent=node.parent))
                self.nodes.remove(node)
                node.parent = None
        else:
            print("Pruning mode error!")
        return(self)

    def prune(self, dataset, consider_mut_freqs=True):
        self = self.prune_by_p_values(dataset)
        if consider_mut_freqs:
            self = self.prune_by_mut_freqs(dataset)
        return(self)

    def prune_by_mut_freqs(self, dataset, th_f=0.005, just_f=True):
        # th_p: threshold for prob. of driver mutation to keep the node
        #       set to pfp
        # th_f: threshold for f to keep the node
        change_occured = True
        pruned_tree = deepcopy(self)
        while change_occured:
            change_occured = False
            th_p = pruned_tree.pfp/100 ### could be set to pruned_tree.pfp
            nodes_to_remove = []
            for node in PostOrderIter(pruned_tree.root):
                if node.f <= th_f:
                    nodes_to_remove.append(node.genes)
                elif not(node.is_root):
                    if not just_f:
                        total_f = 1
                        for anc_node in node.ancestors:
                            total_f *= anc_node.f
                        total_f *= node.f
                        total_f *= (1/len(node.genes))
                        if total_f <= th_p:
                            nodes_to_remove.append(node.genes)
            if len(nodes_to_remove)>0:
                change_occured = True
            for item in nodes_to_remove:
                g = item[0]
                for _i, _n in enumerate(pruned_tree.nodes):
                    if g in _n.genes:
                        the_node = _n
                if the_node.parent.is_root:
                    pruned_tree = pruned_tree.remove_subtree(the_node, mode="into_passengers")
                else:
                    pruned_tree = pruned_tree.remove_subtree(the_node, mode="into_simple_nodes")
            pruned_tree = pruned_tree.assign_error_values(dataset)
            pruned_tree = pruned_tree.assign_f_values(dataset, fine_tuning=True)
            pruned_tree,_=pruned_tree.fit_error_params(dataset)
        return(pruned_tree)
    
    def prune_by_p_values(self, dataset, mode="into_simple_nodes"):
        # Prunes based on p-values for progression
        # modes: "into_simple_nodes", "into_passengers"
        pruned_tree = deepcopy(self)
        ##### STEP 1: GET RID OF BAD ME SETS! #####
        nodes_to_remove = []
        for node in PostOrderIter(pruned_tree.root):
            if not(node.is_root):
                if len(node.genes)>1:
                    ME_score, ME_p = ME_test(node, dataset)
                    if ME_score < 0:
                        nodes_to_remove.append(node.genes)
        for item in nodes_to_remove:
            g = item[0]
            for _i, _n in enumerate(pruned_tree.nodes):
                if g in _n.genes:
                    the_node = _n
            for child_node in the_node.children:
                pruned_tree = pruned_tree.remove_subtree(child_node, mode="spr_into_grandparent")
            pruned_tree = pruned_tree.remove_subtree(the_node, mode="break_leaf")
        ##### STEP 2: GET RID OF BAD PR EDGES! #####
        change_happend = True
        while change_happend:
            change_happend = False
            nodes_to_remove = []
            for node in PostOrderIter(pruned_tree.root):
                if not(node.is_root):
                    if not(node.parent.is_root):
                        PR_forward, _, _, _, BtoF = PR_test(node, dataset)
                        if BtoF > 1 or PR_forward < 0:
                            nodes_to_remove.append(node.genes)
            if len(nodes_to_remove)>0:
                change_happend = True
            for item in nodes_to_remove:
                g = item[0]
                for _i, _n in enumerate(pruned_tree.nodes):
                    if g in _n.genes:
                        the_node = _n
                pruned_tree = pruned_tree.remove_subtree(the_node, mode="spr_into_grandparent")
        pruned_tree = pruned_tree.assign_error_values(dataset)
        pruned_tree = pruned_tree.assign_f_values(dataset, fine_tuning=True)
        pruned_tree,_=pruned_tree.fit_error_params(dataset)
        return(pruned_tree)
                
    def sample_mut_set(self, node=None):
        if node is None:
            node = self.root
        if len(node.genes)>0:
            mut_set = [np.random.choice(node.genes, p=self.mut_rates[node.genes]/(np.sum(self.mut_rates[node.genes])))]
        else:
            mut_set = []
        for child in node.children:
            if bool(np.random.binomial(1, child.f)):
                mut_set.extend(self.sample_mut_set(node=child))
        return(mut_set)
    
    def draw_sample(self, n_tumors):
        clean_dataset = np.zeros(shape=(n_tumors, self.n_genes), dtype=bool)
        for tumor_idx in range(n_tumors):
            driver_genes = self.sample_mut_set()
            clean_dataset[tumor_idx, driver_genes] = True
        dataset = np.zeros(shape=(n_tumors, self.n_genes), dtype=bool)
        for i in range(n_tumors):
            for j in range(self.n_genes):
                if clean_dataset[i,j]:
                    dataset[i,j] = not(np.random.binomial(1, self.pfn))
                else:
                    dataset[i,j] = bool(np.random.binomial(1, self.pfp))
        return(dataset, clean_dataset)

    def plot_single_tumor(self, the_row, gene_names=None, dot_file='tmp.dot', fig_file='tmp.png'):
        driver_nodes = [self.root]
        driver_nodes.extend([node for node in self.root.descendants])
        if gene_names is None:
            gene_names= ['g%i'%tmp_i for tmp_i in range(self.n_genes)]
        txt = 'digraph tree {\n'
        for i, node in enumerate(driver_nodes):
            genes_list = ','.join(gene_names[tmp_i] for tmp_i in node.genes)
            if len(node.genes)==0:
                label = '< >'
                txt += '    Node%i [label=%s, peripheries=1, shape=circle, style=filled, fillcolor=grey34];\n'%(i, label)
            else:
                n_muts = np.sum([the_row[_tri] for _tri in node.genes])
                if n_muts==0:
                    fillcolor = 'grey95'
                elif n_muts==1:
                    fillcolor = 'limegreen'                    
                else:
                    fillcolor = 'lightcoral'
                bordercolor = 'black'
                peripheries = 1
                label = '<%s>'%(genes_list)
                txt += '    Node%i [label=%s, peripheries=%i, shape=box, style=\"rounded, filled\", fillcolor=%s, color=%s];\n'%(i, label, peripheries, fillcolor, bordercolor)
        # plotting the set of passengers
        genes_list = ','.join(gene_names[tmp_i] for tmp_i in self.ps.genes)
        n_muts = np.sum([the_row[_tri] for _tri in self.ps.genes])
        if n_muts>0:
            fillcolor = 'lightcoral'
        else:
            fillcolor='grey76'
        txt += '    PS [label=<%s>][shape=box, peripheries=1, style=\"rounded, filled\", fillcolor=%s];\n'%(genes_list, fillcolor)
        for i, node in enumerate(driver_nodes):
            if node.is_root:
                for child in node.children:
                    j = driver_nodes.index(child)
                    txt += '    Node%i -> Node%i [label=< >];\n' %(i, j)
            else:
                for child in node.children:
                    j = driver_nodes.index(child)
                    #PR_score, PR_p = PR_test(child, dataset)
                    arrow_color = 'black'
                    label = '< >'
                    txt += '    Node%i -> Node%i [style=solid, color=%s, label=%s];\n' %(i, j, arrow_color, label)

        txt += '}'
        with open(dot_file, 'w') as f:
            f.write(txt)
        if fig_file.endswith('.pdf'):
            check_call(['dot','-Tpdf',dot_file,'-o',fig_file])
        else:
            check_call(['dot','-Tpng',dot_file,'-o',fig_file])
        return(txt)

    def to_dot(self, dataset, gene_names=None, dot_file='tmp.dot', fig_file='tmp.png', show_passengers=False, plot_reverse_edges=False):
        driver_nodes = [self.root]
        driver_nodes.extend([node for node in self.root.descendants])
        if gene_names is None:
            gene_names= ['g%i'%tmp_i for tmp_i in range(self.n_genes)]
        txt = 'digraph tree {\n'
        for i, node in enumerate(driver_nodes):
            genes_list = ','.join(gene_names[tmp_i] for tmp_i in node.genes)
            if len(genes_list)==0:
                label = '< >'
                txt += '    Node%i [label=%s, peripheries=1, shape=circle, style=filled, fillcolor=grey34];\n'%(i, label)
            elif len(node.genes)==1:
                label = '<%s>'%genes_list
                txt += '    Node%i [label=%s, peripheries=1, shape=box, style=\"rounded, filled\", fillcolor=grey95, color=black];\n'%(i, label)
            else:
                ME_score, ME_p = ME_test(node, dataset)
                if ME_p<0.01: # significant mutual exclusivity!
                    #fillcolor = 'mistyrose'
                    fillcolor = 'grey95'
                    #bordercolor = 'red'
                    bordercolor = 'black'
                else:
                    fillcolor = 'grey95'
                    bordercolor = 'black'
                if perfect_ME(node, dataset):
                    #peripheries = 2 #(used for double-line borders in case of perfect ME)
                    peripheries = 1
                else:
                    peripheries = 1
                label = '<%s<br/><font color=\'Blue\' POINT-SIZE=\'12\'> %.2f </font><br/><font color=\'ForestGreen\' POINT-SIZE=\'12\'> (%.2e) </font>>'%(genes_list,ME_score,ME_p)
                txt += '    Node%i [label=%s, peripheries=%i, shape=box, style=\"rounded, filled\", fillcolor=%s, color=%s];\n'%(i, label, peripheries, fillcolor, bordercolor)
        if (show_passengers and len(self.ps.genes)>0):
            genes_list = ','.join(gene_names[tmp_i] for tmp_i in self.ps.genes)
            txt += '    PS [label=<%s>][shape=box, peripheries=1, style=\"rounded, filled\", fillcolor=grey76];\n'%(genes_list)
        for i, node in enumerate(driver_nodes):
            if node.is_root or np.sum(np.sum(dataset[:,node.genes], axis=1)>0)==dataset.shape[0]:
                for child in node.children:
                    j = driver_nodes.index(child)
                    txt += '    Node%i -> Node%i [label=< %.2f >];\n' %(i, j, child.f)
            else:
                for child in node.children:
                    j = driver_nodes.index(child)
                    #PR_score, PR_p = PR_test(child, dataset)
                    PR_forward, F_p, PR_backward, B_p, FtoB_p_ratio = PR_test(child, dataset)
                    if F_p < 0.01:
                        arrow_c = 'black'
                        #arrow_c = 'red'
                    else:
                        arrow_c = 'black'
                    if perfect_PR(child, dataset):
                        #arrow_color = '\"%s:%s\"'%(arrow_c, arrow_c) #(used for double-line arrow in case of perfect PR)
                        arrow_color = arrow_c
                    else:
                        arrow_color = arrow_c
                    label = '< %.2f <br/> <font color=\'Blue\' POINT-SIZE=\'12\'> %.2f </font><br/><font color=\'ForestGreen\' POINT-SIZE=\'12\'> (%.2e) </font>>'%(child.f, PR_forward, F_p)
                    txt += '    Node%i -> Node%i [style=solid, color=%s, label=%s];\n' %(i, j, arrow_color, label)
                    if plot_reverse_edges and B_p < 0.1: # strong reverse relation as well!
                        txt += '    Node%i -> Node%i [style=dashed, color=black];\n' %(j, i)

        txt += '}'
        with open(dot_file, 'w') as f:
            f.write(txt)
        if fig_file.endswith('.pdf'):
            check_call(['dot','-Tpdf',dot_file,'-o',fig_file])
        else:
            check_call(['dot','-Tpng',dot_file,'-o',fig_file])
        return(txt)

    def min_errors(self, dataset):
        constant_to_be_added = 0.00001
        n_errors = np.array([0, 0])
        for tumor_idx in range(dataset.shape[0]):
            r = {}
            q_0 = {}
            q_1 = {}
            w_0 = {}
            w_1 = {}
            for node in PostOrderIter(self.root):
                if node.is_root:
                    q_0[node] = np.array([0, 0])
                    q_1[node] = np.array([0, 0])
                else:
                    r[node] = np.sum(dataset[tumor_idx, node.genes])
                    q_0[node] = np.array([r[node], 0])
                    if r[node] == 0:
                        q_1[node] = np.array([0, 1])
                    else:
                        q_1[node] = np.array([r[node]-1, 0])
                w_0[node] = np.array([0, 0])
                w_0[node] += q_0[node]
                for child in node.children:
                    w_0[node] += w_0[child]
                w_1[node] = np.array([0, 0])
                w_1[node] += q_1[node]
                for child in node.children:
                    if w_1[child][0]+w_1[child][1] <= w_0[child][0]+w_0[child][1]:
                        w_1[node] += w_1[child]
                    else:
                        w_1[node] += w_0[child]
            n_errors += w_1[self.root] + np.array([np.sum(dataset[tumor_idx, self.ps.genes]), 0])
        n_ones = np.sum(dataset)
        n_zeros = np.size(dataset)-n_ones
        epsilon_hat = (n_errors[0]+constant_to_be_added)/(n_zeros-n_errors[1]+n_errors[0]+2*constant_to_be_added)
        delta_hat = (n_errors[1]+constant_to_be_added)/(n_ones-n_errors[0]+n_errors[1]+2*constant_to_be_added)
        e_hat = (n_errors[0]+n_errors[1]+constant_to_be_added)/(n_ones+n_zeros+2*constant_to_be_added)
        if epsilon_hat > 0.5 or delta_hat > 0.5:
            # print(len(self.ps.genes))
            # print(len(self.nodes))
            # print('dataset shape is :')
            # print(dataset.shape)
            # print('with %i zeros and %i ones'%(n_zeros, n_ones))
            # print('epsilon: %.4f -- delta: %.4f'%(epsilon_hat,delta_hat))
            # print('n_fp: %i and n_fn: %i'%(n_errors[0], n_errors[1]))
            epsilon_hat = 0.49
            delta_hat = 0.49
            e_hat = 0.49
        return(epsilon_hat, delta_hat, e_hat)
    
    def assign_error_values(self, dataset):
        epsilon_hat, delta_hat, e_hat = self.min_errors(dataset)
        if self.single_error:
            self.pfp = e_hat
            self.pfn = e_hat
        else:
            self.pfp = epsilon_hat
            self.pfn = delta_hat
        return(self)

    def assign_f_values(self, dataset=None, fine_tuning=False):
        if fine_tuning:
            upperbound = 0.9999
            lowerbound = 0.0001
        else:
            upperbound = 0.95
            lowerbound = 0.05
        # Dataset-free version:
        if dataset is None:
            for node in PostOrderIter(self.root):
                if node.is_root:
                    node.f = 1
                else:
                    tmp = 1
                    for child in node.children:
                        tmp *= 1-child.f
                    node.f = 1/(1+tmp)
                node.f = np.min([node.f, upperbound])
                node.f = np.max([node.f, lowerbound])
        else:
            n_tumors = dataset.shape[0]
            for node in PostOrderIter(self.root):
                if node.is_root:
                    node.f = 1
                elif node.parent.is_root:
                    n_p = n_tumors
                    n_u_p = np.sum(np.sum(dataset[:,node.genes], axis=1)>0)
                    node.f = np.max((n_u_p-self.pfp*n_p)/(n_p*(1-self.pfp-self.pfn)), 0)
                else:
                    n_p = np.sum(
                        np.sum(dataset[:,node.parent.genes], axis=1)>0
                    )
                    if n_p == 0:
                        node.f = 0
                    else:
                        n_u_p = np.sum(
                            (np.sum(dataset[:,node.parent.genes], axis=1)>0)*(np.sum(dataset[:,node.genes], axis=1)>0)
                        )
                        node.f = np.max((n_u_p-self.pfp*n_p)/(n_p*(1-self.pfp-self.pfn)), 0)
                node.f = np.min([node.f, upperbound])
                node.f = np.max([node.f, lowerbound])
        return(self)

    def prior(self, power=0):
        # power = 0 : Uniform prior
        # power = 1 : Prior proportional to 1/(number of possible b*'s)
        p = 0
        if power != 0:
            # for node in self.nodes:
            #     if not(node.is_ps):
            #         p += -np.log(len(node.genes)+1)
            v = {}
            for node in PostOrderIter(self.root):
                if not node.is_root:
                    v[node] = np.log(len(node.genes))
                else:
                    v[node] = 0
                for child in node.children:
                    v[node] += logsumexp([v[child], 0])
            p = -v[self.root]*power
        return(p)

    def fit_error_params(self, dataset):
        step_size = 10**(-6)
        _, llh, llh_eps, llh_del = self.error_derivatives(dataset, self.pfp, self.pfn)
        llhs = [llh]
        progress = 1
        while progress>0.1:
            if self.single_error:
                llh_e = llh_eps + llh_del
                new_pfp = min(0.999, max(10**(-5), self.pfp+step_size*llh_e))
                new_pfn = min(0.999, max(10**(-5), self.pfp+step_size*llh_e))
            else:
                new_pfp = min(0.999, max(10**(-5), self.pfp+step_size*llh_eps))
                new_pfn = min(0.999, max(10**(-5), self.pfn+step_size*llh_del))
            _, llh, llh_eps, llh_del = self.error_derivatives(dataset, new_pfp, new_pfn)
            if llh>llhs[-1]:
                # Accepting new values:
                self.pfp = new_pfp
                self.pfn = new_pfn
                progress = llh-llhs[-1]
                llhs.append(llh)
            else:
                progress = -1
        return(self, llhs)

    def error_derivatives(self, dataset, pfp, pfn):
        # computes derivatives of the log-likelihood w.r.t. the error parameters
        log_pfp = np.log(pfp)
        log_ptn = np.log(1-pfp)
        log_pfn = np.log(pfn)
        log_ptp = np.log(1-pfn)
        llh = 0
        llh_eps = 0
        llh_del = 0
        for tumor_idx in range(dataset.shape[0]):
            r = {}
            _gamma = {}
            _gamma_eps = {}
            _lambda = {}
            _lambda_eps = {}
            _lambda_del = {}
            _omega = {}
            _omega_eps = {}
            _psi = {}
            _psi_eps = {}
            _psi_del = {}
            for node in PostOrderIter(self.root):
                r[node] = np.sum(dataset[tumor_idx, node.genes])
                _gamma[node] = r[node]*log_pfp + (node.s-r[node])*log_ptn
                _gamma_eps[node] = r[node]/pfp - (node.s-r[node])/(1-pfp)
                if node.s == 0:
                    _lambda[node] = 0
                    _lambda_eps[node] = 0
                    _lambda_del[node] = 0
                elif r[node] == 0:
                    _lambda[node] = log_pfn+(node.s-r[node]-1)*log_ptn
                    _lambda_eps[node] = r[node]/pfp - (node.s-r[node]-1)/(1-pfp)
                    _lambda_del[node] = 1/pfn
                elif r[node] == node.s:
                    _lambda[node] = log_ptp+(r[node]-1)*log_pfp
                    _lambda_eps[node] = (r[node]-1)/pfp - (node.s-r[node])/(1-pfp)
                    _lambda_del[node] = -1/(1-pfn)
                else:
                    the_coeff = (np.sum(dataset[tumor_idx, node.genes]*self.mut_rates[node.genes]))/(np.sum(self.mut_rates[node.genes]))
                    if the_coeff < 1: # to prevent numerical issues when the coeff is one
                        _lambda[node] = logsumexp([
                            np.log(the_coeff)+log_ptp+(r[node]-1)*log_pfp+(node.s-r[node])*log_ptn,
                            np.log(1-the_coeff)+log_pfn+r[node]*log_pfp+(node.s-r[node]-1)*log_ptn
                        ])
                    else:
                        _lambda[node] = log_ptp+(r[node]-1)*log_pfp+(node.s-r[node])*log_ptn
                    _lambda_eps[node] = (
                            (the_coeff*(1-pfn))*((r[node]-1)*(pfp**(r[node]-2))*((1-pfp)**(node.s-r[node]))-(node.s-r[node])*((1-pfp)**(node.s-r[node]-1))*(pfp**(r[node]-1)))+
                            ((1-the_coeff)*pfn)*((r[node])*(pfp**(r[node]-1))*((1-pfp)**(node.s-r[node]-1))-(node.s-r[node]-1)*((1-pfp)**(node.s-r[node]-2))*(pfp**(r[node])))
                        )/np.exp(_lambda[node])
                    _lambda_del[node] = ((node.s*pfp-r[node])*((pfp**(r[node]-1))*((1-pfp)**(node.s-r[node]-1))))/(node.s*np.exp(_lambda[node]))
                    _lambda_del[node] = (
                        (-the_coeff*(pfp**(r[node]-1))*((1-pfp)**(node.s-r[node])))+
                        ((1-the_coeff)*(pfp**(r[node]))*((1-pfp)**(node.s-r[node]-1)))
                        )/np.exp(_lambda[node])
                _omega[node] = _gamma[node]+np.sum([_omega[child] for child in node.children])
                _omega_eps[node] = _gamma_eps[node]+np.sum([_omega_eps[child] for child in node.children])
                tmp = np.zeros(len(node.children))
                for i, child in enumerate(node.children):
                    if child.f == 0:
                        tmp[i] = _omega[child]
                    elif child.f == 1:
                        tmp[i] = _psi[child]
                    else:
                        tmp[i] = logsumexp([
                            np.log(child.f)+_psi[child],
                            np.log(1-child.f)+_omega[child]
                        ])
                _psi[node] = _lambda[node] + np.sum(tmp)
                tmp_2 = np.exp(-tmp)
                tmp_3 = np.zeros(len(node.children))
                tmp_4 = np.zeros(len(node.children))
                for i, child in enumerate(node.children):
                    tmp_3[i] = child.f*np.exp(_psi[child])*_psi_eps[child]+(1-child.f)*np.exp(_omega[child])*_omega_eps[child]
                    tmp_4[i] = child.f*np.exp(_psi[child])*_psi_del[child]
                _psi_eps[node] = _lambda_eps[node] + np.sum(tmp_2*tmp_3)
                _psi_del[node] = _lambda_del[node] + np.sum(tmp_2*tmp_4)
            llh += _psi[self.root]
            llh_eps += _psi_eps[self.root]
            llh_del += _psi_del[self.root]
            if self.ps is not None:
                r[self.ps] = np.sum(dataset[tumor_idx, self.ps.genes])
                _psi[self.ps] = r[self.ps]*log_pfp+(self.ps.s-r[self.ps])*log_ptn
                llh += _psi[self.ps]
                llh_eps += r[self.ps]/pfp - (self.ps.s-r[self.ps])/(1-pfp)
        return(self, llh, llh_eps, llh_del)

    def likelihood(self, dataset, pfp=None, pfn=None):
        if (pfp is None) and (pfn is None):
            pfp = deepcopy(self.pfp)
            pfn = deepcopy(self.pfn)
        log_pfp = np.log(pfp)
        log_ptn = np.log(1-pfp)
        log_pfn = np.log(pfn)
        log_ptp = np.log(1-pfn)
        llh = 0
        for tumor_idx in range(dataset.shape[0]):
            r = {}
            _gamma = {}
            _lambda = {}
            _omega = {}
            _psi = {}
            for node in PostOrderIter(self.root):
                r[node] = np.sum(dataset[tumor_idx, node.genes])
                _gamma[node] = r[node]*log_pfp + (node.s-r[node])*log_ptn
                if node.s == 0:
                    _lambda[node] = 0
                elif r[node] == 0:
                    _lambda[node] = log_pfn+(node.s-r[node]-1)*log_ptn
                elif r[node] == node.s:
                    _lambda[node] = log_ptp+(r[node]-1)*log_pfp
                else:
                    if self.mut_rates is None:
                        the_coeff = r[node]/node.s
                    else:
                        the_coeff = (np.sum(dataset[tumor_idx, node.genes]*self.mut_rates[node.genes]))/(np.sum(self.mut_rates[node.genes]))
                    if the_coeff < 1: # to prevent numerical issues when the coeff is one
                        _lambda[node] = logsumexp([
                            np.log(the_coeff)+log_ptp+(r[node]-1)*log_pfp+(node.s-r[node])*log_ptn,
                            np.log(1-the_coeff)+log_pfn+r[node]*log_pfp+(node.s-r[node]-1)*log_ptn
                        ])
                    else:
                        _lambda[node] = log_ptp+(r[node]-1)*log_pfp+(node.s-r[node])*log_ptn
                _omega[node] = _gamma[node]+np.sum([_omega[child] for child in node.children])
                tmp = np.zeros(len(node.children))
                for i, child in enumerate(node.children):
                    if child.f == 0:
                        tmp[i] = _omega[child]
                    elif child.f == 1:
                        tmp[i] = _psi[child]
                    else:
                        tmp[i] = logsumexp([
                            np.log(child.f)+_psi[child],
                            np.log(1-child.f)+_omega[child]
                        ])
                _psi[node] = _lambda[node] + np.sum(tmp)
            llh += _psi[self.root]
            if self.ps is not None:
                r[self.ps] = np.sum(dataset[tumor_idx, self.ps.genes])
                _psi[self.ps] = r[self.ps]*log_pfp+(self.ps.s-r[self.ps])*log_ptn
                llh += _psi[self.ps]
        return(llh)

    def sample_structure(self, dataset, p_moves, log_p_moves, current_posterior, pp, error_estimation=True):
        accepted_proposal = False
        new_posterior = current_posterior
        move_type = np.random.choice(list(p_moves.keys()), p=list(p_moves.values()))
        if move_type == 'hmerge':
            proposal, forward_prob, backward_prob, novel_proposal = self.ss_hmerge(dataset, error_estimation)
            forward_prob += log_p_moves['hmerge']
            backward_prob += log_p_moves['hsplit']
        elif move_type == 'hsplit':
            proposal, forward_prob, backward_prob, novel_proposal = self.ss_hsplit(dataset, error_estimation)
            forward_prob += log_p_moves['hsplit']
            backward_prob += log_p_moves['hmerge']
        elif move_type == 'vmerge':
            proposal, forward_prob, backward_prob, novel_proposal = self.ss_vmerge(dataset, error_estimation)
            forward_prob += log_p_moves['vmerge']
            backward_prob += log_p_moves['vsplit']
        elif move_type == 'vsplit':
            proposal, forward_prob, backward_prob, novel_proposal = self.ss_vsplit(dataset, error_estimation)
            forward_prob += log_p_moves['vsplit']
            backward_prob += log_p_moves['vmerge']
        elif move_type == 'swap':
            proposal, forward_prob, backward_prob, novel_proposal = self.ss_swap(dataset, error_estimation)
        elif move_type == 'spr':
            proposal, forward_prob, backward_prob, novel_proposal = self.ss_spr(dataset, error_estimation)
        elif move_type == 'gt_d2p':
            proposal, forward_prob, backward_prob, novel_proposal = self.ss_gt_d2p(dataset, error_estimation)
            forward_prob += log_p_moves['gt_d2p']
            backward_prob += log_p_moves['gt_p2d']
        elif move_type == 'gt_p2d':
            proposal, forward_prob, backward_prob, novel_proposal = self.ss_gt_p2d(dataset, error_estimation)
            forward_prob += log_p_moves['gt_p2d']
            backward_prob += log_p_moves['gt_d2p']
        elif move_type == 'gt_d2s':
            proposal, forward_prob, backward_prob, novel_proposal = self.ss_gt_d2s(dataset, error_estimation)
            forward_prob += log_p_moves['gt_d2s']
            backward_prob += log_p_moves['gt_s2d']
        elif move_type == 'gt_s2d':
            proposal, forward_prob, backward_prob, novel_proposal = self.ss_gt_s2d(dataset, error_estimation)
            forward_prob += log_p_moves['gt_s2d']
            backward_prob += log_p_moves['gt_d2s']
        elif move_type == 'sdetach':
            proposal, forward_prob, backward_prob, novel_proposal = self.ss_sdetach(dataset, error_estimation)
            forward_prob += log_p_moves['sdetach']
            backward_prob += log_p_moves['sattach']
        elif move_type == 'sattach':
            proposal, forward_prob, backward_prob, novel_proposal = self.ss_sattach(dataset, error_estimation)
            forward_prob += log_p_moves['sattach']
            backward_prob += log_p_moves['sdetach']
        elif move_type == 'pdetach':
            proposal, forward_prob, backward_prob, novel_proposal = self.ss_pdetach(dataset, error_estimation)
            forward_prob += log_p_moves['pdetach']
            backward_prob += log_p_moves['pattach']
        elif move_type == 'pattach':
            proposal, forward_prob, backward_prob, novel_proposal = self.ss_pattach(dataset, error_estimation)
            forward_prob += log_p_moves['pattach']
            backward_prob += log_p_moves['pdetach']
        else:
            print('UNDEFINED MOVE!')
        
        if novel_proposal:
            proposal_posterior = proposal.likelihood(dataset) + proposal.prior(pp)
            ar = proposal_posterior-current_posterior-forward_prob+backward_prob
            if np.random.binomial(n=1, p=np.exp(np.min([0,ar]))):
                self=proposal
                new_posterior = proposal_posterior
                accepted_proposal = True

        return(self, new_posterior, move_type, novel_proposal, accepted_proposal)

    def ss_hmerge(self, dataset, error_estimation, debugging=False):
        # horizontal merge
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0
        leafset = [node for node in proposal.nodes if (node.is_leaf and not(node.is_ps) and not(node.is_root))]
        candidates = []
        if len(leafset)>1:
            for i in range(len(leafset)):
                for j in range(i+1, len(leafset)):
                    if leafset[i].parent == leafset[j].parent:
                        candidates.append((i, j))
        if debugging:
            print("Candidates:")
            for _tuple in candidates:
                print(leafset[_tuple[0]].genes,',',leafset[_tuple[1]].genes) 
        if len(candidates)>0:
            selected_tuple = candidates[np.random.choice(len(candidates))]
            if debugging:
                print("Selected pair:")
                print(leafset[selected_tuple[0]].genes,',',leafset[selected_tuple[1]].genes)
            forward_prob = -np.log(len(candidates))
            leafset[selected_tuple[0]].genes.extend(leafset[selected_tuple[1]].genes)
            new_n_genes = len(leafset[selected_tuple[0]].genes)
            leafset[selected_tuple[1]].parent = None
            proposal.nodes.remove(leafset[selected_tuple[1]])
            bk_candidates = [node for node in proposal.nodes if (node.is_leaf and len(node.genes)>1 and not(node.is_ps) and not(node.is_root))]
            if debugging:
                print("Backward candidates:")
                for _node in bk_candidates:
                    print(_node.genes)
            if new_n_genes>10: # for numerical stability
                backward_prob = -np.log(len(bk_candidates))-((new_n_genes-1)*np.log(2))
            else:
                backward_prob = -np.log(len(bk_candidates))-np.log((2**new_n_genes-2)/2)
            if error_estimation:
                proposal = proposal.assign_error_values(dataset)
            proposal = proposal.assign_f_values(dataset)
            novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def ss_hsplit(self, dataset, error_estimation, debugging=False):
        # horizontal split
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0        
        candidates = [node for node in proposal.nodes if (node.is_leaf and len(node.genes)>1 and not(node.is_ps) and not(node.is_root))]
        if debugging:
            print("Node candidates:")
            for _node in candidates:
                print(_node.genes)
        if len(candidates)>0:
            selected_node = np.random.choice(candidates)
            if debugging:
                print("Selected node:")
                print(selected_node.genes)
            selected_subset = []
            while len(selected_subset)==0 or len(selected_subset)==len(selected_node.genes):
                selected_subset = []
                for gene in selected_node.genes:
                    if np.random.binomial(n=1, p=0.5):
                        selected_subset.append(gene)
            if debugging:
                print("Selected subset:")
                print(selected_subset)
            if len(selected_node.genes)>10: # for numerical stability
                forward_prob = -np.log(len(candidates))-((len(selected_node.genes)-1)*np.log(2))
            else:
                forward_prob = -np.log(len(candidates))-np.log((2**(len(selected_node.genes))-2)/2)
            for gene in selected_subset:
                selected_node.genes.remove(gene)
            proposal.nodes.append(OncoNode(genes=selected_subset, f=0.5, parent=selected_node.parent))
            new_leafset = [node for node in proposal.nodes if (node.is_leaf and not(node.is_ps) and not(node.is_root))]
            bk_candidates = []
            if len(new_leafset)>1:
                for i in range(len(new_leafset)):
                    for j in range(i+1, len(new_leafset)):
                        if new_leafset[i].parent == new_leafset[j].parent:
                            bk_candidates.append((i, j))
            if debugging:
                print("Backward candidates:")
                for _tuple in bk_candidates:
                    print(new_leafset[_tuple[0]].genes,',',new_leafset[_tuple[1]].genes)
            backward_prob = -np.log(len(bk_candidates))
            if error_estimation:
                proposal = proposal.assign_error_values(dataset)
            proposal = proposal.assign_f_values(dataset)
            novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def ss_vmerge(self, dataset, error_estimation, debugging=False):
        # vertival merge
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0 
        candidates = [node for node in proposal.nodes if (node.is_leaf and not(node.is_ps) and not(node.is_root) and not(node.parent.is_root))]
        if debugging:
            print("Candidates:")
            for _node in candidates:
                print(_node.genes)
        if len(candidates)>0:
            selected_node = np.random.choice(candidates)
            if debugging:
                print("Selected node:")
                print(selected_node.genes)
            forward_prob = -np.log(len(candidates))
            selected_node.parent.genes.extend(selected_node.genes)
            new_n_genes = len(selected_node.parent.genes)
            selected_node.parent = None
            proposal.nodes.remove(selected_node)
            bk_candidates = [node for node in proposal.nodes if (len(node.genes)>1 and not(node.is_ps))]
            if debugging:
                print("Backward candidates:")
                for _node in bk_candidates:
                    print(_node.genes)
            if new_n_genes>10: # for numerical stability
                backward_prob = -np.log(len(bk_candidates))-(new_n_genes*np.log(2))
            else:
                backward_prob = -np.log(len(bk_candidates))-np.log(2**new_n_genes-2)
            if error_estimation:
                proposal = proposal.assign_error_values(dataset)
            proposal = proposal.assign_f_values(dataset)
            novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def ss_vsplit(self, dataset, error_estimation, debugging=False):
        # vertical split
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0 
        candidates = [node for node in proposal.nodes if (len(node.genes)>1 and not(node.is_ps))]
        if debugging:
            print("Candidates:")
            for _node in candidates:
                print(_node.genes)
        if len(candidates)>0:
            selected_node = np.random.choice(candidates)
            if debugging:
                print("Selected node:")
                print(selected_node.genes)
            selected_subset = []
            while len(selected_subset)==0 or len(selected_subset)==len(selected_node.genes):
                selected_subset = []
                for gene in selected_node.genes:
                    if np.random.binomial(n=1, p=0.5):
                        selected_subset.append(gene)
            if debugging:
                print("Selected subset:")
                print(selected_subset)
            if len(selected_node.genes)>10: # for numerical stability
                forward_prob = -np.log(len(candidates))-((len(selected_node.genes))*np.log(2))
            else:
                forward_prob = -np.log(len(candidates))-np.log(2**(len(selected_node.genes))-2)
            for gene in selected_subset:
                selected_node.genes.remove(gene)
            proposal.nodes.append(OncoNode(genes=selected_subset, f=0.5, parent=selected_node))
            bk_candidates = [node for node in proposal.nodes if (node.is_leaf and not(node.is_ps) and not(node.is_root) and not(node.parent.is_root))]
            if debugging:
                print("Backward candidates:")
                for _node in bk_candidates:
                    print(_node.genes)
            backward_prob = - np.log(len(bk_candidates))
            if error_estimation:
                proposal = proposal.assign_error_values(dataset)
            proposal = proposal.assign_f_values(dataset)
            novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def ss_swap(self, dataset, error_estimation, debugging=False):
        # Sample Structure - swap the genes of two connected nodes
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0 
        candidates = [node for node in proposal.nodes if ((node.parent is not None) and not(node.parent.is_root))]
        if debugging:
            print("Candidates:")
            for _node in candidates:
                print(_node.genes)
        if len(candidates)>0:
            selected_child = np.random.choice(candidates)
            if debugging:
                print("Selected node:")
                print(selected_child.genes)
            child_genes = deepcopy(selected_child.genes)
            selected_parent = selected_child.parent
            selected_child.genes = deepcopy(selected_parent.genes)
            selected_parent.genes = deepcopy(child_genes)
            if error_estimation:
                proposal = proposal.assign_error_values(dataset)
            proposal = proposal.assign_f_values(dataset)
            novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def ss_spr(self, dataset, error_estimation, debugging=False):
        # Sample Structure - Subtree Pruning and Regrafting
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0 
        candidates = [node for node in proposal.nodes if ((node.parent is not None) and not(node.is_simple))]
        if debugging:
            print("Candidates:")
            for _node in candidates:
                print(_node.genes)
        if len(candidates)>0:
            selected_node = np.random.choice(candidates)
            if debugging:
                print("Selected node:")
                print(selected_node.genes)
            subtree_nodes = list(selected_node.descendants)
            subtree_nodes.append(selected_node)
            parent_candidates = [node for node in proposal.nodes if (
                not(node in subtree_nodes) and not(node.is_ps) and not((len(selected_node.genes)==1) and (len(selected_node.children)==0) and (node.is_root))
                )]
            if debugging:
                print("Parent candidates:")
                for _node in parent_candidates:
                    print(_node.genes)
            selected_parent = np.random.choice(parent_candidates)
            if debugging:
                print("Selected parent:")
                print(selected_parent.genes)
            if selected_parent != selected_node.parent:
                selected_node.parent = selected_parent
                if error_estimation:
                    proposal = proposal.assign_error_values(dataset)
                proposal = proposal.assign_f_values(dataset)
                novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def ss_gt_d2p(self, dataset, error_estimation, debugging=False):
        # Sample Structure - Gene transfer, driver to passenger
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0 
        candidates = [node for node in proposal.nodes if (len(node.genes)>1 and not(node.is_ps))]
        if debugging:
            print("Candidates:")
            for _node in candidates:
                print(_node.genes)
        if len(candidates)>0:
            selected_node = np.random.choice(candidates)
            if debugging:
                print("Selected node:")
                print(selected_node.genes)
            selected_gene = np.random.choice(selected_node.genes)
            if debugging:
                print("Selected gene:")
                print(selected_gene)
            forward_prob = -np.log(len(candidates))-np.log(len(selected_node.genes))
            selected_node.genes.remove(selected_gene)
            proposal.ps.genes.append(selected_gene)
            backward_prob = -np.log(len(proposal.ps.genes))-np.log(len(proposal.nodes)-2)
            if error_estimation:
                proposal = proposal.assign_error_values(dataset)
            proposal = proposal.assign_f_values(dataset)
            novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)
    
    def ss_gt_p2d(self, dataset, error_estimation, debugging=False):
        # Sample Structure - Gene transfer, passenger to driver
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0
        if debugging:
            print("Candidate genes:")
            print(proposal.ps.genes)
        if len(proposal.ps.genes)>0:
            selected_gene = np.random.choice(proposal.ps.genes)
            if debugging:
                print("Selected gene:")
                print(selected_gene)
            candidates = [node for node in proposal.nodes if (not(node.is_ps) and not(node.is_root))]
            if debugging:
                print("Candidates:")
                for _node in candidates:
                    print(_node.genes)
            if len(candidates)>0:
                selected_node = np.random.choice(candidates)
                if debugging:
                    print("Selected node:")
                    print(selected_node.genes)
                forward_prob = -np.log(len(proposal.ps.genes))-np.log(len(candidates))
                selected_node.genes.append(selected_gene)
                proposal.ps.genes.remove(selected_gene)
                bk_candidates = [node for node in proposal.nodes if (len(node.genes)>1 and not(node.is_ps))]
                if debugging:
                    print("Backward candidates:")
                    for _node in bk_candidates:
                        print(_node.genes)
                backward_prob = -np.log(len(bk_candidates))-np.log(len(selected_node.genes))
                if error_estimation:
                    proposal = proposal.assign_error_values(dataset)
                proposal = proposal.assign_f_values(dataset)
                novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def ss_gt_d2s(self, dataset, error_estimation, debugging=False):
        # Sample Structure - Gene transfer, driver to simple node
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0 
        candidates = [node for node in proposal.nodes if (len(node.genes)>1 and not(node.is_ps) and not((node.parent.is_root) and (node.is_leaf)))]
        if debugging:
            print("Candidates:")
            for _node in candidates:
                print(_node.genes)
        if len(candidates)>0:
            selected_node = np.random.choice(candidates)
            if debugging:
                print("Selected node:")
                print(selected_node.genes)
            selected_gene = np.random.choice(selected_node.genes)
            if debugging:
                print("Selected gene:")
                print(selected_gene)
            forward_prob = -np.log(len(candidates))-np.log(len(selected_node.genes))
            selected_node.genes.remove(selected_gene)
            proposal.nodes.append(OncoNode(genes=[selected_gene], f=0.5, parent=proposal.root))
            bk_list_of_simple_nodes = [node for node in proposal.nodes if node.is_simple]
            if debugging:
                print("List of simple nodes for backward move:")
                for _node in bk_list_of_simple_nodes:
                    print(_node.genes)
            bk_list_of_dest_nodes = []
            for node in proposal.nodes:
                if node.parent is not None:
                    if not(node.parent.is_root and (node.is_leaf)):
                        bk_list_of_dest_nodes.append(node)
            if debugging:
                print("List of dest nodes for backward move:")
                for _node in bk_list_of_dest_nodes:
                    print(_node.genes)
            backward_prob = -np.log(len(bk_list_of_simple_nodes))-np.log(len(bk_list_of_dest_nodes))
            if error_estimation:
                proposal = proposal.assign_error_values(dataset)
            proposal = proposal.assign_f_values(dataset)
            novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)
    
    def ss_gt_s2d(self, dataset, error_estimation, debugging=False):
        # Sample Structure - Gene transfer, simple node to driver
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0
        list_of_simple_nodes = [node for node in proposal.nodes if node.is_simple]
        if debugging:
            print("List of simple nodes:")
            for _node in list_of_simple_nodes:
                print(_node.genes)
        if len(list_of_simple_nodes)>0:
            selected_node = np.random.choice(list_of_simple_nodes)
            if debugging:
                print("Selected node:")
                print(selected_node.genes)
            candidates = []
            for node in proposal.nodes:
                if node.parent is not None:
                    if not(node.parent.is_root and (node.is_leaf)):
                        candidates.append(node)
            if len(candidates)>0:
                if debugging:
                    print("List of destination candidates:")
                    for _node in candidates:
                        print(_node.genes)
                selected_dest = np.random.choice(candidates)
                if debugging:
                    print("Selected destination node:")
                    print(selected_dest.genes)
                forward_prob = -np.log(len(list_of_simple_nodes))-np.log(len(candidates))
                selected_dest.genes.extend(selected_node.genes)
                selected_node.parent = None
                proposal.nodes.remove(selected_node)
                bk_candidates = [node for node in proposal.nodes if (len(node.genes)>1 and not(node.is_ps) and not((node.parent.is_root) and (node.is_leaf)))]
                if debugging:
                    print("Backward candidates:")
                    for _node in bk_candidates:
                        print(_node.genes)
                backward_prob = -np.log(len(bk_candidates))-np.log(len(selected_dest.genes))
                if error_estimation:
                    proposal = proposal.assign_error_values(dataset)
                proposal = proposal.assign_f_values(dataset)
                novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def ss_sdetach(self, dataset, error_estimation, debugging=False):
        # prune a leaf (not in the first layer) and put its genes into simple nodes
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0 
        candidates = []
        for node in proposal.nodes:
            if (node.is_leaf and not(node.is_ps) and not(node.is_root)):
                if not(node.parent.is_root and (len(node.genes)<3)):
                    candidates.append(node)
        if debugging:
            print("Candidates:")
            for _node in candidates:
                print(_node.genes)
        if len(candidates)>0:
            selected_node = np.random.choice(candidates)
            if debugging:
                print("Selected node:")
                print(selected_node.genes)
            n_genes_for_bk = len(selected_node.genes) # Variable used for backward_prob calculations
            forward_prob = -np.log(len(candidates))
            for gene in selected_node.genes:
                proposal.nodes.append(OncoNode(genes=[gene], f=0.5, parent=proposal.root))
            selected_node.genes = []
            selected_node.parent = None
            proposal.nodes.remove(selected_node)
            bk_list_of_simple_nodes = [node for node in proposal.nodes if node.is_simple]
            if debugging:
                print("Simple nodes for backward move:")
                for _node in bk_list_of_simple_nodes:
                    print(_node.genes)
            if n_genes_for_bk < 3:
                n_cases_for_bk_dest = len(proposal.nodes)-2
            else:
                n_cases_for_bk_dest = len(proposal.nodes)-1
            n_cases_for_bk_dest = n_cases_for_bk_dest - n_genes_for_bk # Excluding the selected nodes themselves
            if len(bk_list_of_simple_nodes)>10:  # for numerical stability
                backward_prob = -np.log(n_cases_for_bk_dest)-((len(bk_list_of_simple_nodes))*np.log(2))
            else:
                backward_prob = -np.log(n_cases_for_bk_dest)-np.log(2**(len(bk_list_of_simple_nodes))-1)
            if error_estimation:
                proposal = proposal.assign_error_values(dataset)
            proposal = proposal.assign_f_values(dataset)
            novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def ss_sattach(self, dataset, error_estimation, debugging=False):
        # attach a new leaf including genes from simple nodes (cannot be root's child)
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0 
        list_of_simple_nodes = [node for node in proposal.nodes if node.is_simple]
        if debugging:
            print("List of simple nodes:")
            for _node in list_of_simple_nodes:
                print(_node.genes)
        if len(list_of_simple_nodes)>0:
            selected_subset = []
            while len(selected_subset)==0:
                for node in list_of_simple_nodes:
                    if np.random.binomial(n=1, p=0.5):
                        selected_subset.append(node)
            if debugging:
                print("Selected subset:")
                for _node in selected_subset:
                    print(_node.genes)
            if len(selected_subset) < 3:
                dest_candidates = [node for node in proposal.nodes if (not(node.is_ps) and not(node.is_root) and not(node in selected_subset))]
            else:
                dest_candidates = [node for node in proposal.nodes if (not(node.is_ps) and not(node in selected_subset))]
            if debugging:
                print("Candidates for parent node:")
                for _node in dest_candidates:
                    print(_node.genes)
            if len(dest_candidates)>0:
                selected_node = np.random.choice(dest_candidates)
                if debugging:
                    print("Selected parent:")
                    print(selected_node.genes)
                if len(list_of_simple_nodes)>10: # for numerical stability
                    forward_prob = -np.log(len(dest_candidates))-((len(list_of_simple_nodes))*np.log(2))
                else:
                    forward_prob = -np.log(len(dest_candidates))-np.log(2**(len(list_of_simple_nodes))-1)
                genes_set = []
                for node in selected_subset:
                    genes_set.extend(node.genes)
                    node.parent = None
                    proposal.nodes.remove(node)
                proposal.nodes.append(OncoNode(genes=genes_set, f=0.5, parent=selected_node))
                bk_candidates = []
                for node in proposal.nodes:
                    if (node.is_leaf and not(node.is_ps)):
                        if not(node.parent.is_root and (len(node.genes)<3)):
                            bk_candidates.append(node)
                if debugging:
                    print("Backward candidates:")
                    for _node in bk_candidates:
                        print(_node.genes)
                backward_prob = -np.log(len(bk_candidates))
                if error_estimation:
                    proposal = proposal.assign_error_values(dataset)
                proposal = proposal.assign_f_values(dataset)
                novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def ss_pdetach(self, dataset, error_estimation, debugging=False):
        # prune a leaf (can be in the first layer) and put its genes into the set of passengers
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0 
        candidates = [node for node in proposal.nodes if (node.is_leaf and not(node.is_root) and not(node.is_ps))]
        if debugging:
            print("Candidates:")
            for _node in candidates:
                print(_node.genes)
        if len(candidates)>0:
            selected_node = np.random.choice(candidates)
            if debugging:
                print("Selected node:")
                print(selected_node.genes)
            forward_prob = -np.log(len(candidates))
            proposal.ps.genes.extend(selected_node.genes)
            selected_node.genes = []
            selected_node.parent = None
            proposal.nodes.remove(selected_node)
            if len(proposal.ps.genes)>10: # for numerical stability
                backward_prob = -np.log(len(proposal.nodes)-1)-((len(proposal.ps.genes))*np.log(2))
            else:
                backward_prob = -np.log(len(proposal.nodes)-1)-np.log(2**(len(proposal.ps.genes))-1)
            if error_estimation:
                proposal = proposal.assign_error_values(dataset)
            proposal = proposal.assign_f_values(dataset)
            novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def ss_pattach(self, dataset, error_estimation, debugging=False):
        # attach a new leaf including genes from the set of passengers (can be root's child)
        novel_proposal = False
        proposal = deepcopy(self)
        forward_prob = 0
        backward_prob = 0 
        if len(proposal.ps.genes)>0:
            selected_subset = []
            while len(selected_subset)==0:
                for gene in proposal.ps.genes:
                    if np.random.binomial(n=1, p=0.5):
                        selected_subset.append(gene)
            if debugging:
                print("Selected subset of passengers:")
                print(selected_subset)
            dest_candidates = [node for node in proposal.nodes if not(node.is_ps)]
            if debugging:
                print("Candidate nodes:")
                for _node in dest_candidates:
                    print(_node.genes)
            selected_node = np.random.choice(dest_candidates)
            if debugging:
                print("Selected node:")
                print(selected_node.genes)
            if len(proposal.ps.genes)>10: # for numerical stability
                forward_prob = -np.log(len(dest_candidates))-((len(proposal.ps.genes))*np.log(2))
            else:
                forward_prob = -np.log(len(dest_candidates))-np.log(2**(len(proposal.ps.genes))-1)
            for gene in selected_subset:
                proposal.ps.genes.remove(gene)
            proposal.nodes.append(OncoNode(genes=selected_subset, f=0.5, parent=selected_node))
            bk_candidates = [node for node in proposal.nodes if (node.is_leaf and not(node.is_root) and not(node.is_ps))]
            if debugging:
                print("Backward candidates")
                for _node in bk_candidates:
                    print(_node.genes)
            backward_prob = -np.log(len(bk_candidates))
            if error_estimation:
                proposal = proposal.assign_error_values(dataset)
            proposal = proposal.assign_f_values(dataset)
            novel_proposal = True
        return(proposal, forward_prob, backward_prob, novel_proposal)

    def fast_training_iteration(self, dataset, n_iters, pp, seed=None, current_posterior=None, p_moves=None, collapse_interval=10000):
        # hardcoding to collapse every 1000 samples
        # if n_iters is divisable by 1000, the output will be already pruned
        collapse_interval = collapse_interval
        if seed is not None:
            np.random.seed(seed)
        if p_moves is None:
            p_moves = {
                'hmerge': 1,
                'hsplit': 1,
                'vmerge': 1,
                'vsplit': 1,
                'swap': 1,
                'spr': 1,
                'gt_d2p': 1,
                'gt_p2d': 1,
                'gt_d2s': 1,
                'gt_s2d': 1,
                'sdetach': 1,
                'sattach': 1,
                'pdetach': 1,
                'pattach': 1
            }
            # Normalization:
            factor=1.0/sum(p_moves.values())
            for k in p_moves:
                p_moves[k] = p_moves[k]*factor
        log_p_moves = {k: np.log(v) for k, v in p_moves.items() if v>0}
        if current_posterior is None:
            current_posterior = self.likelihood(dataset)+self.prior(pp)
        # To record move-specific stats #
        n_proposed = {k: 0 for k, _ in p_moves.items()}
        n_novel = {k: 0 for k, _ in p_moves.items()}
        n_accepted = {k: 0 for k, _ in p_moves.items()}
        # Main outputs #
        posteriors_list = []
        best_sample = deepcopy(self)
        best_posterior = current_posterior
        n_updates = 0
        for _iter in range(n_iters):
            if (_iter+1) % collapse_interval == 0: # special iteration!
                # collapsing the tree
                self = self.prune(dataset)
                new_posterior = self.likelihood(dataset) + self.prior(pp)
            else: # normal iteration
                self, new_posterior, move_type, novel_proposal, accepted_proposal = self.sample_structure(dataset, p_moves, log_p_moves, current_posterior, pp)
                n_proposed[move_type] += 1
                if novel_proposal:
                    n_novel[move_type] += 1
                if accepted_proposal:
                    n_accepted[move_type] += 1
                    n_updates+=1
            # in any iteration we have:
            posteriors_list.append(new_posterior)
            current_posterior = new_posterior 
            if new_posterior>best_posterior:
                best_sample = deepcopy(self)
                best_posterior = deepcopy(new_posterior)
        details = {
            'n_proposed': n_proposed,
            'n_novel': n_novel,
            'n_accepted': n_accepted
        }
        return(self, current_posterior, best_sample, best_posterior, posteriors_list, n_updates, details)
