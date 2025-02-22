# -*- coding: utf-8 -*-
# 图操作相关
import numpy as np
import scipy.sparse as ssp
import random
import networkx as nx
import torch
import dgl
import copy 
import itertools
from tqdm import tqdm
import logging

from save_utils import serialize

def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs.
    Modified from dgl.contrib.data.knowledge_graph to accomodate node sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)


def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def incidence_matrix(adj_list):  
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def get_average_subgraph_size(sample_size, links, A, params):
    total_size = 0
    for (n1, n2, r_label) in links[np.random.choice(len(links), sample_size)]:
        nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A, params.hop, params.enclosing_sub_graph, params.max_nodes_per_hop)
        datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
        total_size += len(serialize(datum))
    return total_size / sample_size


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T  

    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)  
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)  

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_sub_graph:  
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)  

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]  

    labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=h)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]

    if max_node_label_value is not None: 
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    subgraph_size = len(pruned_subgraph_nodes)
    enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
    num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)

    return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes


def node_label(subgraph, max_distance=1):  
    roots = [0, 1]  # 
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)  #

    target_node_labels = np.array([[0, 1], [1, 0]])  # 
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels  

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]  
    return labels, enclosing_subgraph_nodes


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]

def ssp_multigraph_to_dgl(graph, n_feats=None):  
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    # make dgl graph
    g_dgl = dgl.DGLGraph(multigraph=True)
    g_dgl.from_networkx(g_nx, edge_attrs=['type'])
    # add node features
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl


def ssp_multigraph_to_dgl_new(graph, n_feats=None):  
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """
    in_nodes = [[] for i in range(graph[0].shape[0])]
    out_nodes = copy.deepcopy(in_nodes)
    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    e_id = 0
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel, 'id': e_id}))
            out_nodes[src].append(e_id)
            in_nodes[dst].append(e_id)
            e_id = e_id + 1
        g_nx.add_edges_from(nx_triplets)

    g_dgl = dgl.DGLGraph(multigraph=True)
    g_dgl.from_networkx(g_nx, edge_attrs=['type', 'id'])

    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl, in_nodes, out_nodes


def create_line_graph_etype(graph, in_nodes, out_nodes):  
    graph.edata["etype"] = torch.tensor([0]*graph.edges()[0].shape[0])
    edge_num = graph.edges()[1].shape[0]
    graph.add_edges(graph.edges()[1], graph.edges()[0], {'etype': torch.tensor([1]*edge_num)})

    _, indices = torch.sort(graph.ndata["id"])

    
    for nodes in tqdm(out_nodes, total = len(out_nodes)):
        src = [] 
        dst = []
        for node_pair in itertools.product(nodes, nodes):
                if node_pair[0] != node_pair[1]:
                    src.append(indices[node_pair[0]])
                    dst.append(indices[node_pair[1]])
        if src != [] and dst != []:
            graph.add_edges(src, dst, {'etype': torch.tensor([2]*len(src))})

    for nodes in tqdm(in_nodes, total = len(in_nodes)):
        src = []
        dst = []
        for node_pair in itertools.product(nodes, nodes):
                if node_pair[0] != node_pair[1]:
                    src.append(indices[node_pair[0]])
                    dst.append(indices[node_pair[1]])
        if src != [] and dst != []:
            graph.add_edges(src, dst, {'etype': torch.tensor([3]*len(src))})
    return graph
