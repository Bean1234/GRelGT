# -*- coding: utf-8 -*-

import os, sys
import argparse
import logging
from warnings import simplefilter
import torch
from scipy.sparse import SparseEfficiencyWarning
import numpy as np
import random
import json
import dgl

sys.path.append(os.path.join(os.path.dirname(__file__), f'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), f'model'))

from utils.initialization_utils import initialize_experiment, initialize_model
from utils.process_utils import generate_subgraph_datasets, process_files
from utils.data_utils import SubgraphDataset
from utils.batch_utils import collate_dgl, move_batch_to_device_dgl, send_graph_to_device
from model.gcn_lg2.graph_classifier import GraphClassifier as dgl_model
from utils.graph_utils import ssp_multigraph_to_dgl, ssp_multigraph_to_dgl_new, create_line_graph_etype


from trainer import Trainer
from evaluator import Evaluator



def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)
    params.db_path = os.path.join(params.main_dir, f'../../{params.data_dir}/{params.dataset}/subgraphs_en_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}')
    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)

    relation2id_path = os.path.join(params.main_dir, f'../../{params.data_dir}/{params.dataset}/relation2id.json')  
    with open(relation2id_path, 'r') as f:
        relation2id = json.load(f)
    id2revid = {}  

    train = SubgraphDataset(params.db_path, id2revid, 'train_pos', 'train_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.train_file)
    valid = SubgraphDataset(params.db_path, id2revid, 'valid_pos', 'valid_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.valid_file)

    params.num_rels = train.num_rels
    params.aug_num_rels = train.aug_num_rels
    params.inp_dim = train.n_feat_dim
    params.max_label_value = train.max_n_label

    logging.info(f"Device: {params.device}")
    logging.info(f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")

    graph_classifier = initialize_model(params, dgl_model, params.load_model)

    
    adj_list, _, _, _, _, _ = process_files(params.file_paths, None)
    graph, in_nodes, out_nodes = ssp_multigraph_to_dgl_new(adj_list)
    
    line_graph = dgl.line_graph(graph, shared = True)
    line_graph = create_line_graph_etype(line_graph, in_nodes, out_nodes)
    
    in_deg = line_graph.in_degrees(range(line_graph.number_of_nodes())).float().numpy()
    in_deg[in_deg == 0] = 1  
    node_norm = 1.0 / in_deg
    line_graph.ndata['norm'] = node_norm
    line_graph = send_graph_to_device(line_graph, params.device)



    valid_evaluator = Evaluator(params, graph_classifier, line_graph, valid)
    trainer = Trainer(params, graph_classifier, train, line_graph, valid_evaluator)

    logging.info('Starting training with full batch...')

    trainer.train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="default",  # WN18RR  fb237
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str, default='fb237_v1', help="Dataset string")
    parser.add_argument("--data_dir", "-dr", type=str, default='data', help="data directory")
    parser.add_argument("--gpu", "-g", type=int, default=0, help="Which GPU to use?")
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid",
                        help="Name of file containing validation triplets")
    parser.add_argument('--load_model', action='store_true',
                        help='Load existing model?')

    # Training regime params
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate of the optimizer")
    parser.add_argument("--l2", type=float, default=0.0001,
                        help="Regularization constant for GNN weights")
    parser.add_argument("--margin", type=float, default=8,  # 8
                        help="The margin between positive and negative samples in the max-margin loss")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", "-ne", type=int, default=30,
                        help="Learning rate of the optimizer")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Interval of epochs to evaluate the model?")
    parser.add_argument("--eval_every_iter", type=int, default=67,
                        help="Interval of iterations to evaluate the model?")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Interval of epochs to save a checkpoint of the model?")
    parser.add_argument("--early_stop", type=int, default=100000,
                        help="Early stopping patience")

    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Which optimizer to use?")
    parser.add_argument("--clip", type=int, default=1000,
                        help="Maximum gradient norm allowed")

    # Data processing pipeline params
    parser.add_argument("--hop", type=int, default=3,  # 3
                        help="Enclosing subgraph hop number")
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='whether to append adj matrix list with symmetric relations')
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')
    parser.add_argument("--num_workers", type=int, default=6,
                        help="Number of dataloading processes")
    parser.add_argument("--max_links", type=int, default=1000000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--model_type', '-m', type=str, choices=['ssp', 'dgl'], default='dgl',
                        help='what format to store subgraphs in for model')
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0,
                        help='with what probability to sample constrained heads/tails while neg sampling')

    # Model params
    parser.add_argument("--rel_emb_dim", "-r_dim", type=int, default=32,
                        help="Relation embedding size")
    parser.add_argument("--attn_rel_emb_dim", "-ar_dim", type=int, default=32,
                        help="Relation embedding size for attention")
    parser.add_argument("--emb_dim", "-dim", type=int, default=32,
                        help="Entity embedding size")
    parser.add_argument("--num_gcn_layers", "-l", type=int, default=3,  # 2
                        help="Number of GCN layers")
    parser.add_argument("--num_bases", "-b", type=int, default=4,
                        help="Number of basis functions to use for GCN weights")
    parser.add_argument("--dropout", type=float, default=0.0,  # 0.0
                        help="Dropout rate in GNN layers")
    parser.add_argument("--edge_dropout", type=float, default=0.1,
                        help="Dropout rate in edges of the subgraphs")
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='sum',
                        help='what type of aggregation to do in gnn msg passing')
    parser.add_argument('--cg_agg_type', '-cg', type=str, choices=['mean', 'max'], default='max', help='context graph aggregation type')
    parser.add_argument('--add_ht_emb', '-ht', type=bool, default=True,
                        help='whether to concatenate head/tail embedding with pooled graph representation')
    parser.add_argument('--has_attn', '-attn', action='store_true', default=True,
                        help='whether to have attn in model or not')

    parser.add_argument('--no_jk', action='store_true', default=True,
                        help='Disable JK connection')
    parser.add_argument("--loss", type=int, default=0,
                        help='0,1 correspond ')
    parser.add_argument('--critic', type=int, default=1,  # 1
                        help='0,1,2 correspond to auc, auc_pr, mrr')
    parser.add_argument('--epoch', type=int, default=0,
                        help='to record epoch')
    parser.add_argument('--ablation', type=int, default=0,
                        help='0,1,2,3 correspond to normal, no-sub, no-ent, only-rel')

    params = parser.parse_args()
    initialize_experiment(params, __file__)

    params.file_paths = {
        'train': os.path.join(params.main_dir, '../../{}/{}/{}.txt'.format(params.data_dir, params.dataset, params.train_file)),
        'valid': os.path.join(params.main_dir, '../../{}/{}/{}.txt'.format(params.data_dir, params.dataset, params.valid_file))
    }

    params.device = torch.device('cuda:%d' % params.gpu)

    params.collate_fn = collate_dgl
    params.move_batch_to_device = move_batch_to_device_dgl

    main(params)
