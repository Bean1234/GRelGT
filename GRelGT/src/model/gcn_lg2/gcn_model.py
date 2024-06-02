"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import RGCNBasisLayer as RGCNLayer
from .layers import LGCNLayer

from .aggregators import SumAggregator, MLPAggregator, GRUAggregator


class UniGCN(nn.Module): 
    def __init__(self, params):
        super(UniGCN, self).__init__()

        # self.max_label_value = params.max_label_value
        self.inp_dim = params.inp_dim
        self.emb_dim = params.emb_dim
        self.attn_rel_emb_dim = params.attn_rel_emb_dim  # g图
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        self.num_bases = params.num_bases
        self.num_hidden_layers = params.num_gcn_layers
        self.dropout = params.dropout
        self.edge_dropout = params.edge_dropout
        # self.aggregator_type = params.gnn_agg_type
        self.has_attn = params.has_attn  # g图
        self.no_jk = params.no_jk

        self.device = params.device
        self.batch_size = params.batch_size

        self.rel_emb = nn.Parameter(torch.Tensor(params.num_rels + 1, params.rel_emb_dim))  
        nn.init.xavier_uniform_(self.rel_emb, gain=nn.init.calculate_gain('relu'))

        self.line1_ent = nn.Linear(self.inp_dim, self.emb_dim) 
        self.line1_rel = nn.Linear(self.emb_dim*3, self.emb_dim) 
        if params.gnn_agg_type == "sum":
            self.aggregator = SumAggregator(self.emb_dim)
        elif params.gnn_agg_type == "mlp":
            self.aggregator = MLPAggregator(self.emb_dim)
        elif params.gnn_agg_type == "gru":
            self.aggregator = GRUAggregator(self.emb_dim)

        self.build_model()

    def build_model(self):
        self.layers_rgcn = nn.ModuleList()

        self.layers_rgcn.append(self.build_input_layer())
        for idx in range(self.num_hidden_layers - 1):
            self.layers_rgcn.append(self.build_hidden_layer())

    def build_input_layer(self):  
        return RGCNLayer(self.emb_dim, self.emb_dim, self.aggregator, self.attn_rel_emb_dim, self.aug_num_rels, self.num_bases , device=self.device, activation=F.relu,
                         dropout=self.dropout, edge_dropout=self.edge_dropout, is_input_layer=True, has_attn=self.has_attn, no_jk=self.no_jk)

    def build_hidden_layer(self):
        return RGCNLayer(self.emb_dim, self.emb_dim, self.aggregator, self.attn_rel_emb_dim, self.aug_num_rels, self.num_bases, device=self.device, activation=F.relu,
                         dropout=self.dropout, edge_dropout=self.edge_dropout, has_attn=self.has_attn, no_jk=self.no_jk)

    def forward(self, g, norm, x_input, rel_labels):
        path_agg = None

        index_offset = 1
        batch_edges = g.batch_num_edges
        edge_types = g.edata['type'] + index_offset
        edge_rel_emd = []
        target_rel_emd_new = []
        index_start = 0
        index_end = 0
        
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1) 
        head_embs = g.ndata['feat'][head_ids]  # repr
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['feat'][tail_ids]
        head_embs_new = []
        tail_embs_new = []

        target_rel_emd = []
        for j in rel_labels:
            target_rel_emd.append(x_input[j+1].unsqueeze(0))
        
        target_rel_emd = torch.cat(target_rel_emd, dim=0)

        for kk, num_edges in enumerate(batch_edges):
            index_end = index_start + num_edges
        
            temp_edge_rel_emd = torch.index_select(x_input, dim=0,
                                                    index=edge_types[index_start:index_end])
            edge_rel_emd.append(temp_edge_rel_emd)

            temp_tar = [target_rel_emd[kk]] * num_edges
            target_rel_emd_new.extend(temp_tar)
            temp_tar = [head_embs[kk]] * num_edges
            head_embs_new.extend(temp_tar)
            temp_tar = [tail_embs[kk]] * num_edges
            tail_embs_new.extend(temp_tar)

            index_start = index_end


        edge_rel_emd = torch.cat(edge_rel_emd, dim=0).detach()
        target_rel_emd_new = torch.stack(target_rel_emd_new, dim=0).detach()
        for i in range(self.num_hidden_layers):
            rgcn_layer = self.layers_rgcn[i]

            if i == 0:
                g_feats = self.line1_ent(g.ndata['feat']) 
                g.ndata['feat'] = g_feats 
                g.edata["lg_feat"] = edge_rel_emd

            rgcn_layer(g, norm, edge_rel_emd, target_rel_emd_new)  # h

            if i != 0 and self.no_jk == False:  
                target_rel = torch.cat([target_rel, target_rel_emd], dim=1)
            else:
                target_rel = target_rel_emd
        return target_rel, path_agg

class LGCN(nn.Module):  
    def __init__(self, params):
        super(LGCN, self).__init__()

        # self.max_label_value = params.max_label_value
        self.inp_dim = params.inp_dim
        self.emb_dim = params.emb_dim
        self.attn_rel_emb_dim = params.attn_rel_emb_dim  
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        self.num_bases = params.num_bases
        self.num_hidden_layers = params.num_gcn_layers
        self.dropout = params.dropout
        self.edge_dropout = params.edge_dropout
        self.has_attn = params.has_attn  
        self.no_jk = params.no_jk

        self.device = params.device
        self.batch_size = params.batch_size

        self.num_lgcn_layers = 3

        self.rel_emb = nn.Parameter(torch.Tensor(params.num_rels + 1, params.rel_emb_dim))  
        # torch.nn.init.normal_(self.rel_emb)
        nn.init.xavier_uniform_(self.rel_emb, gain=nn.init.calculate_gain('relu'))

        self.line1_ent = nn.Linear(self.inp_dim, self.emb_dim)  
        self.line1_rel = nn.Linear(self.emb_dim*3, self.emb_dim)  

        self.build_model()


    def build_model(self):
        self.layers_lgcn = nn.ModuleList()
        for idx in range(self.num_lgcn_layers):
            self.layers_lgcn.append(LGCNLayer(self.emb_dim))


    def get_first_lg_feat(self, lg):  # mean max

        index_offset = 1  
        index_lg = lg.ndata["type"] + index_offset
        lg_feats = torch.index_select(self.rel_emb, dim=0, index=index_lg)
        lg.ndata["feat"] = lg_feats

        return lg_feats

    def forward(self, lg):
        for i in range(self.num_lgcn_layers):
            lgcn_layer = self.layers_lgcn[i]

            if i == 0:
                lg_feats = self.get_first_lg_feat(lg)

            _ = lgcn_layer(lg, [])  
        
        lg_feats = lg.ndata["feat"]

        return lg_feats