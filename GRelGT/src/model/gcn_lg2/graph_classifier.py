import math
from dgl import mean_nodes
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from .gcn_model import UniGCN
from .gcn_model import LGCN

class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True,
                          bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size),
                                1.0 / math.sqrt(self.hidden_size))

    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_node_len = max(a_scope)
        # padding
        message_lst = []
        hidden_lst = []
        a_start = 0
        for i in a_scope:
            i = int(i)
            if i == 0:
                assert 0
            cur_message = message.narrow(0, a_start, i)
            cur_hidden = hidden.narrow(0, a_start, i)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            a_start += i
            cur_message = torch.nn.ZeroPad2d((0, 0, 0, MAX_node_len - cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))

        message_lst = torch.cat(message_lst, 0)
        hidden_lst = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2, 1, 1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)

        # unpadding
        cur_message_unpadding = []
        kk = 0
        for a_size in a_scope:
            a_size = int(a_size)
            cur_message_unpadding.append(cur_message[kk, :a_size].view(-1, 2 * self.hidden_size))
            kk += 1
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)

        return cur_message_unpadding


class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id
        self.max_label_value = params.max_label_value
        # self.relation_list = list(self.relation2id.values())
        self.no_jk = self.params.no_jk
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        self.device = params.device

        self.lgnn = LGCN(params)
        self.gnn = UniGCN(params)

        num_final_gcn = self.params.num_gcn_layers
        if self.no_jk:  
            num_final_gcn = 1

        self.line_ent = nn.Linear(num_final_gcn * self.params.emb_dim, 200)
        self.line_rel = nn.Linear(num_final_gcn * self.params.emb_dim, 200)

        self.bn = torch.nn.BatchNorm1d(self.params.emb_dim)
        self.linear1 = nn.Linear(self.params.emb_dim, self.params.emb_dim)
        self.linear2 = nn.Linear(self.params.emb_dim, 1)

        self.gru = BatchGRU(params.emb_dim)
        self.linear3 = nn.Linear(params.emb_dim * 2, params.emb_dim)
        self.dropout = nn.Dropout(0.5)


    def DistMult(self, head, relation, tail):
        head = self.line_ent(head)
        tail = self.line_ent(tail)
        relation = self.line_rel(relation)
        s = head * relation
        ans = s * tail
        pred = ans.sum(dim=1,keepdim=True)
        return pred

    def ComplEx(self, head, tail,target_rel_emd, path_agg_emd):
        head = self.line_ent(head)
        tail = self.line_ent(tail)
        relation = (target_rel_emd+path_agg_emd) / 2.0
        relation = self.line_rel(relation)

        re_head, im_head = torch.chunk(head, 2, dim=1)
        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = re_score * re_tail + im_score * im_tail

        pred = score.sum(dim=1, keepdim=True)
        return pred


    def LinearScore(self, head, tail,target_rel_emd, path_agg_emd):
        relation = target_rel_emd
        features = head + relation - tail
        scores = self.linear1(features)
        scores = self.linear2(scores)

        return scores 


    def forward(self, data, lg, b_idx):
        g, rel_labels = data
        local_g = g.local_var()
        in_deg = local_g.in_degrees(range(local_g.number_of_nodes())).float().numpy()
        in_deg[in_deg == 0] = 1  
        node_norm = 1.0 / in_deg
        local_g.ndata['norm'] = node_norm
        local_g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
        norm = local_g.edata['norm']  

        if self.params.gpu >= 0:
            norm = norm.cuda(device=self.params.gpu)

        lg_type = lg.ndata["type"]
        lg_feats = self.lgnn(lg)
        x_input = torch.zeros(self.num_rels+1, self.params.emb_dim).float().cuda()        
        for r in range(0, self.num_rels):
            total_rel = torch.zeros(1, self.params.emb_dim).float().cuda()
            idx = torch.nonzero(lg_type == r)
            if len(idx)==0:
                continue
            else:
                for re in (idx):
                    total_rel = total_rel + lg_feats[re]
                x_input[r+1,:] = total_rel/len(idx)

        target_rel, path_agg = self.gnn(g, norm, x_input, rel_labels)


        emds = g.ndata['feat']
        batch_num_sizes = g.batch_num_nodes
        a_message = self.gru(emds, batch_num_sizes)
        a_message = torch.relu(self.linear3(a_message))
        a_message = self.dropout(a_message)



        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)  
        head_embs = a_message[head_ids]  
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = a_message[tail_ids]


        output = self.LinearScore(head_embs, tail_embs, target_rel, path_agg)

        return output

