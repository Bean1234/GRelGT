"""
File baseed off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x


class RGCNLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, aggregator, bias=None, activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False, no_jk=False):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.no_jk = no_jk

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))

        self.aggregator = aggregator

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if edge_dropout:
            self.edge_dropout = nn.Dropout(edge_dropout)
        else:
            self.edge_dropout = Identity()

        self.line = nn.Linear(out_dim, out_dim)
        self.bn = torch.nn.BatchNorm1d(out_dim)
        # self.bn = torch.nn.LayerNorm(out_dim)


    # define how propagation is done in subclass
    def propagate(self, g, norm, edge_rel_emd, target_rel_emd_new):
        raise NotImplementedError

    def forward(self, g, norm, edge_rel_emd, target_rel_emd_new):

        self.propagate(g, norm, edge_rel_emd, target_rel_emd_new)

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias

        node_repr_ = self.line(node_repr)

        node_repr_ = self.activation(node_repr_)
        # node_repr_ = self.bn(node_repr_)

        if self.dropout:
            node_repr_ = self.dropout(node_repr_)

        g.ndata['h'] = node_repr_

        g.ndata['feat'] = g.ndata['h']

class RGCNBasisLayer(RGCNLayer):
    def __init__(self, inp_dim, out_dim, aggregator, attn_rel_emb_dim, num_rels, num_bases=-1, bias=None, device=0,
                 activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False, has_attn=False, no_jk=False):
        super(
            RGCNBasisLayer,
            self).__init__(
            inp_dim,
            out_dim,
            aggregator,
            bias,
            activation,
            dropout=dropout,
            edge_dropout=edge_dropout,
            is_input_layer=is_input_layer,
            no_jk=no_jk)
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.attn_rel_emb_dim = attn_rel_emb_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        self.has_attn = has_attn
        self.device = device

        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        self.weight = nn.Linear(inp_dim, out_dim)
        self.weight2 = nn.Linear(3 * out_dim, out_dim)

        self.lfeat_weight = nn.Linear(inp_dim, out_dim)
        self.lfeat_activation = nn.LeakyReLU()

        self.w1 = nn.Linear(2*inp_dim, out_dim)
        self.w2 = nn.Linear(2*inp_dim, out_dim)
        self.w3 = nn.Linear(2*inp_dim, out_dim)


        self.self_loop_weight = nn.Parameter(torch.Tensor(self.inp_dim, self.out_dim))

        if self.has_attn:
            self.A1 = nn.Linear(2 * self.inp_dim + self.attn_rel_emb_dim, inp_dim)  # att1
            self.B1 = nn.Linear(inp_dim, 1)
            self.A2 = nn.Linear(self.attn_rel_emb_dim, inp_dim)  # att2
            self.B2 = nn.Linear(inp_dim, 1)

        
        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))

    def propagate(self, g, norm, edge_rel_emd, target_rel_emd_new):
        # generate all weights from bases
        if norm is not None:
            g.edata['norm'] = norm

        g.edata['w'] = self.edge_dropout(torch.ones(g.number_of_edges(), 1).to(self.device))  
        input_ = 'feat'

        def msg_func(edges):
            w1 = self.weight(edges.src[input_])
            w1 = self.weight2(torch.cat([edge_rel_emd + w1, edge_rel_emd - w1, edge_rel_emd * w1], dim=1))
            msg = edges.data['w'] * w1  # [E, O]
            curr_emb = torch.mm(edges.dst[input_], self.self_loop_weight)  # (E, O)

            if self.has_attn:
                e1 = torch.cat([edges.src[input_], edges.dst[input_], edge_rel_emd], dim=1)
                a1 = torch.sigmoid(self.B1(torch.relu(self.A1(e1))))
                e2 = edge_rel_emd - target_rel_emd_new
                a2 = torch.sigmoid(self.B2(torch.relu(self.A2(e2))))
                a = a1 + 1.3*a2  # a1 * a2
            else:
                a = torch.ones((len(edges), 1)).to(self.device)
            return {'curr_emb': curr_emb, 'msg': msg, 'alpha': a}

        g.update_all(msg_func, self.aggregator, None)

class LGCNLayer(nn.Module): 
    def __init__(self, out_dim):
        super(LGCNLayer, self).__init__()
        self.out_dim = out_dim
        self.self_loop = True
        self.skip_connect = False
        self.activation = nn.Tanh()
        self.dropout = 0.2

        self.weight_relation = nn.Parameter(torch.Tensor(self.out_dim, self.out_dim))
        self.inv_weight_relation = nn.Parameter(torch.Tensor(self.out_dim, self.out_dim))
        self.in_weight_relation = nn.Parameter(torch.Tensor(self.out_dim, self.out_dim))
        self.out_weight_relation = nn.Parameter(torch.Tensor(self.out_dim, self.out_dim))

        self.LineAtt = nn.Linear(2*self.out_dim, 1)

        nn.init.xavier_uniform_(self.weight_relation, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.inv_weight_relation, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.in_weight_relation, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.out_weight_relation, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(self.out_dim, self.out_dim))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(self.out_dim, self.out_dim))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(self.out_dim, self.out_dim))   
            nn.init.xavier_uniform_(self.skip_connect_weight,gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(self.out_dim))
            nn.init.zeros_(self.skip_connect_bias)  
        if self.dropout:
            self.dropout = nn.Dropout(self.dropout)
        else:
            self.dropout = None

    def msg_func_e(self, edges):
        edge_type = edges.data["etype"]
        normal_index = torch.where(edge_type==0)[0]
        inv_index = torch.where(edge_type==1)[0]
        out_index = torch.where(edge_type==2)[0]
        in_index = torch.where(edge_type==3)[0]

        atten = torch.sigmoid(F.relu(self.LineAtt(torch.cat([edges.src['feat'], edges.dst['feat']], dim=1))).squeeze())

        
        edges = edges.src['feat'].view(-1, self.out_dim)

        normal_edge = torch.mm(edges[normal_index], self.weight_relation)
        inv_edge = torch.mm(edges[inv_index], self.weight_relation)
        out_edge = torch.mm(edges[out_index], self.weight_relation)
        in_edge = torch.mm(edges[in_index], self.weight_relation)

        msg = torch.cat([normal_edge, inv_edge, out_edge, in_edge], dim = 0)
        return {'msg': msg,"atten": atten}

    def reduce_func(self, nodes):
        msg = nodes.mailbox['msg']
        atten = nodes.mailbox['atten'].unsqueeze(1)
        f = torch.matmul(atten, msg).squeeze(1)
        return {'feat': f}

    def apply_func_e(self, nodes):
        return {'feat': nodes.data['feat'] * nodes.data['norm'].unsqueeze(1)}

    def forward(self, g, prev_h):
        if self.self_loop:
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),
                (g.in_degrees(range(g.number_of_nodes())).cuda() > 0))
            loop_message = torch.mm(g.ndata['feat'], self.evolve_loop_weight)
            loop_message[masked_index, :] = torch.mm(g.ndata['feat'], self.loop_weight)[masked_index, :]
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = torch.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     
        g.update_all(self.msg_func_e, self.reduce_func, self.apply_func_e)

        node_repr = g.ndata['feat']
        if len(prev_h) != 0 and self.skip_connect:  
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['feat'] = node_repr
        return node_repr

        


