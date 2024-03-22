import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.functional import edge_softmax


class RGCNLayer(nn.Module):
    def __init__(
        self,
        in_feat,
        out_feat,
        bias=None,
        activation=None,
        self_loop=False,
        skip_connect=False,
        dropout=0.0,
        layer_norm=False,
    ):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.layer_norm = layer_norm

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain("relu"))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )
            # self.loop_weight = nn.Parameter(torch.eye(out_feat), requires_grad=False)

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(
                torch.Tensor(out_feat, out_feat)
            )  # 和self-loop不一样，是跨层的计算
            nn.init.xavier_uniform_(
                self.skip_connect_weight, gain=nn.init.calculate_gain("relu")
            )

            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if self.layer_norm:
            self.normalization_layer = nn.LayerNorm(out_feat, elementwise_affine=False)

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g, prev_h=[]):
        if self.self_loop:
            # print(self.loop_weight)
            loop_message = torch.mm(g.ndata["h"], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)
        # self.skip_connect_weight.register_hook(lambda g: print("grad of skip connect weight: {}".format(g)))
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(
                torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias
            )  # 使用sigmoid，让值在0~1
            # print("skip_ weight")
            # print(skip_weight)
            # print("skip connect weight")
            # print(self.skip_connect_weight)
            # print(torch.mm(prev_h, self.skip_connect_weight))

        self.propagate(g)  # 这里是在计算从周围节点传来的信息

        # apply bias and activation
        node_repr = g.ndata["h"]
        if self.bias:
            node_repr = node_repr + self.bias
        # print(len(prev_h))
        if (
            len(prev_h) != 0 and self.skip_connect
        ):  # 两次计算loop_message的方式不一样，前者激活后再加权
            previous_node_repr = (1 - skip_weight) * prev_h
            if self.activation:
                node_repr = self.activation(node_repr)
            if self.self_loop:
                if self.activation:
                    loop_message = skip_weight * self.activation(loop_message)
                else:
                    loop_message = skip_weight * loop_message
                node_repr = node_repr + loop_message
            node_repr = node_repr + previous_node_repr
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.layer_norm:
                node_repr = self.normalization_layer(node_repr)
            if self.activation:
                node_repr = self.activation(node_repr)
            # print("node_repr")
            # print(node_repr)
        g.ndata["h"] = node_repr
        return node_repr


class RGCNBasisLayer(RGCNLayer):
    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        num_bases=-1,
        bias=None,
        activation=None,
        is_input_layer=False,
    ):
        super(RGCNBasisLayer, self).__init__(in_feat, out_feat, bias, activation)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add basis weights
        self.weight = nn.Parameter(
            torch.Tensor(self.num_bases, self.in_feat, self.out_feat)
        )
        if self.num_bases < self.num_rels:
            # linear combination coefficients
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain("relu"))

    def propagate(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases, self.in_feat * self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat
            )
        else:
            weight = self.weight

        if self.is_input_layer:

            def msg_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data["type"] * self.in_feat + edges.src["id"]
                return {"msg": embed.index_select(0, index)}

        else:

            def msg_func(edges):
                w = weight.index_select(0, edges.data["type"])
                msg = torch.bmm(edges.src["h"].unsqueeze(1), w).squeeze()
                return {"msg": msg}

        def apply_func(nodes):
            return {"h": nodes.data["h"] * nodes.data["norm"]}

        g.update_all(msg_func, fn.sum(msg="msg", out="h"), apply_func)


class RGCNBlockLayer(RGCNLayer):
    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        num_bases,
        bias=None,
        activation=None,
        self_loop=False,
        dropout=0.0,
        skip_connect=False,
        layer_norm=False,
    ):
        super(RGCNBlockLayer, self).__init__(
            in_feat,
            out_feat,
            bias,
            activation,
            self_loop=self_loop,
            skip_connect=skip_connect,
            dropout=dropout,
        )
        self.num_rels = num_rels
        self.num_bases = num_bases

        assert self.num_bases > 0

        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        # assuming in_feat and out_feat are both divisible by num_bases
        self.weight = nn.Parameter(
            torch.Tensor(
                self.num_rels, self.num_bases * self.submat_in * self.submat_out
            )
        )
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))

    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data["type"]).view(
            -1, self.submat_in, self.submat_out
        )  # [edge_num, submat_in, submat_out]
        node = edges.src["h"].view(
            -1, 1, self.submat_in
        )  # [edge_num * num_bases, 1, submat_in]->
        msg = torch.bmm(node, weight).view(-1, self.out_feat)  # [edge_num, out_feat]
        return {"msg": msg}

    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg="msg", out="h"), self.apply_func)
        # g.updata_all ({'msg': msg} , fn.sum(msg='msg', out='h'), {'h': nodes.data['h'] * nodes.data[''norm]})

    def apply_func(self, nodes):
        return {"h": nodes.data["h"] * nodes.data["norm"]}


class UnionRGCNLayer(nn.Module):
    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        num_bases=-1,
        bias=None,
        activation=None,
        self_loop=False,
        dropout=0.0,
        skip_connect=False,
        rel_emb=None,
    ):
        super(UnionRGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect
        self.ob = None
        self.sub = None

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(
            self.weight_neighbor, gain=nn.init.calculate_gain("relu")
        )

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(
                self.evolve_loop_weight, gain=nn.init.calculate_gain("relu")
            )

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(
                torch.Tensor(out_feat, out_feat)
            )  # 和self-loop不一样，是跨层的计算
            nn.init.xavier_uniform_(
                self.skip_connect_weight, gain=nn.init.calculate_gain("relu")
            )
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def propagate(self, g):
        g.update_all(
            lambda x: self.msg_func(x), fn.sum(msg="msg", out="h"), self.apply_func
        )

    def forward(self, g, prev_h, emb_rel):
        self.rel_emb = emb_rel
        # self.sub = sub
        # self.ob = ob
        if self.self_loop:
            # loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            # masked_index = torch.masked_select(torch.arange(0, g.number_of_nodes(), dtype=torch.long), (g.in_degrees(range(g.number_of_nodes())) > 0))
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),
                (g.in_degrees(range(g.number_of_nodes())) > 0),
            )
            loop_message = torch.mm(g.ndata["h"], self.evolve_loop_weight)
            loop_message[masked_index, :] = torch.mm(g.ndata["h"], self.loop_weight)[
                masked_index, :
            ]
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(
                torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias
            )  # 使用sigmoid，让值在0~1

        # calculate the neighbor message with weight_neighbor
        self.propagate(g)
        node_repr = g.ndata["h"]

        # print(len(prev_h))
        if (
            len(prev_h) != 0 and self.skip_connect
        ):  # 两次计算loop_message的方式不一样，前者激活后再加权
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
        g.ndata["h"] = node_repr
        return node_repr

    def msg_func(self, edges):
        # if reverse:
        #     relation = self.rel_emb.index_select(0, edges.data['type_o']).view(-1, self.out_feat)
        # else:
        #     relation = self.rel_emb.index_select(0, edges.data['type_s']).view(-1, self.out_feat)
        relation = self.rel_emb.index_select(0, edges.data["etype"]).view(
            -1, self.out_feat
        )
        edge_type = edges.data["etype"]
        edge_num = edge_type.shape[0]
        node = edges.src["h"].view(-1, self.out_feat)
        # node = torch.cat([torch.matmul(node[:edge_num // 2, :], self.sub),
        #                  torch.matmul(node[edge_num // 2:, :], self.ob)])
        # node = torch.matmul(node, self.sub)

        # after add inverse edges, we only use message pass when h as tail entity
        # 这里计算的是每个节点发出的消息，节点发出消息时其作为头实体
        # msg = torch.cat((node, relation), dim=1)
        msg = node + relation
        # calculate the neighbor message with weight_neighbor
        msg = torch.mm(msg, self.weight_neighbor)
        return {"msg": msg}

    def apply_func(self, nodes):
        return {"h": nodes.data["h"] * nodes.data["norm"]}


class AURGCN(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.code_embedding = nn.Embedding(args.n_codes, args.emb_dim)
        self.relation_embedding = nn.Embedding(args.n_rels, args.emb_dim)
        nn.init.uniform_(self.code_embedding.weight, -1.0, 1.0)
        nn.init.uniform_(self.relation_embedding.weight, -1.0, 1.0)

        self.layers = nn.ModuleList()
        n_rels = args.n_rels
        in_feats = args.emb_dim
        n_hidden = args.emb_dim
        activation = F.relu
        Layer = AURGCNLayer
        for i in range(args.n_layers):
            if i == 0:
                self.layers.append(
                    Layer(
                        in_feats,
                        n_hidden,
                        num_rels=n_rels,
                        activation=activation,
                        self_loop=True,
                        dropout=args.dropout,
                    )
                )
            else:
                self.layers.append(
                    Layer(
                        n_hidden,
                        n_hidden,
                        num_rels=n_rels,
                        activation=activation,
                        self_loop=True,
                        dropout=args.dropout,
                    )
                )

    def forward(self, g, state):

        features = self.code_embedding(g.ndata["id"])
        g.ndata["h"] = features
        rel_emb = self.relation_embedding.weight

        for layer in self.layers:
            layer(g, rel_emb, state)

        emb = g.ndata.pop("h")

        g.ndata["emb"] = emb
        x = dgl.readout_nodes(g, "emb", weight="readout_weight", op="sum")
        # l, b = mask.shape
        # x = x.reshape(b, l, -1).swapaxes(0, 1)
        return x


class AURGCNLayer(nn.Module):
    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        bias=None,
        activation=None,
        self_loop=False,
        dropout=0.0,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        num_heads=4,
    ):
        super(AURGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(
            self.weight_neighbor, gain=nn.init.calculate_gain("relu")
        )

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(
                self.evolve_loop_weight, gain=nn.init.calculate_gain("relu")
            )

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.fc = nn.Linear(self.in_feat, out_feat * num_heads, bias=False)
        self.num_heads = num_heads
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feat)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feat)))
        self.attn_s = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feat)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_s, gain=gain)

    def forward(self, g, rel_emb, state):

        h = self.feat_drop(g.ndata["h"])
        feat = self.fc(h).view(-1, self.num_heads, self.out_feat)
        g.ndata["ft"] = feat
        g.edata["a"] = self.cal_attention(g, feat, state)

        self.rel_emb = rel_emb

        if self.self_loop:
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),
                (g.in_degrees(range(g.number_of_nodes())) > 0),
            )
            loop_message = torch.mm(g.ndata["h"], self.evolve_loop_weight)
            loop_message[masked_index, :] = torch.mm(g.ndata["h"], self.loop_weight)[
                masked_index, :
            ]

        g.update_all(lambda x: self.msg_func(x), fn.sum(msg="msg", out="h"))
        node_repr = g.ndata["h"]

        if self.self_loop:
            node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata["h"] = node_repr
        return node_repr

    def cal_attention(self, graph, feat, state):
        with graph.local_scope():

            # linear transformation
            state = self.fc(state).view(-1, self.num_heads, self.out_feat)

            # FNN
            el = (feat * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
            es = (state * self.attn_s).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat, "el": el, "es": es})
            graph.dstdata.update({"er": er})

            # compute edge attention
            graph.apply_edges(fn.u_add_v("el", "er", "elr"))  # elr = el + er
            graph.apply_edges(fn.u_add_e("es", "elr", "e"))  # e = el + er + es
            e = self.leaky_relu(graph.edata.pop("e"))

            # compute softmax
            # graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            return self.attn_drop(edge_softmax(graph, e))

    def msg_func(self, edges):
        relation = self.rel_emb.index_select(0, edges.data["rel_type"]).view(
            -1, self.out_feat
        )
        feat = edges.src["ft"]  # (E, num_heads, d)
        node = (edges.data["a"] * feat).sum(dim=1)
        # msg = node + relation
        msg = node + torch.mm(relation, self.weight_neighbor)
        return {"msg": msg}
