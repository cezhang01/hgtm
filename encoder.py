import torch
import torch.nn as nn
import numpy as np
import collections
import math


class HypConvLayer(nn.Module):

    def __init__(self, args, dim_in, dim_out):

        super(HypConvLayer, self).__init__()
        self.parse_args(args)
        self.generate_modules(args, dim_in, dim_out)

    def parse_args(self, args):

        self.use_bias = args.use_bias
        self.dropout_keep_prob = 1.0

    def generate_modules(self, args, dim_in, dim_out):

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(1 - self.dropout_keep_prob)
        self.weight = nn.Parameter(torch.Tensor(dim_out, dim_in), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(dim_out), requires_grad=True)
        self.att_layer = nn.ModuleList([nn.Linear(2 * dim_out, 1)])
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.weight, gain=0.1)
        nn.init.constant_(self.bias, 0)

    def message_passing(self, self_emb, neigh_emb, act, c_in, c_out, mode):

        if mode == 'train':
            self.weight = self.dropout(self.weight)

        self_emb = self.hyp_linear(self_emb, c_in)
        neigh_emb = self.hyp_linear(neigh_emb, c_in)

        att = self.hyp_att(self_emb, neigh_emb, c_in)
        self_emb_agg = self.hyp_agg_act(self_emb, neigh_emb, att, act, c_in, c_out)
        # self_emb_agg = self.hyp_agg(self_emb, neigh_emb, att, c_in)
        # self_emb_agg = self.hyp_act(self_emb_agg, act, c_in, c_out)

        return self_emb_agg

    def hyp_linear(self, emb, c_in):

        emb_proj = self.manifold.mobius_matvec(self.weight, emb, c_in)
        emb_proj = self.manifold.proj(emb_proj, c_in)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), c_in)
            hyp_bias = self.manifold.expmap0(bias, c_in)
            hyp_bias = self.manifold.proj(hyp_bias, c_in)
            emb_proj = self.manifold.mobius_add(emb_proj, hyp_bias, c_in)
            emb_proj = self.manifold.proj(emb_proj, c_in)

        return emb_proj

    def hyp_att(self, self_emb, neigh_emb, c_in):

        n = torch.div(neigh_emb.size(0), self_emb.size(0)).int()

        self_emb = torch.reshape(torch.tile(torch.unsqueeze(self_emb, dim=1), [1, n, 1]), neigh_emb.size())
        self_emb_tan = self.manifold.logmap0(self_emb, c_in)
        neigh_emb_tan = self.manifold.logmap0(neigh_emb, c_in)
        emb_concat = torch.concat([self_emb_tan, neigh_emb_tan], dim=1)
        att = torch.reshape(self.att_layer[0](emb_concat), [-1, n])
        att = self.softmax(self.sigmoid(att))

        return att

    def hyp_agg(self, self_emb, neigh_emb, att, c_in):

        # exp and log mapping at the north pole
        self_emb_tan = self.manifold.logmap0(self_emb, c_in)
        neigh_emb_tan = self.manifold.logmap0(neigh_emb, c_in)
        att = torch.unsqueeze(att, dim=1)
        neigh_emb_tan = torch.reshape(neigh_emb_tan, [att.size(0), att.size(-1), -1])
        neigh_emb_agg = torch.squeeze(torch.matmul(att, neigh_emb_tan))
        self_emb_agg = 0.5 * (self_emb_tan + neigh_emb_agg)
        self_emb_agg = self.manifold.proj_tan0(self_emb_agg, c_in)
        self_emb_agg = self.manifold.expmap0(self_emb_agg, c_in)
        self_emb_agg = self.manifold.proj(self_emb_agg, c_in)

        return self_emb_agg

    def hyp_act(self, emb, act, c_in, c_out):

        emb_tan = self.manifold.logmap0(emb, c_in)
        emb_tan = act(emb_tan)
        emb_tan = self.manifold.proj_tan0(emb_tan, c_out)
        emb_proj = self.manifold.expmap0(emb_tan, c_out)
        emb_proj = self.manifold.proj(emb_proj, c_out)

        return emb_proj

    def hyp_agg_act(self, self_emb, neigh_emb, att, act, c_in, c_out):

        # exp and log mapping at the north pole
        self_emb_tan = self.manifold.logmap0(self_emb, c_in)
        neigh_emb_tan = self.manifold.logmap0(neigh_emb, c_in)
        att = torch.unsqueeze(att, dim=1)
        neigh_emb_tan = torch.reshape(neigh_emb_tan, [att.size(0), att.size(-1), -1])
        neigh_emb_agg = torch.squeeze(torch.matmul(att, neigh_emb_tan))
        self_emb_agg = 0.5 * (self_emb_tan + neigh_emb_agg)
        self_emb_agg = act(self_emb_agg)
        self_emb_agg = self.manifold.proj_tan0(self_emb_agg, c_out)
        self_emb_agg = self.manifold.expmap0(self_emb_agg, c_out)
        self_emb_agg = self.manifold.proj(self_emb_agg, c_out)

        return self_emb_agg

    def forward(self, manifold, self_emb, neigh_emb, act, c_in, c_out, mode='train'):

        self.manifold = manifold
        self_emb_agg = self.message_passing(self_emb, neigh_emb, act, c_in, c_out, mode)

        return self_emb_agg


class Encoder(nn.Module):

    def __init__(self, args, manifold):

        super(Encoder, self).__init__()
        self.manifold = manifold
        self.parse_args(args)
        self.generate_modules(args)

    def parse_args(self, args):

        self.device = args.device
        self.emb_dim = args.emb_dim
        self.num_conv_layers = args.num_conv_layers
        self.minibatch_size = args.minibatch_size
        self.init_curvature = args.init_curvature

    def generate_modules(self, args):

        self.conv_dim = [args.ptr_word_emb_dim] + [128] * (self.num_conv_layers - 1) + [self.emb_dim]
        if self.manifold.name == 'Hyperboloid':
            self.conv_dim[0] += 1

        self.conv_layers = nn.ModuleDict()
        for type in ['docs', 'words', 'adj']:
            self.conv_layers[type] = nn.ModuleList()
            for layer_id in range(self.num_conv_layers):
                dim_in, dim_out = self.conv_dim[layer_id], self.conv_dim[layer_id + 1]
                self.conv_layers[type] += nn.ModuleList([HypConvLayer(args, dim_in, dim_out)])

        self.acts = nn.ModuleList([nn.Tanh()] * (self.num_conv_layers - 1)) + nn.ModuleList([nn.Tanh()])

    def get_neighbors(self, doc_ids):

        neighbors = collections.defaultdict(list)
        neighbors['doc_ids'] = [doc_ids]
        neighbors['word_ids'] = [np.reshape(self.data.doc_word_neighbors[doc_ids], [-1])]
        neighbors['adj_ids'] = [doc_ids]
        for layer_id in range(self.num_conv_layers):
            if layer_id % 2 == 0:
                neighbor_word_ids = np.reshape(self.data.doc_word_neighbors[neighbors['doc_ids'][layer_id]], [-1])
                neighbors['doc_ids'].append(neighbor_word_ids)
                neighbor_doc_ids = np.reshape(self.data.word_doc_neighbors[neighbors['word_ids'][layer_id]], [-1])
                neighbors['word_ids'].append(neighbor_doc_ids)
            else:
                neighbor_doc_ids = np.reshape(self.data.word_doc_neighbors[neighbors['doc_ids'][layer_id]], [-1])
                neighbors['doc_ids'].append(neighbor_doc_ids)
                neighbor_word_ids = np.reshape(self.data.doc_word_neighbors[neighbors['word_ids'][layer_id]], [-1])
                neighbors['word_ids'].append(neighbor_word_ids)
            neighbor_adj_ids = np.reshape(self.data.doc_doc_neighbors[neighbors['adj_ids'][layer_id]], [-1])
            neighbors['adj_ids'].append(neighbor_adj_ids)

        return neighbors

    def init_features(self, features):

        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(features)
            features = torch.cat([o[:, 0:1], features], dim=1)
        features = self.manifold.proj_tan0(features, self.init_curvature)
        features = self.manifold.expmap0(features, self.init_curvature)
        features = self.manifold.proj(features, self.init_curvature)

        return features

    def get_features(self, doc_ids):

        neighbors = self.get_neighbors(doc_ids)

        features = collections.defaultdict(list)
        for layer_id in range(self.num_conv_layers + 1):
            if layer_id % 2 == 0:
                feat_docs = self.init_features(torch.FloatTensor(self.data.doc_raw_feat[neighbors['doc_ids'][layer_id]])).to(self.device)
                features['docs'].append(feat_docs)
                feat_words = self.init_features(torch.FloatTensor(self.data.ptr_word_emb[neighbors['word_ids'][layer_id]])).to(self.device)
                features['words'].append(feat_words)
            else:
                feat_docs = self.init_features(torch.FloatTensor(self.data.ptr_word_emb[neighbors['doc_ids'][layer_id]])).to(self.device)
                features['docs'].append(feat_docs)
                feat_words = self.init_features(torch.FloatTensor(self.data.doc_raw_feat[neighbors['word_ids'][layer_id]])).to(self.device)
                features['words'].append(feat_words)
            feat_adj = self.init_features(torch.FloatTensor(self.data.doc_raw_feat[neighbors['adj_ids'][layer_id]])).to(self.device)
            features['adj'].append(feat_adj)

        return features

    def graph_conv_encoder(self, doc_ids, mode):

        features = self.get_features(doc_ids)

        z, emb = collections.defaultdict(list), collections.defaultdict(list)
        emb['docs'], emb['words'], emb['adj'] = features['docs'], features['words'], features['adj']
        for layer_id in range(self.num_conv_layers):
            next_emb = collections.defaultdict(list)
            next_emb['docs'], next_emb['words'], next_emb['adj'] = [], [], []
            for hop in range(self.num_conv_layers - layer_id):
                for type in ['docs', 'words', 'adj']:
                    emb_agg = self.conv_layers[type][layer_id](manifold=self.manifold,
                                                               self_emb=emb[type][hop],
                                                               neigh_emb=emb[type][hop + 1],
                                                               act=self.acts[layer_id],
                                                               c_in=self.init_curvature,
                                                               c_out=self.init_curvature,
                                                               mode=mode)
                    next_emb[type].append(emb_agg)
            emb['docs'], emb['words'], emb['adj'] = next_emb['docs'], next_emb['words'], next_emb['adj']

        # hyperbolic mean pooling
        doc_emb = self.manifold.logmap0(emb['docs'][0], self.init_curvature)
        adj_emb = self.manifold.logmap0(emb['adj'][0], self.init_curvature)
        doc_emb = self.manifold.proj_tan0(0.5 * (doc_emb + adj_emb), self.init_curvature)
        doc_emb = self.manifold.expmap0(doc_emb, self.init_curvature)
        doc_emb = self.manifold.proj(doc_emb, self.init_curvature)
        z['docs'] = doc_emb
        z['words'] = emb['words'][0]

        return z

    def forward(self, doc_ids, data, mode='train'):

        self.data = data
        z = self.graph_conv_encoder(doc_ids, mode)

        return z['docs'], z['words']