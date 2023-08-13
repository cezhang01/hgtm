import torch
import torch.nn as nn
import numpy as np
import collections


class DoublyRNNCell(nn.Module):

    def __init__(self, emb_dim, manifold, device, name):

        super(DoublyRNNCell, self).__init__()
        self.emb_dim = emb_dim
        self.manifold = manifold
        self.device = device
        self.name = name

        self.generate_modules()

    def generate_modules(self):

        self.drnn_layers = nn.ModuleDict()
        for layer_name in ['ancestral', 'fraternal', 'hidden', 'output']:
            self.drnn_layers[layer_name] = nn.Linear(self.emb_dim, self.emb_dim).to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.identity = nn.Identity()
        self.init_state = nn.ModuleList([nn.Linear(self.emb_dim, 1, bias=False)]).to(self.device)

    def get_init_state(self):

        return self.init_state[0].weight

    def get_zero_state(self):

        zero_state = torch.zeros(1, self.emb_dim, dtype=torch.float32).to(self.device)

        return zero_state

    def forward(self, state_ancestral, state_fraternal):

        state_ancestral = self.tanh(self.drnn_layers['ancestral'](state_ancestral))
        state_fraternal = self.tanh(self.drnn_layers['fraternal'](state_fraternal))
        state_hidden = self.drnn_layers['hidden'](state_ancestral + state_fraternal)
        output = self.drnn_layers['output'](self.tanh(state_hidden))
        if self.name == 'evaluate_doc_topic_dist':
            if self.manifold.name == 'Hyperboloid':
                o = torch.zeros_like(output)
                output = torch.cat([o[:, 0:1], output], dim=1)
            output = self.manifold.proj_tan0(output, c=1.0)
            output = self.manifold.expmap0(output, c=1.0)
            output = self.manifold.proj(output, c=1.0)

        return output, state_hidden


class DoublyRNN(nn.Module):

    def __init__(self, emb_dim, manifold, device, name):

        super(DoublyRNN, self).__init__()
        self.emb_dim = emb_dim
        self.manifold = manifold
        self.device = device
        self.name = name

        self.generate_modules()

    def generate_modules(self):

        self.drnn_cell = DoublyRNNCell(self.emb_dim, self.manifold, self.device, self.name)

    def forward(self, par2child):

        outputs, states_parent = collections.defaultdict(float), collections.defaultdict(float)

        init_state_parent = self.drnn_cell.get_init_state()
        init_state_sibling = self.drnn_cell.get_zero_state()
        output, state_sibling = self.drnn_cell(init_state_parent, init_state_sibling)
        outputs[0], states_parent[0] = output, state_sibling

        for parent_id, child_ids in par2child.items():
            state_parent = states_parent[parent_id]
            state_sibling = init_state_sibling
            for child_id in child_ids:
                output, state_sibling = self.drnn_cell(state_parent, state_sibling)
                outputs[child_id], states_parent[child_id] = output, state_sibling

        return outputs