import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import collections


class GraphDecoder(nn.Module):

    def __init__(self, args, manifold):

        super(GraphDecoder, self).__init__()
        self.manifold = manifold
        self.device = args.device
        self.num_negative_samples = args.num_negative_samples
        self.r = 2.0
        self.t = 1.0

        self.generate_modules(args)

    def generate_modules(self, args):

        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.functional.binary_cross_entropy

    def fermi_dirac_decoder(self, sqdist):

        prob = 1. / (torch.exp((sqdist - self.r) / self.t) + 1.0)

        return prob

    def decoder_graph(self, emb_1, emb_2, emb_neg, c):

        n = torch.div(emb_2.size(0), emb_1.size(0)).int()

        emb_1 = torch.reshape(torch.tile(torch.unsqueeze(emb_1, dim=1), [1, n, 1]), emb_2.size())
        pos_sqdist = self.manifold.sqdist(emb_1, emb_2, c)
        pos_scores = self.fermi_dirac_decoder(pos_sqdist)

        emb_1 = torch.reshape(torch.tile(torch.unsqueeze(emb_1, dim=1), [1, self.num_negative_samples, 1]), emb_neg.size())
        neg_sqdist = self.manifold.sqdist(emb_1, emb_neg, c)
        neg_scores = self.fermi_dirac_decoder(neg_sqdist)

        pos_loss = self.bce_loss(pos_scores, torch.ones_like(pos_scores), reduction='none')
        neg_loss = self.bce_loss(neg_scores, torch.zeros_like(neg_scores), reduction='none')
        pos_loss = torch.mean(torch.sum(torch.reshape(pos_loss, [-1, n]), dim=-1))
        neg_loss = torch.mean(torch.sum(torch.reshape(torch.sum(torch.reshape(neg_loss, [-1, self.num_negative_samples]), dim=-1), [-1, n]), dim=-1))

        loss = pos_loss + neg_loss

        return loss

    def forward(self, emb_1, emb_2, emb_neg, c):

        loss = self.decoder_graph(emb_1, emb_2, emb_neg, c)

        return loss


class TextDecoder(nn.Module):

    def __init__(self, args):

        super(TextDecoder, self).__init__()
        self.emb_dim = args.emb_dim
        self.word_emb_dim = args.word_emb_dim
        self.device = args.device
        self.generate_modules(args)

    def generate_modules(self, args):

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.bce_loss = nn.functional.binary_cross_entropy

    def decoder_text(self, doc_topic_dist, topic_word_dist, bow_true):

        y_pred = torch.matmul(doc_topic_dist, topic_word_dist)
        y_pred = torch.clamp(y_pred, min=1e-10)
        loss = - torch.sum(torch.multiply(bow_true, torch.log(y_pred)), dim=-1)
        loss = torch.mean(loss)

        return loss, y_pred

    def forward(self, doc_topic_dist, topic_word_dist, bow_true):

        loss, bow_pred = self.decoder_text(doc_topic_dist, topic_word_dist, bow_true)

        return loss, bow_pred


class TopicEmbReg(nn.Module):

    def __init__(self, args):

        super(TopicEmbReg, self).__init__()
        self.device = args.device

    def topic_emb_reg(self, topic_emb, tree):

        def get_tree_mask_reg(all_child_ids):

            tree_mask_reg = np.zeros([len(all_child_ids), len(all_child_ids)], dtype=np.float32)
            for parent_id, child_ids in tree.par2child.items():
                for child_id1 in child_ids:
                    for child_id2 in child_ids:
                        child_idx1 = all_child_ids.index(child_id1)
                        child_idx2 = all_child_ids.index(child_id2)
                        tree_mask_reg[child_idx1, child_idx2] = tree_mask_reg[child_idx2, child_idx1] = 1.0

            return tree_mask_reg

        all_child_ids = np.sort(list(tree.child2par.keys()))
        diff_topic_emb = torch.concat([topic_emb[child_id] - topic_emb[tree.child2par[child_id]] for child_id in all_child_ids], dim=0)
        diff_topic_emb_norm = diff_topic_emb / torch.norm(diff_topic_emb, dim=1, keepdim=True)
        # diff_topic_emb_norm = nn.functional.normalize(diff_topic_emb, dim=1)
        topic_dots = torch.clamp(torch.matmul(diff_topic_emb_norm, diff_topic_emb_norm.T), min=-1.0, max=1.0)

        tree_mask_reg = get_tree_mask_reg(all_child_ids.tolist())
        tree_mask_reg = torch.FloatTensor(tree_mask_reg).to(self.device)
        topic_emb_reg = torch.square(topic_dots - torch.eye(len(all_child_ids), dtype=torch.float32).to(self.device)) * tree_mask_reg
        topic_emb_reg = torch.sum(topic_emb_reg) / torch.sum(tree_mask_reg)

        return topic_emb_reg

    def forward(self, topic_emb, tree):

        loss = self.topic_emb_reg(topic_emb, tree)

        return loss


class HypTopicEmbReg(nn.Module):

    def __init__(self, args, manifold):

        super(HypTopicEmbReg, self).__init__()
        self.manifold = manifold
        self.device = args.device
        self.init_curvature = args.init_curvature

        self.generate_modules(args)

    def generate_modules(self, args):

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.bce_loss = nn.functional.binary_cross_entropy

    def distance_matrix(self, G, topics_weight, topic_ids):

        graph_pairs = {topic_id: {} for topic_id in topic_ids}

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), 1):
                if len(lst[i:i + n]) == n:
                    yield lst[i:i + n]

        for topic_id in topic_ids:
            shortest_path_source = nx.shortest_path(G, source=topic_id)
            for k, nodes in shortest_path_source.items():
                d = 0
                for pair in chunks(nodes, 2):
                    d = d + topics_weight[min(pair), max(pair)]
                graph_pairs[topic_id][k] = d

        return graph_pairs

    def topic_emb_reg(self, topic_emb_hyp, tree):

        G = nx.Graph()
        G.add_nodes_from(tree.topic_ids)
        topic_pairs = [[parent_id, child_id] for parent_id, child_ids in tree.par2child.items() for child_id in child_ids]
        G.add_edges_from(topic_pairs)

        topics_weight = {}
        for parent_id, child_ids in tree.par2child.items():
            for child_id in child_ids:
                topics_weight[parent_id, child_id] = 1.0

        topic_pair_dist_dict = self.distance_matrix(G, topics_weight, tree.topic_ids)

        topic_pair_dist = []
        for i in tree.topic_ids:
            stack_col = []
            for j in tree.topic_ids:
                dist = topic_pair_dist_dict[i][j]
                stack_col.append(dist)
            topic_pair_dist.append(stack_col)
        topic_pair_dist = torch.FloatTensor(topic_pair_dist).to(self.device)

        num_topics = len(tree.topic_ids)
        topic_emb_hyp = torch.concat([topic_emb_hyp[topic_id] for topic_id in tree.topic_ids], dim=0)
        topic_embeds_hyp_repeat1 = torch.tile(topic_emb_hyp, [num_topics, 1])
        topic_embeds_hyp_repeat2 = torch.reshape(torch.tile(torch.unsqueeze(topic_emb_hyp, dim=1), [1, num_topics, 1]), [num_topics * num_topics, -1])
        sqdist = self.manifold.sqdist(topic_embeds_hyp_repeat1, topic_embeds_hyp_repeat2, c=self.init_curvature)
        sqdist = torch.reshape(sqdist, [num_topics, num_topics])

        diag = torch.eye(num_topics).to(self.device)
        reg_loss = 0.5 * (sqdist - topic_pair_dist) ** 2
        reg_loss = torch.triu(reg_loss, diagonal=-1)
        reg_loss = torch.sum(reg_loss * (1 - diag))

        return reg_loss

    def forward(self, topic_emb, tree):

        loss = self.topic_emb_reg(topic_emb, tree)

        return loss