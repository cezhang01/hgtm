import torch
import torch.nn as nn
import numpy as np
import collections
import copy


class Tree(nn.Module):

    def __init__(self, args, manifold):

        super(Tree, self).__init__()
        self.manifold = manifold
        self.parse_args(args)
        self.generate_modules(args)
        self.init_tree()

    def parse_args(self, args):

        self.device = args.device
        self.max_levels = args.max_levels
        self.max_children_per_parent = args.max_children_per_parent
        self.add_threshold = args.add_threshold
        self.remove_threshold = args.remove_threshold
        self.emb_dim = args.emb_dim
        self.init_num_children_per_parent = [3, 3]
        self.init_num_levels = len(self.init_num_children_per_parent) + 1

    def generate_modules(self, args):

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def init_tree(self):

        self.par2child, self.level2topic = collections.defaultdict(list), collections.defaultdict(list)
        self.level2topic[0] = [0]
        for level in range(self.init_num_levels - 1):
            for parent_id in self.level2topic[level]:
                child_ids = [parent_id * self.max_children_per_parent + i for i in range(1, self.init_num_children_per_parent[level] + 1)]
                self.par2child[parent_id] = child_ids
                self.level2topic[level + 1].extend(child_ids)

        self.construct_tree(self.par2child)

    def construct_tree(self, par2child):

        def get_topic2level(par2child, parent_id=0, topic2level=None, level=0):

            if topic2level is None:
                topic2level = {0: level}

            child_ids = par2child[parent_id]
            level += 1
            for child_id in child_ids:
                topic2level[child_id] = level
                if child_id in par2child:
                    get_topic2level(par2child, child_id, topic2level, level)

            return topic2level

        def get_topic2descendant(parent_id, descendant_ids=None):

            if descendant_ids is None:
                descendant_ids = [parent_id]

            if parent_id in self.par2child:
                child_ids = self.par2child[parent_id]
                descendant_ids += child_ids
                for child_id in child_ids:
                    if child_id in self.par2child:
                        descendant_ids = get_topic2descendant(child_id, descendant_ids)

            return descendant_ids

        self.topic_ids = [0] + [child_id for parent_id, child_ids in par2child.items() for child_id in child_ids]
        self.topic_ids.sort()
        self.child2par = {child_id: parent_id for parent_id, child_ids in par2child.items() for child_id in child_ids}
        self.topic2level = get_topic2level(par2child)

        self.level2topic = collections.defaultdict(list)
        for topic_id, level in self.topic2level.items():
            self.level2topic[level].append(topic_id)
        self.num_levels = len([level for level in self.level2topic.keys()])

        self.leaf2ancestor = collections.defaultdict(list)
        for topic_id in self.topic_ids:
            if topic_id not in par2child:
                x = topic_id
                self.leaf2ancestor[topic_id].append(topic_id)
                while x != 0:
                    x = self.child2par[x]
                    self.leaf2ancestor[topic_id].append(x)
                self.leaf2ancestor[topic_id].sort()

        self.topic2descendant = {topic_id: get_topic2descendant(topic_id) for topic_id in self.topic_ids}

    def evaluate_level_dist(self, doc_emb, topic_emb, c=1.0, dist_type='gauss'):

        eta_topic, sum_eta = {}, 0

        d_root = torch.reshape(self.manifold.sqdist(doc_emb, topic_emb[0], c), [1, -1])

        if dist_type == 'gauss':
            eta_topic[0] = torch.exp(- 0.5 * d_root)
        if dist_type == 'inv':
            eta_topic[0] = 1.0 / (1.0 + d_root)

        sum_eta += eta_topic[0]

        levels = np.sort([level for level in self.level2topic.keys()])
        for level in levels:
            topic_ids = self.level2topic[level]
            if level != 0:
                distance = {}
                for topic_id in topic_ids:
                    d = torch.reshape(self.manifold.sqdist(doc_emb, topic_emb[topic_id], c), [1, -1])
                    distance[topic_id] = d
                min_level_distance = torch.min(torch.concat([value for value in distance.values()], dim=0), dim=0, keepdim=True)[0]
                if dist_type == 'gauss':
                    eta_topic[level] = torch.exp(- 0.5 * min_level_distance)
                if dist_type == 'inv':
                    eta_topic[level] = 1.0 / (1.0 + min_level_distance)
                sum_eta += eta_topic[level]

        level_dist = []
        for level in levels:
            level_dist.append(eta_topic[level] / (sum_eta + 1e-20))
        level_dist = torch.concat(level_dist, dim=0).T

        return level_dist

    def evaluate_path_dist(self, doc_emb, topic_emb, c=1.0, dist_type='gauss'):

        path_dist, gamma_topic = {}, {}
        gamma_topic[0] = torch.ones([1, doc_emb.size(0)], dtype=torch.float32).to(self.device)

        distance_c_d = collections.defaultdict(float)
        for parent_id, child_ids in self.par2child.items():
            sum_childs = collections.defaultdict(float)
            sum_childs[parent_id] = 0
            for child_id in child_ids:
                topic_emb_one_topic = topic_emb[child_id]
                d = torch.reshape(self.manifold.sqdist(doc_emb, topic_emb_one_topic, c), [1, -1])
                if dist_type == 'gauss':
                    distance_temp = torch.exp(- 0.5 * d)
                elif dist_type == 'inv':
                    distance_temp = 1.0 / (1.0 + d)
                distance_c_d[child_id] = distance_temp
                sum_childs[parent_id] += distance_temp
            for child_id in child_ids:
                gamma_topic[child_id] = distance_c_d[child_id] / (sum_childs[parent_id] + 1e-20)

        for leaf_id, ancestor_ids in self.leaf2ancestor.items():
            path_dist[leaf_id] = torch.prod(torch.concat([gamma_topic[ancestor_id] for ancestor_id in ancestor_ids], dim=0), dim=0)

        return path_dist

    def evaluate_doc_topic_dist(self, doc_emb, topic_emb, c=1.0, dist_type='inv'):

        level_dist = self.evaluate_level_dist(doc_emb, topic_emb, c=c, dist_type=dist_type)
        path_dist = self.evaluate_path_dist(doc_emb, topic_emb, c=c, dist_type=dist_type)

        doc_topic_dist = collections.defaultdict(float)
        for leaf_id, ancestor_ids in self.leaf2ancestor.items():
            p = path_dist[leaf_id]
            for i, ancestor_id in enumerate(ancestor_ids):
                ancestor_prob = p * level_dist[:, i]
                doc_topic_dist[ancestor_id] += torch.reshape(ancestor_prob, [-1, 1])
        doc_topic_dist = torch.concat([doc_topic_dist[topic_id] for topic_id in self.topic_ids], dim=-1)

        return doc_topic_dist

    def evaluate_topic_word_dist(self, drnn, word_emb, depth_temperature=10.0):

        topic_emb = drnn(self.par2child)

        topic_word_dist = {}
        for topic_id, level in self.topic2level.items():
            topic_e = topic_emb[topic_id]
            temperature = depth_temperature ** (1.0 / (level + 1.0))
            logits = torch.matmul(topic_e, torch.transpose(word_emb, 0, 1))
            topic_word_dist[topic_id] = self.softmax(logits / temperature)
        topic_word_dist = torch.concat([topic_word_dist[topic_id] for topic_id in self.topic_ids], dim=0)

        return topic_word_dist, topic_emb

    def update_tree(self, doc_topic_dist, doc_length):

        self.update_tree_flg = False

        p = np.sum(np.multiply(np.expand_dims(doc_length, -1), doc_topic_dist), axis=0) / np.sum(doc_length)
        p_dict = {topic_id: p[i] for i, topic_id in enumerate(self.topic_ids)}
        recur_p_topic = {parent_id: np.sum([p_dict[child_id] for child_id in recur_child_ids]) for
                         parent_id, recur_child_ids in self.topic2descendant.items()}

        def add_topic(topic_id, par2child):

            if topic_id in par2child:
                child_id = min([self.max_children_per_parent * topic_id + i for i in range(1, self.max_children_per_parent + 1) if
                                self.max_children_per_parent * topic_id + i not in par2child[topic_id]])
                par2child[topic_id].append(child_id)
            else:
                child_id = self.max_children_per_parent * topic_id + 1
                par2child[topic_id] = [self.max_children_per_parent * topic_id + 1]

            return child_id, par2child

        def remove_topic(parent_id, child_id, par2child):

            if parent_id in par2child:
                par2child[parent_id].remove(child_id)
                if child_id in par2child:
                    par2child.pop(child_id)

            return par2child

        added_par2child = copy.deepcopy(self.par2child)
        for parent_id, child_ids in self.par2child.items():
            prob_topic = p_dict[parent_id]
            if prob_topic > self.add_threshold:
                self.update_tree_flg = True
                parent_id, added_par2child = add_topic(parent_id, added_par2child)

        removed_par2child = copy.deepcopy(added_par2child)
        for parent_id, child_ids in self.par2child.items():
            probs_child = np.array([recur_p_topic[child_id] for child_id in child_ids])
            for prob_child, child_id in zip(probs_child, child_ids):
                if prob_child < self.remove_threshold:
                    self.update_tree_flg = True
                    removed_par2child = remove_topic(parent_id, child_id, removed_par2child)
                    if parent_id in removed_par2child:
                        if len(removed_par2child[parent_id]) == 0:
                            ancestor_id = self.child2par[parent_id]
                            removed_par2child = remove_topic(ancestor_id, parent_id, removed_par2child)

        self.par2child = removed_par2child
        self.construct_tree(self.par2child)