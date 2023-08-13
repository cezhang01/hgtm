import torch
import torch.nn as nn
import numpy as np
from encoder import Encoder
from decoder import GraphDecoder, TextDecoder, TopicEmbReg, HypTopicEmbReg
from doubly_rnn import DoublyRNN


class Model(nn.Module):

    def __init__(self, args, data, manifold):

        super(Model, self).__init__()
        self.data = data
        self.manifold = manifold
        self.parse_args(args)
        self.show_config()
        self.generate_modules(args)

    def parse_args(self, args):

        self.device = args.device
        self.dataset_name = args.dataset_name
        self.training_ratio = args.training_ratio
        self.num_docs = self.data.num_docs
        self.num_training_docs = self.data.num_training_docs
        self.num_test_docs = self.data.num_test_docs
        self.num_links = self.data.num_links
        self.num_training_links = self.data.num_training_links
        self.num_words = self.data.num_words
        args.num_words = self.num_words
        if self.data.labels_available:
            self.num_labels = self.data.num_labels
            args.num_labels = self.num_labels
        self.ptr_word_emb_model = self.data.ptr_word_emb_model
        self.ptr_word_emb_dim = self.data.ptr_word_emb_dim
        args.ptr_word_emb_dim = self.ptr_word_emb_dim
        self.num_conv_layers = args.num_conv_layers
        self.num_sampled_neighbors = args.num_sampled_neighbors
        self.num_negative_samples = args.num_negative_samples
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.minibatch_size = args.minibatch_size
        self.emb_dim = args.emb_dim
        self.supervision = args.supervision
        self.reg_s = args.reg_s
        self.reg_text = args.reg_text
        self.init_curvature = args.init_curvature
        self.use_bias = args.use_bias
        self.update_tree = args.update_tree
        self.max_levels = args.max_levels
        self.max_children_per_parent = args.max_children_per_parent
        self.add_threshold = args.add_threshold
        self.remove_threshold = args.remove_threshold
        self.use_ptr_word_emb = args.use_ptr_word_emb
        self.manifold_name = self.manifold.name
        self.dropout_keep_prob = 1.0
        args.dropout_keep_prob = self.dropout_keep_prob
        if self.use_ptr_word_emb:
            self.word_emb_dim = self.ptr_word_emb_dim
        else:
            self.word_emb_dim = 256
        args.word_emb_dim = self.word_emb_dim

    def show_config(self):

        print('******************************************************')
        print('torch version:', torch.__version__)
        print('np version:', np.__version__)
        print('dataset name:', self.dataset_name)
        print('training ratio:', self.training_ratio)
        print('#documents:', self.num_docs)
        print('#training documents:', self.num_training_docs)
        print('#total links:', self.num_links)
        print('#training links:', self.num_training_links)
        print('#words:', self.num_words)
        if self.data.labels_available:
            print('#labels:', self.num_labels)
        print('#convolutional layers:', self.num_conv_layers)
        print('#sampled neighbors:', self.num_sampled_neighbors)
        print('#negative samples:', self.num_negative_samples)
        print('#epochs:', self.num_epochs)
        print('learning rate:', self.learning_rate)
        print('minibatch size:', self.minibatch_size)
        print('dimension of embeddings:', self.emb_dim)
        print('supervision:', self.supervision)
        print('regularizer for supervision:', self.reg_s)
        print('initial curvature:', self.init_curvature)
        print('use bias:', self.use_bias)
        print('update tree:', self.update_tree)
        print('maximum num of tree levels:', self.max_levels)
        print('maximum num of children per parent:', self.max_children_per_parent)
        print('adding threshold:', self.add_threshold)
        print('removal threshold:', self.remove_threshold)
        print('use pretrained word embeddings:', self.use_ptr_word_emb)
        print('manifold:', self.manifold_name)
        print('******************************************************')

    def generate_modules(self, args):

        self.encoder = Encoder(args, self.manifold)
        self.graph_decoder = GraphDecoder(args, self.manifold)
        self.text_decoder = TextDecoder(args)
        self.topic_emb_reg = TopicEmbReg(args)
        self.hyp_topic_emb_reg = HypTopicEmbReg(args, self.manifold)

        self.drnn = nn.ModuleDict()
        self.drnn['doc_topic_dist'] = DoublyRNN(self.emb_dim - 1 if self.manifold.name == 'Hyperboloid' else self.emb_dim, self.manifold, self.device, name='evaluate_doc_topic_dist')
        self.drnn['topic_word_dist'] = DoublyRNN(self.word_emb_dim, self.manifold, self.device, name='evaluate_topic_word_dist')

        if self.use_ptr_word_emb:
            self.word_emb = torch.FloatTensor(self.data.ptr_word_emb).to(self.device)
        else:
            self.word_emb = nn.ModuleList([nn.Embedding(self.num_words, self.word_emb_dim).to(self.device)])

        if self.supervision:
            self.clf_layer = nn.ModuleList([nn.Linear(self.emb_dim, self.num_labels)])
            self.ce_loss = nn.functional.cross_entropy
            self.softmax = nn.Softmax(dim=-1)

    def hyp_clf(self, emb, labels, c):

        emb_tan = self.manifold.logmap0(emb, c)
        logits = self.clf_layer[0](emb_tan)
        y_pred = torch.argmax(logits, dim=-1)
        one_hot = nn.functional.one_hot(torch.tensor(labels, dtype=torch.long).to(self.device), num_classes=self.num_labels)
        one_hot = one_hot.float()
        y_pred_prob = self.softmax(logits)
        y_pred_prob = torch.clamp(y_pred_prob, min=1e-12)
        loss = - torch.sum(torch.multiply(one_hot, torch.log(y_pred_prob)), dim=-1)
        loss = torch.mean(loss)

        return loss, y_pred

    def forward(self, links, data, tree, mode='train'):

        self.data = data

        # encoder
        doc_emb_1, word_emb_1 = self.encoder(links[:, 0], self.data, mode=mode)
        doc_emb_2, word_emb_2 = self.encoder(links[:, 1], self.data, mode=mode)
        doc_neg_indices = torch.randint(2 * doc_emb_1.size(0), size=[self.num_negative_samples * doc_emb_1.size(0)])
        doc_emb_neg = torch.concat([doc_emb_1, doc_emb_2], dim=0)[doc_neg_indices]
        # word_neg_indices = torch.randint(2 * word_emb_1.size(0), size=[self.num_negative_samples * word_emb_1.size(0)])
        # word_emb_neg = torch.concat([word_emb_1, word_emb_2], dim=0)[word_neg_indices]

        # graph decoder
        doc_doc_loss = self.graph_decoder(doc_emb_1, doc_emb_2, doc_emb_neg, self.init_curvature)
        # doc_word_loss_1 = self.graph_decoder(doc_emb_1, word_emb_1, word_emb_neg, self.init_curvature)
        # doc_word_loss_2 = self.graph_decoder(doc_emb_2, word_emb_2, word_emb_neg, self.init_curvature)
        graph_loss = doc_doc_loss #+ self.reg_text * (doc_word_loss_1 + doc_word_loss_2)

        # text decoder
        topic_emb_hyp_space = self.drnn['doc_topic_dist'](tree.par2child)
        doc_topic_dist_1 = tree.evaluate_doc_topic_dist(doc_emb_1, topic_emb_hyp_space, self.init_curvature)
        doc_topic_dist_2 = tree.evaluate_doc_topic_dist(doc_emb_2, topic_emb_hyp_space, self.init_curvature)
        if self.use_ptr_word_emb:
            topic_word_dist, topic_emb_word_space = tree.evaluate_topic_word_dist(self.drnn['topic_word_dist'], self.word_emb)
        else:
            topic_word_dist, topic_emb_word_space = tree.evaluate_topic_word_dist(self.drnn['topic_word_dist'], self.word_emb[0].weight)
        bow_true_1 = torch.FloatTensor(self.data.generate_bow(links[:, 0], normalize=False)).to(self.device)
        bow_true_2 = torch.FloatTensor(self.data.generate_bow(links[:, 1], normalize=False)).to(self.device)
        text_loss_1, bow_pred = self.text_decoder(doc_topic_dist_1, topic_word_dist, bow_true_1)
        text_loss_2, _ = self.text_decoder(doc_topic_dist_2, topic_word_dist, bow_true_2)
        text_loss = text_loss_1 + text_loss_2

        # topic embedding regularizer
        reg_loss = self.topic_emb_reg(topic_emb_word_space, tree)
        reg_loss_hyp = self.hyp_topic_emb_reg(topic_emb_hyp_space, tree)

        loss = graph_loss + self.reg_text * text_loss + 1 * reg_loss + 1000 * reg_loss_hyp

        topic_emb_hyp_space = torch.concat([topic_emb_hyp_space[topic_id] for topic_id in tree.topic_ids], dim=0)

        if self.supervision:
            clf_loss_1, y_pred = self.hyp_clf(doc_emb_1, self.data.labels[links[:, 0]], self.init_curvature)
            clf_loss_2, _ = self.hyp_clf(doc_emb_2, self.data.labels[links[:, 1]], self.init_curvature)
            clf_loss = clf_loss_1 + clf_loss_2
            loss += self.reg_s * clf_loss

            return [[loss, graph_loss, self.reg_text * text_loss, reg_loss, self.reg_s * clf_loss], doc_emb_1, bow_pred, topic_word_dist, doc_topic_dist_1, topic_emb_hyp_space, y_pred]

        return [[loss, graph_loss, self.reg_text * text_loss, reg_loss], doc_emb_1, bow_pred, topic_word_dist, doc_topic_dist_1, topic_emb_hyp_space]