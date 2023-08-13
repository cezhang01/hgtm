from torch.utils.data import Dataset
import numpy as np
import collections
import os
from sklearn.metrics.pairwise import cosine_similarity


class DataCenter():

    def __init__(self, args):

        self.parse_args(args)
        self.load_data()
        self.split_data()
        self.sample_neighbors()

    def parse_args(self, args):

        self.dataset_name = args.dataset_name
        self.minibatch_size = args.minibatch_size
        self.training_ratio = args.training_ratio
        self.num_sampled_neighbors = args.num_sampled_neighbors
        self.ptr_word_emb_model = 'glove'
        self.ptr_word_emb_dim = 300

    def load_data(self):

        self.words, self.doc_length = self.load_files('./data/' + self.dataset_name + '/contents.txt')
        self.num_docs = len(self.words)
        self.links = self.symmetrize_links(np.loadtxt('./data/' + self.dataset_name + '/links.txt', dtype=int))
        self.num_links = len(self.links)
        self.ptr_word_emb = np.loadtxt('./data/' + self.dataset_name + '/word_embeddings_' + self.ptr_word_emb_model + '_' + str(self.ptr_word_emb_dim) + 'd.txt', dtype=float)
        self.labels_available = os.path.isfile('./data/' + self.dataset_name + '/labels.txt')
        if self.labels_available:
            self.labels = np.loadtxt('./data/' + self.dataset_name + '/labels.txt')
            self.num_labels = len(np.unique(self.labels))
        self.voc = np.genfromtxt('./data/' + self.dataset_name + '/voc.txt', dtype=str)
        self.num_words = len(self.voc)

        self.doc_raw_feat = []
        for doc_id in range(len(self.words)):
            self.doc_raw_feat.append(np.mean(self.ptr_word_emb[self.words[doc_id]], axis=0))
        self.doc_raw_feat = np.array(self.doc_raw_feat)

    def load_files(self, file_path):

        files = collections.defaultdict(list)
        length = []
        with open(file_path) as file:
            for row_id, row in enumerate(file):
                row = row.split()
                row = [int(i) for i in row]
                files[row_id] = row
                length.append(len(row))

        return files, length

    def symmetrize_links(self, links):

        symmetric_links = []
        for link in links:
            if link[0] == link[1]:
                continue
            symmetric_links.append([link[0], link[1]])
            symmetric_links.append([link[1], link[0]])
            # symmetric_links.append([link[0], link[0]])
            # symmetric_links.append([link[1], link[1]])
        symmetric_links = np.unique(symmetric_links, axis=0)

        return symmetric_links

    def split_data(self):

        self.collect_doc_doc_neighbors()
        self.collect_word_doc_neighbors()
        self.training_doc_length, self.test_doc_length = self.doc_length[:self.num_training_docs], self.doc_length[self.num_training_docs:]

        if self.labels_available:
            self.training_labels, self.test_labels = self.labels[:self.num_training_docs], self.labels[self.num_training_docs:]

    def collect_doc_doc_neighbors(self):

        self.num_training_docs = int(self.num_docs * self.training_ratio)
        self.num_test_docs = self.num_docs - self.num_training_docs
        doc_doc_neighbors_set, self.training_links, self.test_links = collections.defaultdict(set), [], []
        for link in self.links:
            if link[0] < self.num_training_docs and link[1] < self.num_training_docs:
                doc_doc_neighbors_set[link[0]].add(link[1])
                doc_doc_neighbors_set[link[1]].add(link[0])
                self.training_links.append([link[0], link[1]])
                self.training_links.append([link[1], link[0]])
            elif link[0] >= self.num_training_docs and link[1] < self.num_training_docs:
                doc_doc_neighbors_set[link[0]].add(link[1])
            elif link[0] < self.num_training_docs and link[1] >= self.num_training_docs:
                doc_doc_neighbors_set[link[1]].add(link[0])
            elif link[0] >= self.num_training_docs and link[1] >= self.num_training_docs:
                self.test_links.append([link[0], link[1]])
                self.test_links.append([link[1], link[0]])

        for doc_id in range(self.num_docs):  # add self-loop for docs with no links
            if doc_id not in doc_doc_neighbors_set:
                doc_doc_neighbors_set[doc_id].add(doc_id)

        self.doc_doc_neighbors_dict = collections.defaultdict(list)
        for doc_id in doc_doc_neighbors_set:
            self.doc_doc_neighbors_dict[doc_id] = list(doc_doc_neighbors_set[doc_id])

        self.training_links = np.unique(self.training_links, axis=0)
        self.test_links = np.unique(self.test_links, axis=0)
        self.num_training_links, self.num_test_links = len(self.training_links), len(self.test_links)

    def collect_word_doc_neighbors(self):

        self.word_doc_neighbors_dict = collections.defaultdict(list)
        for doc_id in range(self.num_training_docs):
            word_ids = self.words[doc_id]
            for word_id in np.unique(word_ids):
                self.word_doc_neighbors_dict[word_id].append(doc_id)

        if len(self.word_doc_neighbors_dict) < self.num_words:
            for word_id in range(self.num_words):
                if word_id not in self.word_doc_neighbors_dict:
                    cos_sim = cosine_similarity(self.ptr_word_emb, np.array([self.ptr_word_emb[word_id]]))
                    for related_word_idx in range(1000):
                        related_word_id = np.argsort(cos_sim[0])[-related_word_idx]
                        if related_word_id in self.word_doc_neighbors_dict:
                            self.word_doc_neighbors_dict[word_id] = self.word_doc_neighbors_dict[related_word_id]
                            break

    def sample_neighbors(self):

        self.doc_word_neighbors = self.sample_neighbors_func(self.words)
        self.word_doc_neighbors = self.sample_neighbors_func(self.word_doc_neighbors_dict)
        self.doc_doc_neighbors = self.sample_neighbors_func(self.doc_doc_neighbors_dict)

    def sample_neighbors_func(self, neighbors):

        sampled_neighbors = []
        keys = np.sort([key for key in neighbors.keys()])
        # for id in range(len(neighbors)):
        for id in keys:
            replace = len(neighbors[id]) < self.num_sampled_neighbors
            neighbor_idx = np.random.choice(len(neighbors[id]), size=self.num_sampled_neighbors, replace=replace)
            sampled_neighbors.append([neighbors[id][idx] for idx in neighbor_idx])
        sampled_neighbors = np.array(sampled_neighbors)

        return sampled_neighbors

    def generate_bow(self, doc_ids, normalize=False):

        docs_bow = np.zeros([len(doc_ids), self.num_words], dtype=float)
        for idx, doc_id in enumerate(doc_ids):
            for word_id in self.words[int(doc_id)]:
                docs_bow[idx, word_id] += 1
            if normalize and np.sum(docs_bow[idx]) != 0:
                docs_bow[idx] = docs_bow[idx] / np.sum(docs_bow[idx])

        return docs_bow


class Data(Dataset):

    def __init__(self, args, mode, data):

        super(Data, self).__init__()
        self.parse_args(args, mode, data)

    def parse_args(self, args, mode, data):

        self.num_negative_samples = args.num_negative_samples
        self.mode = mode
        self.data = data

        self.links = self.data.training_links if self.mode == 'train' else np.array([[doc_id, doc_id] for doc_id in range(self.data.num_docs)])

    def __len__(self):

        return len(self.links)

    def __getitem__(self, idx):

        link = self.links[idx]
        doc_ids_neg = np.random.randint(self.data.num_training_docs, size=self.num_negative_samples)

        return link, doc_ids_neg