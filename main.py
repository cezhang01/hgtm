import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from data_loader import *
from model import Model
from tree import Tree
import manifolds
import os
import time
from tqdm import tqdm
from evaluation import *


def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters for model training
    parser.add_argument('-ne', '--num_epochs', type=int, default=200)  # set a small num of epochs for large datasets, such as 10
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-ms', '--minibatch_size', type=int, default=64)
    parser.add_argument('-dn', '--dataset_name', type=str, default='ml', choices=['ml', 'pl', 'covid', 'aminer', 'web'])
    parser.add_argument('-dim', '--emb_dim', type=int, default=16)
    parser.add_argument('-s', '--supervision', type=bool, default=False)
    parser.add_argument('-reg_s', '--reg_s', type=float, default=1)
    parser.add_argument('-reg_text', '--reg_text', type=float, default=0.1)
    parser.add_argument('-reg_kld', '--reg_kld', type=float, default=0)
    parser.add_argument('-tr', '--training_ratio', type=float, default=0.8)
    parser.add_argument('-m', '--manifold', type=str, default='PoincareBall', choices=['PoincareBall', 'Hyperboloid'])
    parser.add_argument('-le', '--log_epochs', type=int, default=25)  # set a small num of log epochs for large datasets, such as 1

    # hyperparameters for encoder
    parser.add_argument('-nl', '--num_conv_layers', type=int, default=2)
    parser.add_argument('-nn', '--num_sampled_neighbors', type=int, default=5)
    parser.add_argument('-neg', '--num_negative_samples', type=int, default=5)
    parser.add_argument('-c', '--init_curvature', type=float, default=1.0)
    parser.add_argument('-b', '--use_bias', type=bool, default=True)

    # hyperparameters for topic tree (decoder)
    parser.add_argument('-ut', '--update_tree', type=bool, default=True)
    parser.add_argument('-max_l', '--max_levels', type=int, default=4)
    parser.add_argument('-max_c', '--max_children_per_parent', type=int, default=20)
    parser.add_argument('-at', '--add_threshold', type=float, default=0.05)
    parser.add_argument('-rt', '--remove_threshold', type=float, default=0.05)
    parser.add_argument('-we', '--use_ptr_word_emb', type=bool, default=False)

    parser.add_argument('-rs', '--random_seed', type=int, default=519)
    parser.add_argument('-gpu', '--gpu', type=int, default=0)

    return parser.parse_args()


def train(args):

    args.device = 'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu'

    print('Preparing data...')
    data_center = DataCenter(args)
    training_data = Data(args, 'train', data_center)
    test_data = Data(args, 'test', data_center)
    training_loader = DataLoader(dataset=training_data, batch_size=args.minibatch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=args.minibatch_size, shuffle=False)

    print('Start training...')
    manifold = getattr(manifolds, args.manifold)()
    tree = Tree(args, manifold).to(args.device)
    model = Model(args, data_center, manifold).to(args.device)
    print(model)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)
    num_minibatches = len(training_loader)
    t = time.time()
    print('Current tree structure:', tree.par2child)
    for epoch_idx in tqdm(range(1, args.num_epochs + 1)):
        # training
        one_epoch_loss = 0.0
        model.train()
        data_center.sample_neighbors()
        for idx, batch in tqdm(enumerate(training_loader)):
            links, doc_ids_neg = batch
            doc_ids_neg = np.reshape(doc_ids_neg, [-1])
            optimizer.zero_grad()
            res = model(links, data_center, tree, mode='train')
            loss = res[0][0]
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                one_epoch_loss = loss.item()

        # testing
        if epoch_idx % args.log_epochs == 0 or epoch_idx == 1:
            print('******************************************************')
            print('Time: %ds' % (time.time() - t), '\tEpoch: %d/%d' % (epoch_idx, args.num_epochs), '\tLoss: %f' % one_epoch_loss)

            model.eval()
            doc_emb, y_pred, bow_pred, topic_word_dist, doc_topic_dist, topic_emb = [], [], [], [], [], []
            for idx, batch in enumerate(test_loader):
                links, _ = batch
                res = model(links, data_center, tree, mode='test')
                doc_emb_tmp, bow_pred_tmp, topic_word_dist, doc_topic_dist_tmp, topic_emb = res[1], res[2], res[3], res[4], res[5]
                doc_emb_tmp = doc_emb_tmp.detach().cpu().numpy().tolist()
                bow_pred_tmp = bow_pred_tmp.detach().cpu().numpy().tolist()
                topic_word_dist = topic_word_dist.detach().cpu().numpy()
                doc_topic_dist_tmp = doc_topic_dist_tmp.detach().cpu().numpy().tolist()
                topic_emb = topic_emb.detach().cpu().numpy()
                doc_emb.extend(doc_emb_tmp)
                bow_pred.extend(bow_pred_tmp)
                doc_topic_dist.extend(doc_topic_dist_tmp)
                if args.supervision:
                    y_pred_tmp = res[-1]
                    y_pred_tmp = y_pred_tmp.detach().cpu().numpy().tolist()
                    y_pred.extend(y_pred_tmp)
            doc_emb = np.array(doc_emb)
            training_doc_emb = doc_emb[:data_center.num_training_docs]
            test_doc_emb = doc_emb[data_center.num_training_docs:]
            test_bow_pred = np.array(bow_pred[data_center.num_training_docs:])
            training_doc_topic_dist = np.array(doc_topic_dist[:data_center.num_training_docs])
            test_doc_topic_dist = np.array(doc_topic_dist[data_center.num_training_docs:])
            topic_emb = np.array(topic_emb)
            if args.supervision:
                y_pred_test = np.array(y_pred[data_center.num_training_docs:])

            # evaluation
            output_topic_keywords(topic_word_dist, data_center.voc, tree)
            if data_center.labels_available:
                if args.supervision:
                    print('Micro F1: %.4f' % f1_score(data_center.test_labels, y_pred_test, average='micro'))
                    print('Macro F1: %.4f' % f1_score(data_center.test_labels, y_pred_test, average='macro'))
                else:
                    classification_knn(training_doc_emb, test_doc_emb, data_center.training_labels, data_center.test_labels)
            link_prediction_auc(test_doc_emb, data_center.test_links, data_center.num_training_docs, args)
            test_bow_true = data_center.generate_bow(range(data_center.num_training_docs, data_center.num_docs), normalize=False)
            perplexity(test_bow_pred, test_bow_true)

        # update tree
        if args.update_tree and epoch_idx % args.log_epochs == 0 and epoch_idx != args.num_epochs:
            tree.update_tree(training_doc_topic_dist, data_center.training_doc_length)
            if tree.update_tree_flg:
                print('Current tree structure:', tree.par2child)

        # save model outputs
        if epoch_idx % args.log_epochs == 0:
            folder = os.path.exists('./data/' + args.dataset_name + '/results')
            if not folder:
                os.makedirs('./data/' + args.dataset_name + '/results')
            np.savetxt('./data/' + args.dataset_name + '/results/doc_emb.txt', doc_emb, delimiter=' ', fmt='%.4f')
            np.savetxt('./data/' + args.dataset_name + '/results/topic_emb.txt', topic_emb, delimiter=' ', fmt='%.4f')
            np.savetxt('./data/' + args.dataset_name + '/results/topic_word_dist.txt', topic_word_dist, delimiter=' ', fmt='%.4f')


def main(args):

    if args.random_seed:
        np.random.seed(args.random_seed)
        torch.random.manual_seed(args.random_seed)
    train(args)


if __name__ == '__main__':
    main(parse_args())