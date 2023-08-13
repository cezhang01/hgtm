from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import manifolds
import torch


def classification_knn(X_train, X_test, Y_train, Y_test):

    for k in [5, 10, 15, 20]:
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, Y_train)
        prediction_label = classifier.predict(X_test)
        print('Micro F1 %d: %.4f' % (k, f1_score(Y_test, prediction_label, average='micro')))
        print('Macro F1 %d: %.4f' % (k, f1_score(Y_test, prediction_label, average='macro')))


def link_prediction_auc(test_embeds, test_links, num_training_docs, args):

    test_links = np.copy(test_links)
    test_links -= num_training_docs
    auc, count = 0, 0
    for row_idx, row in enumerate(test_embeds):
        y_true = np.zeros(len(test_embeds))
        citations = test_links[test_links[:, 0] == row_idx]
        if len(citations) == 0:
            continue
        y_true[citations[:, 1]] = 1
        y_true = np.delete(y_true, row_idx)
        if np.sum(y_true) == 0:
            continue
        manifold = getattr(manifolds, args.manifold)()
        distance = manifold.sqdist(torch.tensor(np.delete(test_embeds, row_idx, axis=0)), torch.tensor(row), args.init_curvature).cpu().detach().numpy()
        y_score = - distance
        auc_tmp = roc_auc_score(y_true, y_score)
        auc += auc_tmp
        count += 1
    auc /= count
    print('Link prediction AUC: %.4f' % auc)


def perplexity(y_pred, y_true):

    power = - np.sum(np.multiply(np.log(y_pred + 1e-12), y_true)) / (np.sum(y_true) + 1e-12)
    print('Perplexity: %.4f' % power)


num_top_words = 10


def output_topic_keywords(topic_word_dist, voc, tree):

    num_keywords = 10
    index = np.argsort(topic_word_dist, axis=1)[:, ::-1][:, :num_keywords]
    keywords = voc[index]
    output_topic_keywords_func(keywords, tree)


def output_topic_keywords_func(keywords, tree, parent_id=0, level=0):

    if level == 0:  # print root
        keywords_one_topic = keywords[tree.topic_ids.index(parent_id)]
        print(parent_id, ' '.join(keywords_one_topic))

    child_ids = tree.par2child[parent_id]
    level += 1
    for child_id in child_ids:
        keywords_one_topic = keywords[tree.topic_ids.index(child_id)]
        print('  ' * level, child_id, ' '.join(keywords_one_topic))
        if child_id in tree.par2child:
            output_topic_keywords_func(keywords=keywords, tree=tree, parent_id=child_id, level=level)