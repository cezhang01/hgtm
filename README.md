# HGTM
This is the pytorch implementation of KDD-2023 paper "[Hyperbolic Graph Topic Modeling Network with Continuously Updated Topic Tree](/papers/KDD23-HGTM.pdf)" by [Delvin Ce Zhang](http://delvincezhang.com/), [Rex Ying](https://www.cs.yale.edu/homes/ying-rex/), and [Hady W. Lauw](http://www.hadylauw.com/home).

HGTM is a topic model designed for interconnected texts in a graph structure, such as academic citation graphs and Webpage hyperlink graphs. HGTM learns text embeddings with the aim of preserving both hierarchical graph connectivity and hierarchical topic structure within textual content. The learned text embeddings can fulfill downstream tasks, inclduing text classification, graph link prediction, and topic analysis.

## Implementation Environment
- python == 3.10
- pytorch == 2.0.0
- numpy == 1.20.3

## Run
`python main.py -s False`  # unsupervised training

`python main.py -s True`   # supervised training

### Parameter Setting
- -ne: number of training epochs, default = 200 (set 200 for small datasets (ml, pl, covid), and 10 for large datasets (aminer and web))
- -lr: learning rate, default = 0.01
- -ms: minibatch size, default = 64
- -dn: dataset name, default = ml
- -dim: dimension of text embeddings H, default = 16
- -s: supervised training, default = False (set True for supervised training, which involves a MLP classifier and document labels for training. See Eq. 26 in the paper.)
- -reg_s: regularizer for MLP classifier in supervised training, default = 1 ($\lambda_{label}$ in the paper. If the above -s is set to False, this -reg_s can be ignored.)
- -reg_text: regularizer for text generation, default = 0.1 ($\lambda_{text}$ in the paper)
- -tr: training ratio, the ratio of training documents to the total documents, default = 0.8
- -le: log epochs, how many epochs should the model output results on testing documents, default = 25 (set 25 for small datasets (ml, pl, covid), and 1 for large datasets (aminer and web))
- -nl: number of graph convolutional layers, default = 2
- -nn: number of sampled neighbors for aggregation, default = 5
- -neg: number of negative samples for graph structure decoding, default = 5
- -c: curvature, default = 1.0
- -b: if the model adds bias for encoding, default = True
- -ut: if the model updates the topic tree during training, default = True
- -max_l: maximum number of levels for topic tree, default = 4
- -max_c: maximum number of children a parent topic has, default = 20
- -at: threshold of adding a new topic, default = 0.05
- -rt: thrshold of remving a topic, default = 0.05
- -we: if the model uses pretrained word embeddings for decoding, default = False
- -rs: random seed
- -gpu: gpu

Please note that some hyperparameters are different from the ones reported in the paper. This is because we used tensorflow to implement the model when doing research of this paper, while we reproduce the results using pytorch upon publication. Different libraries may result in slight deviations of implementation, but the results reported in the paper are reproducible.

## Data
We release ML, PL, COVID, and Aminer datasets in `./data` folder. For Aminer dataset, please unzip `aminer.zip` and put the unzipped file into `./data` folder. For the largest Web dataset, please email Delvin Ce Zhang (delvincezhang@gmail.com) for access.

Each dataset contains contents, links, vocabulary, pretrained glove word embeddings, labels, and label names.

- contents: each row corresponds to a document, containing a sequence of words represented by word IDs in vocabulary. Word ID starts from 0. For example, a row with `[0, 6, 4]` means a document with a sequence of three words, i.e., the 0th, 6th, and 4th words in the vocabulary. There are N rows (N documents) in total.
- links: each row corresponds to a link represented by a pair of document IDs. For example, a row  with `[5, 8]` means a link from document 5 to document 8.
- voc (|V|x1): vocabulary.
- pretrained glove word embeddings (|V|x300): 300-dimensional pretrained glove word embeddings, with each row corresponding to a word in the vocabulary. For example, the 0th word embedding corresponds to the 0th word in the vocabulary.
- labels (Nx1): labels or categories of N documents. For example, the 0th label corresponds to the 0th document.
- label names: the names of labels or categories.

## Output
We output three files for each training. Output results are saved to the `./dataset_name/results` folder.

- doc_emb.txt (NxH): each row is an H-dimensional document embedding. There are N documents in total. The first 80% embeddings are for training documents, and the remaining 20% embeddings are for testing documents. These document embeddings can fulfill downstream tasks, including text classification and graph link prediction.
- topic_emb.txt (KxH): each row is an H-dimensional topic embedding.
- topic_word_dist.txt (Kx|V|): topic-word distribution, where each row is a probability distribution over the whole vocabulary. This topic-word distribution can fulfill topic analysis tasks, including topic coherence.

## Reference
If you find our paper useful, including code and data, please cite

```
@inproceedings{hgtm,
    author = {Zhang, Delvin Ce and Ying, Rex and Lauw, Hady W.},
    title = {Hyperbolic Graph Topic Modeling Network with Continuously Updated Topic Tree},
    year = {2023},
    booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    pages = {3206–3216},
    location = {Long Beach, CA, USA},
    series = {KDD '23}
}
```
