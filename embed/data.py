import torch

import gzip, random, sys, os, wget, pickle, tqdm
from collections import Counter
import util

VALPROP = 0.4
REST = '.rest'
INV  = '.inv'

S = os.sep

def add_neighbors(set, graph, node, depth=2):

    if depth == 0:
        return

    for s, p, o in graph.triples((node, None, None)):
        set.add((s, p, o))
        add_neighbors(set, graph, o, depth=depth-1)

    for s, p, o in graph.triples((None, None, node)):
        set.add((s, p, o))
        add_neighbors(set, graph, s, depth=depth-1)

def load_strings(file):

    with open(file, 'r') as f:
        return [line.split() for line in f]

def load(name, limit=None):
    """
    Loads a knowledge graph dataset for link prediction purposes.

    :param name: Dataset name. "fb" for FB15k-237, "wn" for WN18k-RR, "toy" for a small toy dataset for testing.
    :param limit: If set, the total numnber of triples per set will be limited to this value. Useful for debugging.
    :return: Three lists of integer-triples (train, val, test), a pair of dicts to map entity strings from an to their
        integer ids, and a similar pair of dicts for the relations.
    """

    if name == 'fb': # Freebase 15k 237
        train_file = util.here('data/fb15k237/train.txt')
        val_file = util.here('data/fb15k237/valid.txt')
        test_file = util.here('data/fb15k237/test.txt')

    elif name == 'wn':
        train_file = util.here('data/wn18rr/train.txt')
        val_file = util.here('data/wn18rr/valid.txt')
        test_file = util.here('data/wn18rr/test.txt')

    else:
        if os.path.isdir(util.here('data' + os.sep + name )):
            train_file = util.here(f'data/{name}/train.txt')
            val_file = util.here(f'data/{name}/valid.txt')
            test_file = util.here(f'data/{name}/test.txt')

        else:
            raise Exception(f'Could not find dataset with name {name} at location {util.here("data" + os.sep + name)}.')

    train = load_strings(train_file)
    val   = load_strings(val_file)
    test  = load_strings(test_file)

    if limit:
        train = train[:limit]
        val = val[:limit]
        test = test[:limit]

    # mappings for nodes (n) and relations (r)
    nodes, rels = set(), set()
    for triple in train + val + test:
        nodes.add(triple[0])
        rels.add(triple[1])
        nodes.add(triple[2])

    i2n, i2r = list(nodes), list(rels)
    n2i, r2i = {n:i for i, n in enumerate(nodes)}, {r:i for i, r in enumerate(rels)}

    traini, vali, testi = [], [], []

    for s, p, o in train:
        traini.append([n2i[s], r2i[p], n2i[o]])

    for s, p, o in val:
        vali.append([n2i[s], r2i[p], n2i[o]])

    for s, p, o in test:
        testi.append([n2i[s], r2i[p], n2i[o]])

    train, val, test = torch.tensor(traini), torch.tensor(vali), torch.tensor(testi)

    return train, val, test, (n2i, i2n), (r2i, i2r)