import torch, os, sys, random

from torch import nn
import torch.nn.functional as F

import util
from util import d, tic, toc

def distmult(triples, nodes, relations, biases=None, forward=True):
    """
    Implements the distmult score function.

    :param triples: batch of triples, (b, 3) integers
    :param nodes: node embeddings
    :param relations: relation embeddings
    :return:
    """

    b, _ = triples.size()

    a, b = (0, 2) if forward else (2, 0)

    si, pi, oi = triples[:, a], triples[:, 1], triples[:, b]

    # s, p, o = nodes[s, :], relations[p, :], nodes[o, :]

    # faster?
    s = nodes.index_select(dim=0,     index=si)
    p = relations.index_select(dim=0, index=pi)
    o = nodes.index_select(dim=0,     index=oi)

    baseterm = (s * p * o).sum(dim=1)

    if biases is None:
        return baseterm

    gb, sb, pb, ob = biases

    return baseterm + sb[si] + pb[pi] + ob[oi] + gb

def transe(triples, nodes, relations, biases=None, forward=True):
    """
    Implements the distmult score function.

    :param triples: batch of triples, (b, 3) integers
    :param nodes: node embeddings
    :param relations: relation embeddings
    :return:
    """

    b, _ = triples.size()

    a, b = (0, 2) if forward else (2, 0)

    si, pi, oi = triples[:, a], triples[:, 1], triples[:, b]

    # s, p, o = nodes[s, :], relations[p, :], nodes[o, :]

    # faster?
    s = nodes.index_select(dim=0,     index=si)
    p = relations.index_select(dim=0, index=pi)
    o = nodes.index_select(dim=0,     index=oi)

    baseterm = (s + p - o).norm(p=2, dim=1)

    if biases is None:
        return baseterm

    gb, sb, pb, ob = biases

    return baseterm + sb[si] + pb[pi] + ob[oi] + gb

class LinkPredictor(nn.Module):
    """
    Link prediction model with no message passing

    Outputs raw (linear) scores for the given triples.
    """

    def __init__(self, triples, n, r, embedding=512, decoder='distmult', edropout=None, rdropout=None, init=0.85, biases=False, reciprocal=False):

        super().__init__()

        assert triples.dtype == torch.long

        self.n, self.r = n, r
        self.e = embedding
        self.reciprocal = reciprocal

        self.entities  = nn.Parameter(torch.FloatTensor(n, self.e).uniform_(-init, init))
        self.relations = nn.Parameter(torch.FloatTensor(r, self.e).uniform_(-init, init))
        if reciprocal:
            self.relations_backward = nn.Parameter(torch.FloatTensor(r, self.e).uniform_(-init, init))

        if decoder == 'distmult':
            self.decoder = distmult
        elif decoder == 'transe':
            self.decoder = transe
        else:
            raise Exception()

        self.edo = None if edropout is None else nn.Dropout(edropout)
        self.rdo = None if rdropout is None else nn.Dropout(rdropout)

        self.biases = biases
        if biases:
            self.gbias = nn.Parameter(torch.zeros((1,)))
            self.sbias = nn.Parameter(torch.zeros((n,)))
            self.obias = nn.Parameter(torch.zeros((n,)))
            self.pbias = nn.Parameter(torch.zeros((r,)))

            if reciprocal:
                self.pbias_bw = nn.Parameter(torch.zeros((r,)))

    def forward(self, batch):

        scores = 0

        assert batch.size(-1) == 3

        n, r = self.n, self.r

        dims = batch.size()[:-1]
        batch = batch.reshape(-1, 3)

        for forward in [True, False] if self.reciprocal else [True]:

            nodes = self.entities
            relations = self.relations if forward else self.relations_backward

            if self.edo is not None:
                nodes = self.edo(nodes)
            if self.rdo is not None:
                relations = self.rdo(relations)

            if self.biases:
                if forward:
                    biases = (self.gbias, self.sbias, self.pbias,    self.obias)
                else:
                    biases = (self.gbias, self.sbias, self.pbias_bw, self.obias)
            else:
                biases = None

            scores = scores + self.decoder(batch, nodes, relations, biases=biases, forward=forward)

            assert scores.size() == (util.prod(dims), )

        if self.reciprocal:
            scores = scores / 2

        return scores.view(*dims)

    def penalty(self, rweight, p, which):

        # TODO implement weighted penalty

        if which == 'entities':
            params = [self.entities]
        elif which == 'relations':
            if self.reciprocal:
                params = [self.relations, self.relations_backward]
            else:
                params = [self.relations]
        else:
            raise Exception()

        if p % 2 == 1:
            params = [p.abs() for p in params]

        return (rweight / p) * sum([(p ** p).sum() for p in params])
