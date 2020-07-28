import torch, os, sys, random

from torch import nn
import torch.nn.functional as F

import util
from util import d, tic, toc

from abc import abstractmethod

class Decoder(nn.Module):

    def __init__(self, e):
        super().__init__()

        self.e = e

    def s_dim(self):
        return self.e

    def p_dim(self):
        return self.e

    def o_dim(self):
        return self.e

    @abstractmethod
    def forward(self, triples, corruptions, corr_index, entities, relations, forward):
        pass

class DistMult(Decoder):

    def __init__(self, e):
        super().__init__(e)

    def forward(self, si, pi, oi, nodes, relations):
        """
        Implements the distmult score function.
        """

        s, p, o = nodes[si, :], relations[pi, :], nodes[oi, :]

        if len(s.size()) == len(p.size()) == len(o.size()): # optimizations for common broadcasting
            if pi.size(-1) == 1 and oi.size(-1) == 1:
                singles = p * o # ignoring batch dimensions, this is a single vector
                return torch.matmul(s, singles.transpose(-1, -2)).squeeze(-1)

            if si.size(-1) == 1 and oi.size(-1) == 1:
                singles = s * o
                return torch.matmul(p, singles.transpose(-1, -2)).squeeze(-1)

            if si.size(-1) == 1 and pi.size(-1) == 1:
                singles = s * p
                return torch.matmul(o, singles.transpose(-1, -2)).squeeze(-1)

        return (s * p * o).sum(dim=-1)

class TransE(Decoder):

    def __init__(self, e):
        super().__init__(e)

    def forward(self, triples, nodes, relations, forward=True):
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
        s, p, o = nodes[si, :], relations[pi, :], nodes[oi, :]

        return (s + p - o).norm(p=2, dim=1)

def initialize(tensor, method, parms):
    if method == 'uniform':
        nn.init.uniform_(tensor, parms[0], parms[1])
    elif method == 'glorot_normal':
        nn.init.xavier_normal_(tensor, gain=parms[0])
    elif method == 'glorot_uniform':
        nn.init.xavier_uniform_(tensor, gain=parms[0])
    elif method == 'normal':
        nn.init.normal_(tensor, parms[0], parms[1])
    else:
        raise Exception(f'Initialization method {method} not recognized.')

class LinkPredictor(nn.Module):
    """
    Link prediction model with no message passing

    Outputs raw (linear) scores for the given triples.
    """

    def __init__(self, triples, n, r, embedding=512, decoder='distmult', edropout=None, rdropout=None, init=0.85,
                 biases=False, init_method='uniform', init_parms=(-1.0, 1.0), reciprocal=False):

        super().__init__()

        assert triples.dtype == torch.long

        self.n, self.r = n, r
        self.e = embedding
        self.reciprocal = reciprocal

        self.entities  = nn.Parameter(torch.FloatTensor(n, self.e))
        initialize(self.entities, init_method, init_parms)
        self.relations = nn.Parameter(torch.FloatTensor(r, self.e))
        initialize(self.relations, init_method, init_parms)

        if reciprocal:
            self.relations_backward = nn.Parameter(torch.FloatTensor(r, self.e).uniform_(-init, init))
            initialize(self.relations, init_method, init_parms)

        if decoder == 'distmult':
            self.decoder = DistMult(embedding)
        elif decoder == 'transe':
            self.decoder = TransE(embedding)
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

    def forward(self, s, p, o):
        """
        Takes a batch of triples in s, p, o indices, and computes their scores.

        If s, p and o have more than one dimension, and the same shape, the resulting score
        tensor has that same shape.

        If s, p and o have more than one dimension and mismatching shape, they are broadcast together
        and the score tensor has the broadcast shape. If broadcasting fails, the method fails. In order to trigger the
        correct optimizations, it's best to ensure that all tensors have the same dimensions.

        :param s:
        :param p:
        :param o:
        :return:
        """

        scores = 0

        for forward in [True, False] if self.reciprocal else [True]:

            si, pi, oi = (s, p, o) if forward else (o, p, s)

            nodes = self.entities
            relations = self.relations if forward else self.relations_backward

            # Apply dropout
            nodes = nodes if self.edo is None else self.edo(nodes)
            relations = relations if self.rdo is None else self.rdo(relations)

            scores = scores + self.decoder(si, pi, oi, nodes, relations)
            # -- We let the decoder handle the broadcasting

            if self.biases:
                pb = self.pbias if forward else self.pbias_bw,
                scores = scores + self.sbias[si] + pb[pi] + self.obias[oi] + self.gbias

        if self.reciprocal:
            scores = scores / 2

        return scores

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
