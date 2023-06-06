
import torch
from .util import d

"""
Utilities specific to the task of link prediction.
"""

def corruptions(positives, n, r, negative_rates=(10, 0, 10), loss='bce', ccandidates=None):
    """
    Generates corruptions for the given positives. In order, by corrupting the subjects, predicates and objects.


    :param positives: A tensor of size (b, 3) containing the positive triples.
    :param nrates:
    :param ccandidates: Corruption candidates. If not None, a triple (s, p ,o) where s is a list of the entities that
        can appear in the subject position (and likewise for p and o). This is usually take from the data. By corrupting
        a position only with entities/relations that appear in that position in the data, the corruptions are more
        realistic and the task becomes more challenging.

    :return: A tuple `(s, p ,o), labels` with three long tensors (s, p, o). If one of these is the corruption target, its size is (b, n + 1), containing
    the true value at index (i, 0) and the corruptions in the remaining positions of the second dimension (where `b` is
    the batch size and `n` is the number of negatives). If it's not the corruption target, its size is (b, 1), containing
    only the true value; this is expanded to (b, n) by broadcasting when fed to the model.
    The tensor `labels` indicates the classification targets.
    """
    for ctarget in [0, 1, 2]:  # Corruption target. Which part of the triple to corrupt.

        ng = negative_rates[ctarget]

        if ng > 0:

            with torch.no_grad():
                bs, _ = positives.size()

                if ccandidates is not None:
                    # Select corruptions from the targets that are allowed for thsi position
                    cand = ccandidates[ctarget]
                    mx = cand.size(0)
                    idx = torch.empty(bs, ng, dtype=torch.long, device=d()).random_(0, mx)
                    corruptions = cand[idx]
                else:
                    mx = r if ctarget == 1 else n
                    corruptions = torch.empty(bs, ng, dtype=torch.long, device=d()).random_(0, mx)

                s, p, o = positives[:, 0:1], positives[:, 1:2], positives[:, 2:3]
                if ctarget == 0:
                    s = torch.cat([s, corruptions], dim=1)
                if ctarget == 1:
                    p = torch.cat([p, corruptions], dim=1)
                if ctarget == 2:
                    o = torch.cat([o, corruptions], dim=1)

                # -- NB: two of the index vectors s, p o are now size (bs, 1) and the other is (bs, ng+1)
                #    We will let the model broadcast these to give us a score tensor of (bs, ng+1)
                #    In most cases we can optimize the decoder to broadcast late for better speed.

                if loss == 'bce':
                    labels = torch.cat([torch.ones(bs, 1, device=d()), torch.zeros(bs, ng, device=d())], dim=1)
                    # -- BCE loss treats the problem as a two-class classifcation problem. For each triple
                    #    the task is to predict whether it's true or false (corrupted) independent of the other triples.
                elif loss == 'ce':
                    labels = torch.zeros(bs, dtype=torch.long, device=d())
                    # -- CE loss treats the problem as a multiclass classification problem: for a positive triple,
                    #    together with its k corruptions, identify which is the true triple. This is always triple 0.
                    #    (It may seem like the model could easily cheat by always choosing triple 0, but the score
                    #    function doesn't see the order, so it can't choose by ordering.)

            yield (s, p, o), labels