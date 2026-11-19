"""TRIAD without the unseen-item bonus (as dials/paper/figdata/unseen_bonus.py): rho of every item with f = 0 is
replaced by rho(log 2), its value for an item seen once. wrap(load) returns a triad.train.load that applies it."""
import math
import torch


def wrap(load):
    def no_bonus_load(*args, **kwargs):
        m = load(*args, **kwargs);full = m.cache
        def cache():
            c = full();once = m.prior(torch.tensor([[math.log(2.)]])).squeeze()
            c['prior'] = torch.where(m.frequency == 0, once, c['prior']);return c
        m.cache = cache;return m
    return no_bonus_load
