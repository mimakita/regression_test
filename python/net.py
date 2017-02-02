#!/usr/bin/env python

import chainer
from chainer import cuda, Function, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class MyChain(Chain):
  def __init__(self, n_in, n_unit):
    super(MyChain, self).__init__(
      l1 = L.Linear(n_in, n_unit),
      l2 = L.Linear(n_unit,n_unit),
      l3 = L.Linear(n_unit,n_unit),
      l4 = L.Linear(n_unit,1)
      )

  def forward(self, x):
    h1 = F.sigmoid(self.l1(x))
    h2 = F.sigmoid( self.l2(h1))
    h3 = F.sigmoid( self.l2(h2))
    out = self.l4(h3)
    return out
  

