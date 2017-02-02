#!/usr/bin/env python

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss.mean_squared_error import mean_squared_error as mse

import dataloader as dl
import net

def evaluate(results, sv=None):
  c = 0
  if sv != None:
    for (Y,el) in zip(sv,list(results)):
      seikai = int(Y[0])
      result = int(np.round(el.data[0]))
      if seikai == result:
        c += 1
      print(seikai,result)
  
    prec = c*1.0/n_sample*1.0
    print(prec)
  else:
    for result in results: 
      d = result.data[0]
      d_int = int(np.round(d))
      print(d_int,d)

fp_te = "../data/wq_white_test.csv"
X_test,Y_test = dl.loadCSVdata(fp_te,11,delim=";")
X_test = dl.stddata(X_test)

n_units  = 1024 
n_sample = X_test.shape[0]
n_cols   = X_test.shape[1]
model    = net.MyChain(n_cols,n_units)
serializers.load_npz("model.bin.tmp",model)

opt = optimizers.Adam()
opt.setup(model)

loss_val = 1000
epoch    = 0
x = chainer.Variable(X_test.reshape(n_sample, n_cols))
#if len(Y_test) > 0:
#  t = chainer.Variable(Y_test.reshape(n_sample, 1))
model.zerograds()
y = model.forward(x)

if len(Y_test) > 0:
  evaluate(y,sv=Y_test)
else:
  evaluate(y)

  
