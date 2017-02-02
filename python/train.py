#!/usr/bin/env python

import sys
import re
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss.mean_squared_error import mean_squared_error as mse

import dataloader as dl
import net

xp = cuda.cupy

fp_tr = "../data/wq_white_train.csv"
X,Y = dl.loadCSVdata(fp_tr,11,delim=";")
X_st = dl.stddata(X)

n_units  = 1024
n_sample = X.shape[0]
n_cols = X.shape[1]
model = net.MyChain(n_cols, n_units)

#dev_id = 3
#chainer.cuda.get_device(dev_id).use()
#model.to_gpu()
opt = optimizers.Adam()
opt.setup(model)

loss_val = 10000
epoch    = 0
while loss_val > 0.001:
  x = chainer.Variable(X_st.reshape(n_sample, n_cols))
  t = chainer.Variable(Y.reshape(n_sample, 1))
  model.zerograds()
  y = model.forward(x)
  loss = mse(y,t)
  loss.backward()
  opt.update()

  if epoch % 25 == 0:
    loss_val = loss.data
    print('epoch:',epoch)
    print('train mean loss=', loss_val)
    print('- - - - - - - - -')
    serializers.save_npz('model.bin.tmp', model) 
  if epoch >= 30000:
    break
  epoch += 1 

serializers.save_npz('model.bin', model) 
