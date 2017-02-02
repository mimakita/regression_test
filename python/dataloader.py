#!/usr/bin/env python

import sys,re
import csv
import numpy as np

def loadCSVdata(fp,label_idx=-1,delim=","):
  with open(fp,"r") as f:
    reader = csv.reader(f, delimiter=delim)
    next(reader)
    x = []
    y = []
    label = ""
    for row in reader:
      colvec = []
      for i,r in enumerate(row):
        if ( label_idx > 0 ) and ( i == label_idx ) : 
          try:
            label = int(r) 
          except:
            label = float(r) 
        else:
          if r == 'W':
            r = -1
          elif r == 'R':
            r = 1
          colvec.append(float(r))
      x.append(colvec) 

      if label_idx > 0:
        y.append([label])

  x = np.array(x,dtype=np.float32) 
  y = np.array(y,dtype=np.float32)

  return np.array(x,dtype=np.float32),np.array(y,dtype=np.float32)

def stddata(x):
  n_samples = x.shape[0] 
  n_cols    = x.shape[1]

  row_sum = np.sum(x, axis=0)
  row_avg = row_sum/n_samples 
  row_max = np.max(x, axis=0)
  row_min = np.min(x, axis=0)
   
  new_x = ( x - row_avg ) / ( row_max - row_min ) 

  # except for 'wine color' 
  for i,el in enumerate(new_x):
    new_x[i][0] = x[i][0]

  return new_x

if __name__ == "__main__":
  fp = "/home/m-makita/data/misc/wine-quality2/wine_train.csv"
  x,y = loadCSVdata(fp)
  x_st = stddata(x)

  for x_r in x_st:
    print(list(x_r))
