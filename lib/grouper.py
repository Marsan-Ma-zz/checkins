import os, sys, time, pickle
import pandas as pd
import numpy as np
import multiprocessing as mp
pool_size = mp.cpu_count()

from datetime import datetime
from collections import Counter

from sklearn import cluster

from lib import conventions as conv
from lib import bhtsne

#===========================================
#   Evaluator
#===========================================  
class grouper(object):

  def __init__(self, params):
    self.size     = params['size']
    self.x_ranges = conv.get_range(params['size'], params['x_step'], params['x_inter'])
    self.y_ranges = conv.get_range(params['size'], params['y_step'], params['y_inter'])


  def add_grp(self, df_train, df_valid, df_test):
    print("[add_grp] start @ %s" % datetime.now())
    df_train['grp'] = 'tr'
    df_valid['grp'] = 'va'
    df_test['grp'] = 'te'
    df_all = pd.concat([df_train, df_valid, df_test])
    df_result = []
    processes = []
    for x_idx, (x_min, x_max) in enumerate(self.x_ranges):
      x_min, x_max = conv.trim_range(x_min, x_max, self.size)
      df_row = df_all[(df_all.x >= x_min) & (df_all.x < x_max)]
      mp_pool = mp.Pool(pool_size)
      for y_idx, (y_min, y_max) in enumerate(self.y_ranges): 
        y_min, y_max = conv.trim_range(y_min, y_max, self.size)
        df_grid = df_row[(df_row.y >= y_min) & (df_row.y < y_max)]
        if len(df_grid) == 0: continue

        p = mp_pool.apply_async(do_grouping, (df_grid,))
        processes.append([x_idx, y_idx, p])
      
      # prevent memory explode
      while len(processes) > 30:
        x_idx, y_idx, p = processes.pop(0)
        df_result.append(p.get())
        print("[add_grp] done for (x_idx, y_idx)=(%i,%i) @ %s" % (x_idx, y_idx, datetime.now()))

    # collect rest processes
    while processes:
      x_idx, y_idx, p = processes.pop(0)
      df_result.append(p.get())
      print("[add_grp] done for (x_idx, y_idx)=(%i,%i) @ %s" % (x_idx, y_idx, datetime.now()))

    df_all = pd.concat(df_result)
    print("len(df_train)=%i, len(df_valid)=%i, len(df_test)=%i, len(df_all)=%i" % (len(df_train), len(df_valid), len(df_test), len(df_all)))
    df_train = df_train.merge(df_all[df_all.grp == 'tr'][['row_id', 'tsne_x', 'tsne_y', 'kmeans']], on='row_id', how='left')
    df_valid = df_valid.merge(df_all[df_all.grp == 'va'][['row_id', 'tsne_x', 'tsne_y', 'kmeans']], on='row_id', how='left')
    df_test = df_test.merge(df_all[df_all.grp == 'te'][['row_id', 'tsne_x', 'tsne_y', 'kmeans']], on='row_id', how='left')
    df_train.drop('grp', axis=1, inplace=True)
    df_valid.drop('grp', axis=1, inplace=True)
    df_test.drop('grp', axis=1, inplace=True)
    print("[add_grp] done @ %s" % datetime.now())
    return df_train, df_valid, df_test



#===========================================
#   Multi Tasks
#===========================================  
WEIGHT = {
    'x': 500, 'y':1000, 
    'hour':4, 'logacc':1, 'weekday':3, 
    'qday':1, 'month':2, 'year':10, #'day':1./22,
}
def do_grouping(df_grid):
  perplexity = 2*len(set(df_grid.place_id.values))
  for k, v in WEIGHT.items(): df_grid[k] = df_grid[k]*v
  X = df_grid[list(WEIGHT.keys())].values
  res = bhtsne.bh_tsne(X, perplexity=150) #, verbose=True)
  res = pd.DataFrame(res, columns=['x', 'y'])

  k_means = cluster.KMeans(n_clusters=perplexity)
  k_means.fit(X)
  
  df_grid.is_copy = False
  df_grid['tsne_x'] = res['x'].values
  df_grid['tsne_y'] = res['y'].values
  df_grid['kmeans'] = k_means.labels_
  return df_grid

