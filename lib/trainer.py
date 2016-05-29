import os, sys, datetime, time, pickle
import pandas as pd
import numpy as np
import multiprocessing as mp
pool_size = mp.cpu_count()

import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn import linear_model, ensemble

from lib import conventions as conv

#===========================================
#   Trainer
#===========================================  
class trainer(object):

  def __init__(self, stamp, size, root, x_ranges, y_ranges):
    self.stamp = stamp
    self.size = size
    self.root = root
    self.x_ranges = x_ranges
    self.y_ranges = y_ranges

  #----------------------------------------
  #   Main
  #----------------------------------------
  def train(self, df_train, alg="skrf"):
    os.mkdir("%s/models/%s" % (self.root, self.stamp))
    for x_idx, (x_min, x_max) in enumerate(self.x_ranges):
      start_time_row = time.time()
      for y_idx, (y_min, y_max) in enumerate(self.y_ranges): 
        start_time_cell = time.time()
        x_min, x_max = conv.trim_range(x_min, x_max, self.size)
        y_min, y_max = conv.trim_range(y_min, y_max, self.size)
        X_train, y_train, _ = conv.df2sample(df_train, x_min, x_max, y_min, y_max)
        
        clf = self.get_alg(alg)
        clf.fit(X_train, y_train)

        # save model (can't stay in memory, too large)
        mdl_name = "%s/models/%s/grid_model_x_%s_y_%s.pkl" % (self.root, self.stamp, x_idx, y_idx)
        pickle.dump(clf, open(mdl_name, 'wb'))
        clf = None  # clear memory
        print(("[Train] grid(%i,%i): %i samples for %i classes, %.2f secs" % (x_idx, y_idx, len(y_train), len(set(y_train)), time.time() - start_time_cell)))
      print(("[Train] row %i elapsed time: %.2f secs" % (x_idx, time.time() - start_time_row)))


  #----------------------------------------
  #   Tasks
  #----------------------------------------
  def get_alg(self, alg):
    if alg == 'skrf':
      clf = ensemble.RandomForestClassifier(n_estimators=30, max_depth=5, n_jobs=-1)
    elif alg == 'skgbc':
      clf = ensemble.GradientBoostingClassifier(n_estimators=30, max_depth=5)
    elif alg == 'sklr':
      clf = linear_model.LogisticRegression(multi_class='multinomial', solver = 'lbfgs')
    elif alg == 'xgb':
      clf = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=30, silent=True)
    return clf
  