import os, sys, time, pickle, gzip
import pandas as pd
import numpy as np
import multiprocessing as mp
pool_size = mp.cpu_count()

import xgboost as xgb

from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, ensemble

from lib import conventions as conv

#===========================================
#   Trainer
#===========================================  
class trainer(object):

  def __init__(self, params):
    self.root     = params['root']
    self.stamp    = params['stamp']
    self.size     = params['size']
    self.x_cols   = params['x_cols']
    self.x_ranges = conv.get_range(params['size'], params['x_step'], params['x_inter'])
    self.y_ranges = conv.get_range(params['size'], params['y_step'], params['y_inter'])
    # extra_info
    self.data_cache = params['data_cache']
    self.loc_th_x = params['loc_th_x']
    self.loc_th_y = params['loc_th_y']
    self.en_preprocessing = params['en_preprocessing']

    # global variable for multi-thread
    global LOCATION, AVAIL_WDAYS, AVAIL_HOURS, POPULAR, GRID_CANDS
    if os.path.exists(self.data_cache):
      LOCATION, AVAIL_WDAYS, AVAIL_HOURS, POPULAR, GRID_CANDS = pickle.load(open(self.data_cache, 'rb'))
      LOCATION['x_min'] = LOCATION.x_mean - self.loc_th_x*LOCATION.x_std
      LOCATION['x_max'] = LOCATION.x_mean + self.loc_th_x*LOCATION.x_std
      LOCATION['y_min'] = LOCATION.y_mean - self.loc_th_y*LOCATION.y_std
      LOCATION['y_max'] = LOCATION.y_mean + self.loc_th_y*LOCATION.y_std
    

  #----------------------------------------
  #   Main
  #----------------------------------------
  def train(self, df_train, alg="skrf", mdl_config={}, norm=None):
    mdl_path = "%s/models/%s" % (self.root, self.stamp)
    os.mkdir(mdl_path)
    print("[Train] start with mdl_config=%s, write models to %s @ %s" % (mdl_config, mdl_path, conv.now('full')))
    
    processes = []
    for x_idx, (x_min, x_max) in enumerate(self.x_ranges):
      x_min, x_max = conv.trim_range(x_min, x_max, self.size)
      df_row = df_train[(df_train.x >= x_min) & (df_train.x < x_max)]
      mp_pool = mp.Pool(pool_size)
      for y_idx, (y_min, y_max) in enumerate(self.y_ranges): 
        y_min, y_max = conv.trim_range(y_min, y_max, self.size)
        df_grid = df_row[(df_row.y >= y_min) & (df_row.y < y_max)]

        # normalize for scale sensitive algorithms
        if norm:  
          for k, v in norm.items(): df_grid[k] = df_grid[k]*v

        # preprocessing
        if self.en_preprocessing:
          df_grid, cols_extra = conv.df_preprocess(self.en_preprocessing, df_grid, x_idx, y_idx, LOCATION, AVAIL_WDAYS, AVAIL_HOURS, POPULAR, GRID_CANDS)
          X_train, y_train, _ = conv.df2sample(df_grid, self.x_cols+cols_extra)
        else:
          X_train, y_train, _ = conv.df2sample(df_grid, self.x_cols)

        # save model (can't stay in memory, too large)
        clf = self.get_alg(alg, mdl_config)
        mdl_name = "%s/models/%s/grid_model_x_%s_y_%s.pkl.gz" % (self.root, self.stamp, x_idx, y_idx)
        p = mp_pool.apply_async(save_model, (alg, mdl_name, clf, X_train, y_train))
        processes.append(p)
        clf = None  # clear memory
        # prevent memory explode!
        while (len(processes) > 30): processes.pop(0).get()
      print("[Train] grid(%i,%i): %i samples / %i classes @ %s" % (x_idx, y_idx, len(y_train), len(set(y_train)), conv.now('full')))
      mp_pool.close()
    while processes: processes.pop(0).get()
    print("[Train] done @ %s" % conv.now('full'))
    
      

  #----------------------------------------
  #   Tasks
  #----------------------------------------
  def get_alg(self, alg, mdl_config):
    if alg == 'skrf':
      clf = ensemble.RandomForestClassifier(
        n_estimators=mdl_config.get('n_estimators', 300), 
        max_features=mdl_config.get('max_features', 'auto'),  
        max_depth=mdl_config.get('max_depth', 11), 
        n_jobs=-1
      )
    elif alg == 'skrfp':
      clf = ensemble.RandomForestClassifier(
        n_estimators=mdl_config.get('n_estimators', 300),
        max_features=mdl_config.get('max_features', 'auto'),   
        max_depth=mdl_config.get('max_depth', 11), 
        criterion='entropy', 
        n_jobs=-1
      )
    elif alg =='sket':
      clf = ensemble.ExtraTreesClassifier(
        n_estimators=mdl_config.get('n_estimators', 800), 
        max_features=mdl_config.get('max_features', 'auto'),  
        max_depth=mdl_config.get('max_depth', 15), 
        n_jobs=-1
      )
    elif alg =='sketp':
      clf = ensemble.ExtraTreesClassifier(
        n_estimators=mdl_config.get('n_estimators', 800), 
        max_features=mdl_config.get('max_features', 'auto'),  
        max_depth=mdl_config.get('max_depth', 15), 
        criterion='entropy', 
        n_jobs=-1
      )
    elif alg == 'skgbc':
      clf = ensemble.GradientBoostingClassifier(
        n_estimators=mdl_config.get('n_estimators', 30), 
        max_depth=mdl_config.get('max_depth', 5)
      )
    elif alg == 'xgb':
      # https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py
      clf = xgb.XGBClassifier(
        n_estimators=mdl_config.get('n_estimators', 30), 
        max_depth=mdl_config.get('max_depth', 7), 
        learning_rate=mdl_config.get('learning_rate', 0.1), 
        objective="multi:softprob", 
        silent=True
      )
    elif alg == 'knn':
      clf = KNeighborsClassifier(n_neighbors=25, weights='distance', metric='manhattan', n_jobs=-1)
    elif alg == 'sklr':
      clf = linear_model.LogisticRegression(multi_class='multinomial', solver = 'lbfgs')
    return clf
  

def save_model(alg, mdl_name, clf, X_train, y_train):
  if alg == 'knn':
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    X_train = X_train.values.astype(int)
    clf.fit(X_train, y_train)
    pickle.dump([clf, le], gzip.open(mdl_name, 'wb'))
  else:
    clf.fit(X_train, y_train)
    pickle.dump(clf, gzip.open(mdl_name, 'wb'))
