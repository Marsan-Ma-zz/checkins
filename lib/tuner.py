import os, time, pickle, gzip
import pandas as pd
import numpy as np

import multiprocessing as mp
pool_size = mp.cpu_count()

from sklearn import linear_model, ensemble
from datetime import datetime
from collections import Counter, OrderedDict, defaultdict


from lib import conventions as conv
from lib import parser


#===========================================
#   Tasks
#===========================================
def tree_cv(clf, df_grid, debug=False):
  scores = []
  x_cols = ['hour', 'qday', 'weekday', 'month', 'year', 'logacc', 'x', 'y']
  for ts in np.arange(100000, 700000, 100000):
    df_tr = df_grid[df_grid.time <= ts]
    df_va = df_grid[(df_grid.time > ts) & (df_grid.time < ts + 100000)]
    if debug:
        df_tr = df_tr[:100]
        df_va = df_va[:100]
    X_tr, y_tr, _ = conv.df2sample(df_tr, x_cols)
    X_va, y_va, _ = conv.df2sample(df_va, x_cols)
    clf.fit(X_tr, y_tr)
    sols = clf.predict_proba(X_va)
    score, _ = drill_eva(clf, X_va, y_va)
    # print(ts, score)
    scores.append(score)
  avg_score = np.mean(scores)
  return avg_score


def grid_search_tree_params(df_grid):
  mp_pool = mp.Pool(pool_size)
  processes = []
  for alg in ['skrf', 'skrfp', 'sket', 'sketp']:
    for n_estimators in np.arange(300, 2100, 200):
      for max_features in [0.3, 0.4, 0.5]:
        for max_depth in np.arange(7, 15, 2):
          mcfg = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth}
          clf = get_alg(alg, mcfg)
          p = mp_pool.apply_async(tree_cv, (clf, df_grid))
          processes.append([alg, n_estimators, max_features, max_depth, p])
  
  all_scores = []
  while processes:
    alg, n_est, m_feat, m_dep, p = processes.pop(0)
    score = p.get()
    result = (alg, n_est, m_feat, m_dep, score)
    all_scores.append(result)
  return all_scores

def get_data(cache_name):
  df_train, df_valid, _ = pickle.load(open(cache_name, 'rb'))
  df = pd.concat([df_train, df_valid])
  return df


def tune_grid(df, x_min, x_max, y_min, y_max):
  df_grid = df[(df.x >= x_min) & (df.x < x_max) & (df.y >= y_min) & (df.y < y_max)]
  all_scores = grid_search_tree_params(df_grid)
  return all_scores


def apk(actual, predicted, k=3):
  if len(predicted) > k: 
    predicted = predicted[:k]
  score, num_hits = 0.0, 0.0
  for i,p in enumerate(predicted):
    if p in actual and p not in predicted[:i]:
      num_hits += 1.0
      score += num_hits / (i+1.0)
  if not actual: return 0.0
  return score / min(len(actual), k)


def drill_eva(clf, X, y, time_th_wd=0.003, time_th_hr=0.004):
  final_bests = []
  sols = clf.predict_proba(X)
  sols = [[(clf.classes_[i], v) for i, v in enumerate(line)] for line in sols]
  for i in range(len(X)):
    psol = OrderedDict(sorted(sols[i], key=lambda v: v[1]))
    psol = sorted(list(psol.items()), key=lambda v: v[1], reverse=True)
    psol = [p for p,v in psol]
    final_bests.append(psol[:3])
  if y is not None:
    match = [apk([ans], vals) for ans, vals in zip(y, final_bests)]
    score = sum(match)/len(match)
  else: 
    score = None
  return score, final_bests


def get_alg(alg, mdl_config):
  if alg == 'skrf':
    clf = ensemble.RandomForestClassifier(
      n_estimators=mdl_config.get('n_estimators', 500), 
      max_features=mdl_config.get('max_features', 0.35),  
      max_depth=mdl_config.get('max_depth', 15), 
      n_jobs=-1,
    )
  elif alg == 'skrfp':
    clf = ensemble.RandomForestClassifier(
      n_estimators=mdl_config.get('n_estimators', 500), 
      max_features=mdl_config.get('max_features', 0.35),
      max_depth=mdl_config.get('max_depth', 15), 
      criterion='entropy',
      n_jobs=-1,
    )
  elif alg =='sket':
    clf = ensemble.ExtraTreesClassifier(
      n_estimators=mdl_config.get('n_estimators', 500), 
      max_features=mdl_config.get('max_features', 0.5),  
      max_depth=mdl_config.get('max_depth', 15), 
      n_jobs=-1,
    )
  elif alg =='sketp':
    clf = ensemble.ExtraTreesClassifier(
      n_estimators=mdl_config.get('n_estimators', 500), 
      max_features=mdl_config.get('max_features', 0.5),  
      max_depth=mdl_config.get('max_depth', 11), 
      criterion='entropy', 
      n_jobs=-1,
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

#===========================================
#   Main Flow
#===========================================
def tuner(df, grids):
  print("[Tuner] start @ %s" % datetime.now())
  for x_min, x_max, y_min, y_max in grids:
    all_scores = tune_grid(df, x_min, x_max, y_min, y_max)
    print("%sgrid[%2f,%.2f,%.2f, %.2f]%s @ %s" % ('='*10, x_min, x_max, y_min, y_max, '='*20, datetime.now()))
    for s in sorted(all_scores, key=lambda v: v[4], reverse=True): 
      print(s)
  print("[Tuner] done @ %s" % datetime.now())
  return all_scores


if __name__ == '__main__':
  cache_name = "./data/cache/cache_get_data_split_10000000000_size_10.00_rmol_1.00_mci_3.pkl"
  df = get_data(cache_name)
  grids = [
    (0, 0.08, 0.88, 0.96),
  ]
  tuner(df, grids)
  