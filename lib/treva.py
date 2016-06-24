import os, sys, time, pickle, gzip
import pandas as pd
import numpy as np
import multiprocessing as mp
pool_size = mp.cpu_count()

import xgboost as xgb

from datetime import datetime
from collections import OrderedDict, defaultdict

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
    self.submit_file = "%s/submit/treva_submit_%s.csv" % (self.root, self.stamp)

    # global variable for multi-thread
    global LOCATION, AVAIL_WDAYS, AVAIL_HOURS, POPULAR, GRID_CANDS
    if os.path.exists(self.data_cache):
      LOCATION, AVAIL_WDAYS, AVAIL_HOURS, POPULAR, GRID_CANDS = pickle.load(open(self.data_cache, 'rb'))
    

  #----------------------------------------
  #   Main
  #----------------------------------------
  def train(self, df_train, df_valid, df_test):
    mdl_path = "%s/models/%s" % (self.root, self.stamp)
    os.mkdir(mdl_path)
    print("[Train] start @ %s" % (conv.now('full')))
    
    processes = []
    preds_total = []
    score_stat = []
    for x_idx, (x_min, x_max) in enumerate(self.x_ranges):
      x_min, x_max = conv.trim_range(x_min, x_max, self.size)
      df_row = {
        'tr': df_train[(df_train.x >= x_min) & (df_train.x < x_max)],
        'va': df_valid[(df_valid.x >= x_min) & (df_valid.x < x_max)],
        'te': df_test[(df_test.x >= x_min) & (df_test.x < x_max)],
      }
      mp_pool = mp.Pool(pool_size)
      for y_idx, (y_min, y_max) in enumerate(self.y_ranges): 
        y_min, y_max = conv.trim_range(y_min, y_max, self.size)
        df_grid = {m: df_row[m][(df_row[m].y >= y_min) & (df_row[m].y < y_max)] for m in ['tr', 'va', 'te']}

        p = mp_pool.apply_async(drill_grid, (df_grid, self.x_cols, "(%i,%i)" % (x_idx, y_idx)))
        processes.append(p)

        # prevent memory explode!
        while (len(processes) > 10): 
          score, y_test = processes.pop(0).get()
          score_stat.append((score, len(df_grid)))
          preds_total.append(y_test)
      mp_pool.close()
    while processes: 
      score, y_test = processes.pop(0).get()
      score_stat.append((score, len(df_grid)))
      preds_total.append(y_test)
    # write submit file
    preds_total = pd.concat(preds_total)
    preds_total['place_id'] = preds_total[[0,1,2]].astype(str).apply(lambda x: ' '.join(x), axis=1)
    preds_total[['row_id', 'place_id']].sort_values(by='row_id').to_csv(self.submit_file, index=False)
    # collect scores
    valid_score = sum([s*c for s,c in score_stat]) / sum([c for s,c in score_stat])
    print("[Treva] done, valid_score=%.4f, submit file written %s @ %s" % (valid_score, self.submit_file, conv.now('full')))
    
      

#----------------------------------------
#   Drill Grid
#----------------------------------------
KNN_NORM = {
  'x': 500, 'y':1000, 
  'hour':4, 'logacc':1, 'weekday':3, 
  'qday':1, 'month':2, 'year':10, 'day':1./22,
}

def drill_grid(df_grid, x_cols, title=None, do_blending=True):
  best_score = 0
  best_model = None
  Xs, ys = {}, {}
  for m in ['tr', 'va', 'te']:
    Xs[m], ys[m], row_id = conv.df2sample(df_grid[m], x_cols)
  
  # grid search best models
  all_bests = []
  all_test_bests = []
  mdl_configs = []
  for alg in ['skrf', 'skrfp', 'sket', 'sketp']:
    for n_estimators in [500, 1000, 1500]:
      for max_features in [0.3, 0.4, 0.5]:
        for max_depth in [11, 13, 15]:
          mdl_configs.append({'alg': alg, 'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth}) 

  for mdl_config in mdl_configs:
    # train
    clf = get_alg(mdl_config['alg'], mdl_config)
    clf.fit(Xs['tr'], ys['tr'])
    # valid
    score, bests = drill_eva(clf, Xs['va'], ys['va'])
    print("drill%s va_score %.4f for model %s(%s) @ %s" % (title, score, mdl_config['alg'], mdl_config, conv.now('full')))
    if score > best_score:
      best_score = score
      best_model = clf
      best_config = mdl_config
    # for blending
    if do_blending:
      all_bests.append(bests)
      _, test_bests = drill_eva(clf, Xs['te'], ys['te'])
      all_test_bests.append(test_bests)
  

  # # knn (can't apply to too small grid!)
  # clf = get_alg('knn', {})
  # knn_norm = {k:v for k,v in KNN_NORM.items() if k in x_cols}
  # for m in ['tr', 'va', 'te']:
  #   df_grid_knn = df_grid[m]
  #   for k, v in knn_norm.items(): df_grid_knn[k] = df_grid_knn[k]*v
  #   Xs['knn_'+m], ys['knn_'+m], row_id = conv.df2sample(df_grid_knn, x_cols)
  # clf.fit(Xs['knn_tr'], ys['knn_tr'])
  # score, bests = drill_eva(clf, Xs['knn_va'], ys['knn_va'])
  # print("drill%s va_score %.4f for model 'knn' @ %s" % (title, score, conv.now('full')))
  # if score > best_score:
  #   best_score = score
  #   best_model = clf
  #   best_config = {}
  # # for blending
  # if do_blending:
  #   all_bests.append(bests)
  #   _, test_bests = drill_eva(clf, Xs['knn_te'], ys['knn_te'])
  #   all_test_bests.append(test_bests)


  # blending
  if do_blending:
    blended_bests = blending(all_bests)
    blended_match = [apk([ans], vals) for ans, vals in zip(ys['tr'], blended_bests)]
    blended_score = sum(blended_match)/len(blended_match)
    print("drill%s va_score %.4f for model 'blending' @ %s" % (title, score, conv.now('full')))
  
  # collect results
  if do_blending and (blended_score > best_score):
    best_score = blended_score
    best_model = None
    test_preds = blending(all_test_bests)
    print("[best] model is blending, for %s" % title)
  else:
    _, test_preds = drill_eva(best_model, Xs['te'], ys['te'])
    print("[best] model is %s, for %s" % (best_config, title))
  
  test_preds = pd.DataFrame(test_preds)
  test_preds['row_id'] = row_id
  print("[drill_grid %s] done with best_score=%.4f @ %s" % (title, best_score, datetime.now()))
  return best_score, test_preds


def blending(all_bests, rank_w=[1, 0.6, 0.4]):
  blended_bests = []
  for i in range(len(all_bests[0])):
    stat = defaultdict(float)
    for line in [m[i] for m in all_bests]:
      for c, s in enumerate(line):
        stat[s] += rank_w[c]
    stat = sorted(stat.items(), key=lambda v: v[1], reverse=True)
    stat = [pid for pid,val in stat][:3]
    blended_bests.append(stat)
  return blended_bests


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
    # -----[filter avail places]-----
    s = X.iloc[i]
    psol = {p: v * (
        0.1 * (AVAIL_WDAYS.get((p, s.weekday.astype(int)), 0) > time_th_wd) + 
        0.4 * (AVAIL_HOURS.get((p, s.hour.astype(int)), 0) > time_th_hr)
    ) for p,v in psol.items()}
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


