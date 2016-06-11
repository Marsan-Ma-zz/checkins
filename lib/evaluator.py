import os, sys, datetime, time, pickle, gzip, shutil
import pandas as pd
import numpy as np
import multiprocessing as mp
pool_size = mp.cpu_count()

from collections import OrderedDict

from lib import conventions as conv


#===========================================
#   Evaluator
#===========================================  
class evaluator(object):

  def __init__(self, params):
    self.root     = params['root']
    self.stamp    = params['stamp']
    self.size     = params['size']
    self.x_cols   = params['x_cols']
    self.x_inter  = params['x_inter']
    self.y_inter  = params['y_inter']
    self.x_ranges = conv.get_range(params['size'], params['x_step'], self.x_inter)
    self.y_ranges = conv.get_range(params['size'], params['y_step'], self.y_inter)
    self.mdl_weights = params['mdl_weights']
    # extra_info
    self.data_cache = params['data_cache']
    self.time_th_wd  = params['time_th_wd']
    self.time_th_hr  = params['time_th_hr']
    self.popu_th  = params['popu_th']
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
  def evaluate(self, df, title):
    preds_total = []
    score_total = []

    # check model exists
    if not os.path.exists("%s/models/%s" % (self.root, self.stamp)):
      print("[ERROR] evaluate: model %s does not exists!!" % mdl_path)
      return

    # launch mp jobs
    processes = []
    for x_idx, (x_min, x_max) in enumerate(self.x_ranges):
      if (x_idx % self.x_inter == 0):   # skip interleave blocks
        x_min, x_max = conv.trim_range(x_min, x_max, self.size)
        df_row = df[(df.x >= x_min) & (df.x < x_max)]
        row_scores, row_samples = [], 0
        mp_pool = mp.Pool(pool_size)
      for y_idx, (y_min, y_max) in enumerate(self.y_ranges):
        if (x_idx % self.x_inter == 0) and (y_idx % self.y_inter == 0): # skip interleave blocks
          y_min, y_max = conv.trim_range(y_min, y_max, self.size)
          w_ary = [(x_idx-1, y_idx), (x_idx, y_idx), (x_idx+1, y_idx)]
          mdl_names = ["%s/models/%s/grid_model_x_%s_y_%s.pkl.gz" % (self.root, self.stamp, xi, yi) for xi, yi in w_ary]
          df_grid = df_row[(df_row.y >= y_min) & (df_row.y < y_max)]
          
          # preprocessing
          if self.en_preprocessing:
            df_grid, cols_extra = conv.df_preprocess(self.en_preprocessing, df_grid, x_idx, y_idx, LOCATION, AVAIL_WDAYS, AVAIL_HOURS, POPULAR, GRID_CANDS)
            X, y, row_id = conv.df2sample(df_grid, self.x_cols+cols_extra)
          else:
            X, y, row_id = conv.df2sample(df_grid, self.x_cols)
          
          # fail-safe
          if len(X) == 0:
            print("0 samples in x_idx=%i, y_idx=%i, skip evaluation." % (x_idx, y_idx))
            continue

          # parallel evaluation
          p = mp_pool.apply_async(predict_clf, (mdl_names, self.mdl_weights, X, y, row_id, x_idx//self.x_inter, y_idx//self.y_inter, self.popu_th, self.time_th_wd, self.time_th_hr))
          processes.append([x_idx, y_idx, p])
        # collect mp results
        last_block = (x_idx >= len(self.x_ranges)-1) and (y_idx >= len(self.y_ranges)-1)  
        while (len(processes) > 20) or (processes and last_block):
          xi, yi, p = processes.pop(0)
          preds, score = p.get() #predict_clf(mdl_name, X, y, row_id)
          score_total.append(score)
          preds_total.append(preds)
          # observation
          row_scores.append(score)
          row_samples += len(preds)
          if (yi == 0) and (xi > 0):
            if title == 'Submit':
              print("[Submit] row %i, %i samples @ %s" % (xi-1, row_samples, conv.now('full')))
            else:
              print("[%s] row %i, avg MAP=%.4f, %i samples @ %s" % (title, xi-1, np.average(row_scores), row_samples, conv.now('full')))          
      mp_pool.close()
      row_scores, row_samples = [], 0
      print("[%s] launching row %i processes @ %s ..." % (title, x_idx, conv.now('full')))
    print("[Evaluation] done, rest processes=%i (should be 0!)" % len(processes))

    # final stats
    if title == 'Submit':
      final_score = 'none'
      print("=====[Done submittion data]=====")
    else:
      # print("preds_total", preds_total)
      # print("score_total", score_total)
      scores, cnts = zip(*[(s*len(p), len(p)) for s, p in zip(score_total, preds_total)])
      final_score = sum(scores) / sum(cnts)
      print("=====[%s score] MAP=%.4f =====" % (title, final_score))
    preds_total = pd.concat(preds_total)
    return preds_total, final_score


  #----------------------------------------
  #   Tasks
  #----------------------------------------
  def gen_submit_file(self, preds_total, score, fill_dummy=True):
    preds_total['place_id'] = [" ".join([str(k) for k in l]) for l in preds_total[[0,1,2]].values.tolist()]
    #-----[fill not submitted rows, for fast experiment]------
    if fill_dummy:
      dummy = 1111111111
      missing_rows = set(range(8607230)) - set(preds_total.row_id.values)
      df_dummies = pd.DataFrame([{'row_id': d, 'place_id': "%s %s %s" % (dummy, dummy, dummy)} for d in missing_rows])
      preds_total = pd.concat([preds_total, df_dummies])
    #---------------------------------------------------------
    preds_total[['row_id', 'place_id']].sort_values(by='row_id').to_csv("%s/submit/submit_%s_%.4f.csv" % (self.root, self.stamp, score), index=False)
    

  # clear meta files
  def clear_meta_files(self):
    mdl_path = "%s/models/%s" % (self.root, self.stamp)
    shutil.rmtree(mdl_path)


#----------------------------------------
#   Multi-Thread tasks
#----------------------------------------
def predict_clf(mdl_names, mdl_weights, X, y, row_id, xi, yi, popu_th, time_th_wd, time_th_hr, batch=10000):
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
  def map_score(y, preds):
    if y is None: return None
    try:
      match = [apk([ans], vals) for ans, vals in zip(y, preds[[0,1,2]].values)]
    except Exception as e:
      print("[Exception]:", e, preds)
    return sum(match)/len(match)
  def merge_bests(clfs, weights, samples):
    sols = [clf.predict_proba(samples) for clf in clfs]
    sols = [[[(clf.classes_[i], v*w) for i, v in enumerate(line)] for line in sol] for clf, w, sol in zip(clfs, weights, sols)]
    final_bests = []
    for i in range(len(samples)):
      psol = [j for k in [s[i] for s in sols] for j in k]
      psol = OrderedDict(sorted(psol, key=lambda v: v[1]))
      psol = sorted(list(psol.items()), key=lambda v: v[1], reverse=True)
      psol = [p for p,v in psol]
      # -----[filter avail places]-----
      s = samples.iloc[i]
      avail_place = LOCATION[(LOCATION.x_min <= s.x) & (LOCATION.x_max >= s.x) & (LOCATION.y_min <= s.y) & (LOCATION.y_max >= s.y)].place_id.values
      # psol = [p for p in psol if (p in avail_place) and 
      #   (AVAIL_WDAYS.get((p, s.weekday), 0) > time_th_wd) and 
      #   (AVAIL_HOURS.get((p, s.hour.astype(int)), 0) > time_th_hr) #and
      #   # (POPULAR[(xi, yi)].get(p, 0) > popu_th)
      # ]
      # -------------------------------
      final_bests.append(psol[:3])
    return final_bests
  #
  clfs, weights = zip(*[(pickle.load(gzip.open(mname, 'rb')), w) for mname, w in zip(mdl_names, mdl_weights) if os.path.exists(mname) and (w != 0)])
  preds = []
  for ii in range(0, len(X), batch):
    samples = X[ii:ii+batch]
    if len(samples) > 0:
      preds += merge_bests(clfs, weights, samples)
  clfs = None
  preds = pd.DataFrame(preds)
  preds['row_id'] = row_id
  score = map_score(y, preds)
  return preds, score


