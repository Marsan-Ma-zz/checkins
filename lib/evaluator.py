import os, sys, datetime, time, pickle, gzip, shutil
import pandas as pd
import numpy as np
import multiprocessing as mp
pool_size = mp.cpu_count()

from collections import OrderedDict

from lib import conventions as conv

LOCATION, AVAIL_WDAYS, AVAIL_HOURS = pickle.load(open("./data/cache/location_est.pkl", 'rb'))


#===========================================
#   Evaluator
#===========================================  
class evaluator(object):

  def __init__(self, params):
    self.root     = params['root']
    self.stamp    = params['stamp']
    self.size     = params['size']
    self.x_cols   = params['x_cols']
    self.x_ranges = conv.get_range(params['size'], params['x_step'])
    self.y_ranges = conv.get_range(params['size'], params['y_step'])
    self.mdl_weights = params['mdl_weights']
    # extra_info
    self.location_cache = params['location_cache']
    self.time_th  = params['time_th']
    self.loc_th_x = params['loc_th_x']
    self.loc_th_y = params['loc_th_y']
    self.init_extra_info()


  def init_extra_info(self):
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

    # launch mp jobs
    processes = []
    for x_idx, (x_min, x_max) in enumerate(self.x_ranges):
      x_min, x_max = conv.trim_range(x_min, x_max, self.size)
      df_row = df[(df.x >= x_min) & (df.x < x_max)]
      mp_pool = mp.Pool(pool_size)
      for y_idx, (y_min, y_max) in enumerate(self.y_ranges):
        y_min, y_max = conv.trim_range(y_min, y_max, self.size)
        w_ary = [(x_idx-1, y_idx), (x_idx, y_idx), (x_idx+1, y_idx)]
        mdl_names = ["%s/models/%s/grid_model_x_%s_y_%s.pkl.gz" % (self.root, self.stamp, xi, yi) for xi, yi in w_ary]
        X, y, row_id = conv.df2sample(df_row, None, None, y_min, y_max, self.x_cols)
        p = mp_pool.apply_async(predict_clf, (mdl_names, self.mdl_weights, X, y, row_id, self.time_th))
        processes.append([x_idx, y_idx, p])
      mp_pool.close()
      print("[%s] launching row %i processes @ %s ..." % (title, x_idx, conv.now('full')))
    

    # collect mp results
    row_scores, row_samples = [], 0
    for xi, yi, p in processes:
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
        row_scores, row_samples = [], 0
    processes = []

    # final stats
    if title == 'Submit':
      final_score = 'none'
      print("=====[Done submittion data]=====")
    else:
      scores, cnts = zip(*[(s*len(p), len(p)) for s, p in zip(score_total, preds_total)])
      final_score = sum(scores) / sum(cnts)
      print("=====[%s score] MAP=%.4f =====" % (title, final_score))
    preds_total = pd.concat(preds_total)
    return preds_total, final_score


  #----------------------------------------
  #   Tasks
  #----------------------------------------
  def gen_submit_file(self, preds_total, score):
    preds_total['place_id'] = [" ".join([str(k) for k in l]) for l in preds_total[[0,1,2]].values.tolist()]
    preds_total[['row_id', 'place_id']].sort_values(by='row_id').to_csv("%s/submit/submit_%s_%.4f.csv" % (self.root, self.stamp, score), index=False)
    return  


  # clear meta files
  def clear_meta_files(self):
    mdl_path = "%s/models/%s" % (self.root, self.stamp)
    shutil.rmtree(mdl_path)


#----------------------------------------
#   Multi-Thread tasks
#----------------------------------------
def predict_clf(mdl_names, mdl_weights, X, y, row_id, time_th=0.001, batch=10000):
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
    match = [apk([ans], vals) for ans, vals in zip(y, preds[[0,1,2]].values)]
    return sum(match)/len(match)
  def merge_bests(clfs, weights, samples, time_th):
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
      psol = [p for p in psol if (p in avail_place) and (AVAIL_WDAYS.get((p, s.weekday), 0) > time_th) and (AVAIL_HOURS.get((p, s.hour), 0) > time_th)]
      # -------------------------------
      final_bests.append(psol[:3])
    return final_bests
  #
  clfs, weights = zip(*[(pickle.load(gzip.open(mname, 'rb')), w) for mname, w in zip(mdl_names, mdl_weights) if os.path.exists(mname)])
  # print(weights)
  # all_class = [el for el in clf.classes_]
  preds = []
  for ii in range(0, len(X), batch):
    samples = X[ii:ii+batch]
    if len(samples) > 0:
      preds += merge_bests(clfs, weights, samples, time_th)
  clfs = None
  preds = pd.DataFrame(preds)
  preds['row_id'] = row_id
  score = map_score(y, preds)
  return preds, score


