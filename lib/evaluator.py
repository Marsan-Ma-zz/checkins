import os, sys, datetime, time, pickle, gzip, shutil
import pandas as pd
import numpy as np
import multiprocessing as mp
pool_size = mp.cpu_count()


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
    self.x_ranges = conv.get_range(params['size'], params['step'])
    self.y_ranges = conv.get_range(params['size'], params['step'])
    
  #----------------------------------------
  #   Main
  #----------------------------------------
  def evaluate(self, df, title):
    preds_total = []
    score_total = []

    # launch mp jobs
    processes = []
    mp_pool = mp.Pool(pool_size)
    for x_idx, (x_min, x_max) in enumerate(self.x_ranges):
      x_min, x_max = conv.trim_range(x_min, x_max, self.size)
      df_row = df[(df.x >= x_min) & (df.x < x_max)]

      for y_idx, (y_min, y_max) in enumerate(self.y_ranges):
        y_min, y_max = conv.trim_range(y_min, y_max, self.size)
        mdl_name = "%s/models/%s/grid_model_x_%s_y_%s.pkl.gz" % (self.root, self.stamp, x_idx, y_idx)
        X, y, row_id = conv.df2sample(df_row, None, None, y_min, y_max, self.x_cols)
        p = mp_pool.apply_async(predict_clf, (mdl_name, X, y, row_id))
        processes.append([x_idx, y_idx, p])
      print("[%s] launching row %i processes @ %s ..." % (title, x_idx, conv.now('full')))
    mp_pool.close()

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
def predict_clf(mdl_name, X, y, row_id, batch=10000):
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
  #
  clf = pickle.load(gzip.open(mdl_name, 'rb'))
  # all_class = [el for el in clf.classes_]
  preds = []
  for ii in range(0, len(X), batch):
    samples = X[ii:ii+batch]
    if len(samples) > 0:
      sols = clf.predict_proba(samples).argsort().T[::-1][:3].T
      preds += [[clf.classes_[i] for i in idxs] for idxs in sols]
  clf = None
  preds = pd.DataFrame(preds)
  preds['row_id'] = row_id
  score = map_score(y, preds)
  return preds, score

