import os, sys, datetime, time, pickle, shutil
import pandas as pd
import numpy as np
import multiprocessing as mp
pool_size = mp.cpu_count()


from lib import conventions as conv

#===========================================
#   Evaluator
#===========================================  
class evaluator(object):

  def __init__(self, stamp, size, root, x_ranges, y_ranges):
    self.stamp = stamp
    self.size = size
    self.root = root
    self.x_ranges = x_ranges
    self.y_ranges = y_ranges
    
  #----------------------------------------
  #   Main
  #----------------------------------------
  def evaluate(self, df, batch=10000):
    preds_total = []
    score_total = []
    for x_idx, (x_min, x_max) in enumerate(self.x_ranges):
      start_time_row = time.time()
      for y_idx, (y_min, y_max) in enumerate(self.y_ranges):
        start_time_cell = time.time()
        x_min, x_max = conv.trim_range(x_min, x_max, self.size)
        y_min, y_max = conv.trim_range(y_min, y_max, self.size)

        # load model
        mdl_name = "%s/models/%s/grid_model_x_%s_y_%s.pkl" % (self.root, self.stamp, x_idx, y_idx)
        clf = pickle.load(open(mdl_name, 'rb'))
        X, y, row_id = conv.df2sample(df, x_min, x_max, y_min, y_max)
        preds, score = self.predict_clf(clf, X, y, row_id, batch=batch)
        preds_total.append(preds)
        clf = None  # clear memory
        if score: 
          score_total.append(score)
          print("[Evaluate] grid(%s,%s) MAP=%.4f, %i samples for %.2f secs" % (x_idx, y_idx, score, len(X), time.time() - start_time_cell))
        else:
          print("[Evaluate] grid(%i,%i), %i samples for %.2f secs" % (x_idx, y_idx, len(X), time.time() - start_time_cell))
      print("[Evaluate] row %i elapsed: %.2f secs" % (x_idx, time.time() - start_time_row))

    if score_total:
      scores, cnts = zip(*[(s*len(p), len(p)) for s, p in zip(score_total, preds_total)])
      final_score = sum(scores) / sum(cnts)
      print("=====[Final Validation Score] MAP=%.4f =====" % (final_score))
    else:
      final_score = 'none'
      print("=====[Done test sample predicting]=====")
    preds_total = pd.concat(preds_total)
    return preds_total, final_score


  #----------------------------------------
  #   Tasks
  #----------------------------------------
  def apk(self, actual, predicted, k=3):
    if len(predicted) > k: 
      predicted = predicted[:k]
    score, num_hits = 0.0, 0.0
    for i,p in enumerate(predicted):
      if p in actual and p not in predicted[:i]:
        num_hits += 1.0
        score += num_hits / (i+1.0)
    if not actual: return 0.0
    return score / min(len(actual), k)


  def map_score(self, y, preds):
    if y is None: return None
    match = [self.apk([ans], vals) for ans, vals in zip(y, preds[[0,1,2]].values)]
    return sum(match)/len(match)



  def predict_clf(self, clf, X, y, row_id, batch):
    all_class = [el for el in clf.classes_]
    preds = []
    for ii in range(0, len(X), batch):
      samples = X[ii:ii+batch]
      if len(samples) > 0:
        sols = clf.predict_proba(samples).argsort().T[::-1][:3].T
        preds += [[all_class[i] for i in idxs] for idxs in sols]
    preds = pd.DataFrame(preds)
    preds['row_id'] = row_id
    score = self.map_score(y, preds)
    return preds, score


  def save_and_clear(self, preds_total, score):
    preds_total['results'] = [" ".join([str(k) for k in l]) for l in preds_total[[0,1,2]].values.tolist()]
    preds_total[['row_id', 'results']].sort_values(by='row_id').to_csv("%s/submit/submit_%s_%.4f.csv" % (self.root, self.stamp, score), index=False)
    return  


  # clear meta files
  def clear_meta_files(self):
    mdl_path = "%s/models/%s" % (self.root, self.stamp)
    shutil.rmtree(mdl_path)


