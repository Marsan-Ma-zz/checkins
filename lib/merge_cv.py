import os, time, pickle, gzip
import pandas as pd
import numpy as np
import multiprocessing as mp
pool_size = mp.cpu_count()

import pickle
from datetime import datetime
from collections import OrderedDict, defaultdict
from lib import conventions as conv
from lib import submiter

# global variable for multi-thread
global LOCATION, AVAIL_WDAYS, AVAIL_HOURS, POPULAR, GRID_CANDS
data_cache = './data/cache/data_cache_size10.00_split655200_x0.16y0.08.pkl'
LOCATION, AVAIL_WDAYS, AVAIL_HOURS, POPULAR, GRID_CANDS = pickle.load(open(data_cache, 'rb'))

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


def scoring(y_va, preds):
    score = [apk([y], p) for y,p in zip(y_va, preds)]
    return np.mean(score)


def blendor(y_va, all_va_preds, mdl_weights, rank_w=[1,1,1]):
    preds = []
    for i in range(len(all_va_preds[0])):
        psol = [[(p,v*rank_w[i]*w) for i, (p,v) in enumerate(m[i])] for m, w in zip(all_va_preds, mdl_weights)]
        psol = sorted([j for k in psol for j in k], key=lambda v: v[1])
        psol = sorted(OrderedDict(psol).items(), key=lambda v: v[1], reverse=True)
        psol = [k for k,v in psol][:3]
        preds.append(psol)
    score = None if y_va is None else scoring(y_va, preds)
    return score, preds


def get_row_ids(df_test, size=10.0, x_step=0.16, y_step=0.08):
    x_ranges = conv.get_range(size, x_step, 1)
    y_ranges = conv.get_range(size, y_step, 1)
    row_ids = {}
    for x_idx, (x_min, x_max) in enumerate(x_ranges):
        x_min, x_max = conv.trim_range(x_min, x_max, 10.0)
        df_row = df_test[(df_test.x >= x_min) & (df_test.x < x_max)]
        for y_idx, (y_min, y_max) in enumerate(y_ranges):
            y_min, y_max = conv.trim_range(y_min, y_max, 10.0)
            row_id = df_row[(df_row.y >= y_min) & (df_row.y < y_max)].row_id
            row_ids[(x_idx, y_idx)] = row_id
    return row_ids


def process_grid(path, xi, yi, row_id, eva=True):
    fname = "%s/treva_%i_%i_cv.pkl" % (path, xi, yi)
    if not os.path.exists(fname):
        print("skip cause %s not exists ..." % fname)
        return None, None
    raw = pickle.load(open(fname, 'rb'))
    y_va = raw['y_va']
    all_te_preds = raw['all_te_preds']
    all_va_score = raw['all_va_score']
    all_va_preds = raw['all_va_preds']

    best_idx = all_va_score.index(max(all_va_score))
    mw = [(2 if i == best_idx else 1) for i in range(len(all_va_score))]
    if eva:
        score, _ = blendor(y_va, all_va_preds, mdl_weights=mw)
    else:
        score = None
    _, preds = blendor(None, all_te_preds, mdl_weights=mw)
    preds = pd.DataFrame(preds)
    preds['row_id'] = row_id
    return score, preds


def process_all(path, size=10.0, x_step=0.16, y_step=0.08):
    df_all = []
    score_all = []
    # cache_name = '/home/workspace/checkins/data/cache/cache_get_data_split_655200_size_10.00_rmol_1.00_mci_3.pkl'
    # _, _, df_test = pickle.load(open(cache_name, 'rb'))
    # row_ids = get_row_ids(df_test, size=size, x_step=x_step, y_step=y_step)
    row_ids = pickle.load(open("/home/workspace/checkins/data/cache/row_ids_%.2f_%.2f.pkl" % (x_step, y_step), 'rb'))
    mp_pool = mp.Pool(pool_size)
    processes = []
    for xi in range(int(size//x_step+1)):
        for yi in range(int(size//y_step+1)):
            p = mp_pool.apply_async(process_grid, (path, xi, yi, row_ids[(xi, yi)].values))
            processes.append(p)
            while len(processes) > 30:
                score, preds = processes.pop(0).get()
                if preds is not None:
                    df_all.append(preds)
                    score_all.append((score, len(preds)))
            
    while processes:
        score, preds = processes.pop(0).get()
        if preds is not None:
            df_all.append(preds)
            score_all.append((score, len(preds)))
    df_all = pd.concat(df_all).sort_values(by='row_id')
    return score_all, df_all


def main(stamp, x_step, do_submit=False):
  start_time = time.time()
  timestamp = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
  path = '/home/workspace/checkins/submit/%s' % stamp

  score_all, df_all = process_all(path=path, size=10.0, x_step=x_step, y_step=0.08)
  score_avg = sum([v*l for v, l in score_all]) / sum([l for v,l in score_all])
  print("score_avg: ", score_avg)
  if do_submit:
    sub_fname = '/home/workspace/checkins/submit/%s_%s_cv%.4f.csv' % (stamp, timestamp, score_avg)
    df_all['place_id'] = [" ".join([str(k) for k in l]) for l in df_all[[0,1,2]].values.tolist()]
    df_all[['row_id', 'place_id']].to_csv(sub_fname, index=False)
    print("submit file written in %s" % sub_fname)
    submiter.submiter().submit(entry=sub_fname, message="%s_%s_cv%s" % (stamp, timestamp, score_avg))
  print("[Finish!] Elapsed %.1f secs" % (time.time() - start_time))


#===========================================
#   Main Flow
#===========================================
if __name__ == '__main__':
  main(stamp='treva_full10', x_step=0.16, do_submit=True)



