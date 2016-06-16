import os, sys, time, pickle, gzip, shutil
import pandas as pd
import numpy as np
import multiprocessing as mp
pool_size = mp.cpu_count()

from datetime import datetime
from collections import defaultdict, OrderedDict


#----------------------------------------
#   Blending Models
#----------------------------------------
def csv2df(mname, out='df', nrows=None):
  fname = "/home/workspace/checkins/data/submits/%s" % mname
  df = pd.read_csv(fname, skiprows=1, names=['row_id', 0, 1, 2], sep=' |,', engine='python', nrows=nrows)
  if out == 'dic':
    df = dict(zip(df.row_id.values, df[[0,1,2]].to_dict('records')))
  print("%s loaded @ %s" % (mname, datetime.now()))
  return df


def blending(models, mdl_weights, rank_w):
  print("[Blending Start]", datetime.now())    
  results = {}
  mdl_cnt = len(mdl_names)
  processes = []
  mp_pool = mp.Pool(pool_size)
  for idx, k in enumerate(models[0].keys()):
    stat = defaultdict(float)
    for v,w in [(models[i][k], mdl_weights[i]) for i in range(mdl_cnt)]:
      for rank, pid in v.items():
        stat[pid] += rank_w[rank]*w
    stat = sorted(stat.items(), key=lambda v: v[1], reverse=True)
    results[k] = [k for k,v in stat][:3]
    if (idx % 1e5 == 0): print('%i samples blended @ %s' % (idx, datetime.now()))
  print("[Blending] done with %i samples @ %s" % (len(models[0]), datetime.now()))
  return results
  
# def blend_row(mdl_info):
#   stat = defaultdict(float)
#   for v,w in mdl_info:
#     for rank, pid in v.items():
#       stat[pid] += rank_w[rank]*w
#   stat = sorted(stat.items(), key=lambda v: v[1], reverse=True)
#   stat = [k for k,v in stat][:3]
#   return stat


# def blending(models, mdl_weights, rank_w):
#   print("[Blending Start]", datetime.now())    
#   results = {}
#   mdl_cnt = len(mdl_names)
#   processes = []
#   mp_pool = mp.Pool(pool_size)
#   for idx, k in enumerate(models[0].keys()):
#     mdl_info = [(models[i][k], mdl_weights[i]) for i in range(mdl_cnt)]
#     p = mp_pool.apply_async(blend_row, (mdl_info,))
#     processes.append([idx, k, p])
#     if (idx % 1e5 == 0): print('%i samples launched @ %s' % (idx, datetime.now()))
#   mp_pool.close()
#   for idx, k, p in processes:
#     results[k] = p.get()
#     if (idx % 1e5 == 0): print('%i samples blended @ %s' % (idx, datetime.now()))
#   print("[Blending] done with %i samples @ %s" % (len(models[0]), datetime.now()))
#   return results


def submit_blendings(models, mdl_weights, rank_w, output_fname):
  print("[Start]", datetime.now())    
  results = blending(models, mdl_weights, rank_w)
  with open(output_fname, 'wt') as f:
    f.write("row_id,place_id\n")
    for k,v in results.items():
      for i in range(3):
        v[i] = int(v[i])
        if v[i] < 1000000000: print("[Error] in k=%i, v=%s" % (k,v))
      f.write("%s,%s %s %s\n" % (k, v[0], v[1], v[2]))
  print("[All done]", datetime.now())    


def cal_correlation(ma, mb, rule=2):
  if rule == 1: # faster (15s/model)
    score = sum(sum(ma[[0,1,2]].values == mb[[0,1,2]].values))/len(ma)/3
  elif rule == 2: # slower (40s/model)
    score = sum([len(set(va) & set(vb)) for va, vb in zip(ma[[0,1,2]].values, mb[[0,1,2]].values)])/len(ma)/3
  print("cal_correlation=%.4f @ %s" % (score, datetime.now()))
  return round(score, 2)


def load_models(mdl_names, out='dic', rows=None):
  processes = []
  mp_pool = mp.Pool(pool_size)
  models = {}
  for idx, (mname, w) in enumerate(mdl_names):
    p = mp_pool.apply_async(csv2df, (mname, out, rows))
    processes.append([p, idx])
  mp_pool.close()
  models = {idx: p.get() for p, idx in processes}
  return models



#===========================================
#   Main Flow
#===========================================
if __name__ == '__main__':
  mdl_names = [
      #-----[King]-----
      ('lb_marsan_blending_0614_0.58378.csv.gz'           , 2.0 ),
      # ('lb_marsan_blending_0613_0.58299.csv.gz'           , 2.0 ),
      # ('lb_marsan_blending_0613_0.57664.csv.gz'           , 2.0 ),
      ('lb_grid_knn_lonely_shepard_0.57004.csv.gz'        , 1.5 ),
      
      #-----[Knight]-----
      ('submit_knn_submit_20160615_230955_0.56815.csv.gz',  0.8 ),
      ('submit_skrf_submit_20160605_195424_0.56552.csv.gz', 0.7 ),
      ('submit_skrf_submit_20160608_174129_0.56533.csv.gz', 0.7 ),
      ('submit_skrf_submit_20160602_104038_0.56093.csv.gz', 0.7 ),
      ('submit_skrf_submit_20160604_171454_0.56130.csv.gz', 0.5 ),
      ('submit_skrf_submit_20160612_214750_0.55583.csv.gz', 0.5 ),
      ('submit_xgb_submit_20160604_173333_0.55361.csv.gz' , 0.5 ),
      
      # #-----[Ash]-----
      ('submit_skrf_submit_20160530_155226_0.53946.csv.gz', 0.2 ),
      ('submit_skrf_submit_20160530_143859_0.52721.csv.gz', 0.2 ),
      ('submit_skrf_submit_20160611_233627_0.52375.csv.gz', 0.2 ),
      ('submit_skrf_submit_20160613_213413_0.51390.csv.gz', 0.2 ),
      ('lb_msb_battle_0.51115.csv.gz'                     , 0.1 ),
      ('lb_r_try_to_add_hour_rating_0.51115.csv.gz'       , 0.1 ),
      ('submit_skrf_submit_20160605_195424_0.50870.csv.gz', 0.1 ),
  ]
  # ---------- [corr_matrix] ----------------------------------------
 #[[None 0.66 0.78 0.85 0.76 0.80 0.75 0.67 0.63 0.61 0.63 0.59 0.59]
 # [None None 0.64 0.64 0.62 0.63 0.61 0.56 0.58 0.57 0.56 0.61 0.62]

 # [None None None 0.71 0.68 0.82 0.66 0.66 0.59 0.59 0.61 0.58 0.58]
 # [None None None None 0.65 0.71 0.71 0.61 0.59 0.57 0.60 0.57 0.58]
 # [None None None None None 0.68 0.65 0.60 0.62 0.60 0.60 0.55 0.56]
 # [None None None None None None 0.65 0.66 0.58 0.57 0.60 0.56 0.56]
 # [None None None None None None None 0.57 0.64 0.64 0.59 0.56 0.55]
 # [None None None None None None None None 0.52 0.51 0.56 0.52 0.52]

 # [None None None None None None None None None 0.70 0.49 0.53 0.52]
 # [None None None None None None None None None None 0.51 0.55 0.53]
 # [None None None None None None None None None None None 0.52 0.52]
 # [None None None None None None None None None None None None 0.83]
 # [None None None None None None None None None None None None None]]

  rank_w = [1, 0.4, 0.2]
  mdl_weights = [v for k,v in mdl_names]
  timestamp = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
  output_fname = "./submit/blending_%s.csv" % timestamp
  print(mdl_names)
  
  if True:  # check model correlation
    # models = {idx: csv2df(mname, nrows=100000) for idx, (mname, w) in enumerate(mdl_names)}
    models = load_models(mdl_names, out='df', rows=100000)
    corr_matrix = np.array([[None if i >= j else cal_correlation(models[i], models[j]) for j in range(len(models))] for i in range(len(models))])
    print('-'*10, "[corr_matrix]", '-'*40)
    print(corr_matrix)

  if False:  # blending
    # models = {idx: csv2df(mname, out='dic', nrows=None) for idx, (mname, w) in enumerate(mdl_names)}
    models = load_models(mdl_names, out='dic', rows=None)
    submit_blendings(models, mdl_weights, rank_w, output_fname)
    print("[Finished!!] blending results saved in %s @ %s" % (output_fname, datetime.now()))


### =====[lb_marsan_blending_0613_0.58299.csv.gz]=====
# [[None 0.66 0.78 0.77 0.75 0.67 0.63 0.61 0.63 0.59]
#  [None None 0.64 0.62 0.61 0.56 0.58 0.57 0.56 0.62]
#  [None None None 0.68 0.66 0.66 0.59 0.59 0.61 0.58]
#  [None None None None 0.65 0.6 0.62 0.6 0.6 0.55]
#  [None None None None None 0.57 0.64 0.64 0.59 0.55]
#  [None None None None None None 0.52 0.51 0.56 0.52]
#  [None None None None None None None 0.7 0.49 0.52]
#  [None None None None None None None None 0.51 0.53]
#  [None None None None None None None None None 0.52]
#  [None None None None None None None None None None]]
# lb_marsan_blending_0613_0.57664.csv.gz loaded @ 2016-06-13 21:36:40.793420
# lb_grid_knn_lonely_shepard_0.57004.csv.gz loaded @ 2016-06-13 21:39:28.312524
# submit_skrf_submit_20160605_195424_0.56552.csv.gz loaded @ 2016-06-13 21:42:10.568697
# submit_skrf_submit_20160602_104038_0.56093.csv.gz loaded @ 2016-06-13 21:45:08.484788
# submit_skrf_submit_20160612_214750_0.55583.csv.gz loaded @ 2016-06-13 21:48:36.535278
# submit_xgb_submit_20160604_173333_0.55361.csv.gz loaded @ 2016-06-13 21:52:45.876670
# submit_skrf_submit_20160530_155226_0.53946.csv.gz loaded @ 2016-06-13 21:57:22.735985
# submit_skrf_submit_20160530_143859_0.52721.csv.gz loaded @ 2016-06-13 22:01:49.370027
# submit_skrf_submit_20160611_233627_0.52375.csv.gz loaded @ 2016-06-13 22:06:39.910734
# lb_r_try_to_add_hour_rating_0.51115.csv.gz loaded @ 2016-06-13 22:11:03.237066

