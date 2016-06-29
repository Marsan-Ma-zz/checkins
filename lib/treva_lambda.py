import os, sys, time, pickle, gzip, re, ast, boto3, json
import pandas as pd
import numpy as np
import multiprocessing as mp
pool_size = mp.cpu_count()

import boto3.session
import xgboost as xgb

from os import listdir
from time import sleep
from datetime import datetime
from collections import OrderedDict, defaultdict

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, ensemble

from lib import conventions as conv
from lib import submiter

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
    if self.size == 10.0:
      # mdl_path = "%s/submit/treva_full10" % (self.root)
      mdl_path = "%s/submit/treva_fast10" % (self.root)
    else:
      mdl_path = "%s/submit/treva_%s" % (self.root, self.stamp)
    if not os.path.exists(mdl_path): os.mkdir(mdl_path)
    print("[Train] start @ %s" % (conv.now('full')))
    
    processes = []
    preds_total = []
    score_stat = []
    mp_pool = mp.Pool(500)
    for x_idx, (x_min, x_max) in enumerate(self.x_ranges):
      x_min, x_max = conv.trim_range(x_min, x_max, self.size)
      df_row = {
        'tr': df_train[(df_train.x >= x_min) & (df_train.x < x_max)],
        'va': df_valid[(df_valid.x >= x_min) & (df_valid.x < x_max)],
        'te': df_test[(df_test.x >= x_min) & (df_test.x < x_max)],
      }
      for y_idx, (y_min, y_max) in enumerate(self.y_ranges): 
        # check exists
        grid_submit_path = "%s/treva_%i_%i.csv" % (mdl_path, x_idx, y_idx)
        # if x_idx < 90: continue
        if os.path.exists(grid_submit_path):
          print("%s exists, skip." % grid_submit_path)
          continue

        # get grid
        y_min, y_max = conv.trim_range(y_min, y_max, self.size)
        df_grid = {m: df_row[m][(df_row[m].y >= y_min) & (df_row[m].y < y_max)] for m in ['tr', 'va', 'te']}
        p = mp_pool.apply_async(drill_grid, (df_grid, self.x_cols, x_idx, y_idx, self.stamp, grid_submit_path))
        processes.append(p)

        # prevent memory explode!
        while (len(processes) > 500):
          _, y_test = processes.pop(0).get()
          if y_test is not None: preds_total.append(y_test)

    while processes: 
      _, y_test = processes.pop(0).get()
      if y_test is not None: preds_total.append(y_test)
    mp_pool.close()
    # write submit file
    preds_total = pd.concat(preds_total)
    df2submit(preds_total, self.submit_file)
    # collect scores
    print("[Treva] done, submit file written %s @ %s" % (self.submit_file, conv.now('full')))
    
  
#----------------------------------------
#   Drill Grid
#----------------------------------------
raw_mdl_configs = [
  # {'alg': 'skrf', 'n_estimators': 500, 'max_features': 0.35, 'max_depth': 15},
  # {'alg': 'skrfp', 'n_estimators': 500, 'max_features': 0.35, 'max_depth': 15},
  # {'alg': 'sket', 'n_estimators': 800, 'max_features': 0.5, 'max_depth': 15},
  # {'alg': 'sketp', 'n_estimators': 1000, 'max_features': 0.5, 'max_depth': 11},
  {'alg': 'skrf', 'n_estimators': 50, 'max_features': 0.35, 'max_depth': 15},
  {'alg': 'skrfp', 'n_estimators': 50, 'max_features': 0.35, 'max_depth': 15},
  {'alg': 'sket', 'n_estimators': 80, 'max_features': 0.5, 'max_depth': 15},
  {'alg': 'sketp', 'n_estimators': 100, 'max_features': 0.5, 'max_depth': 11},
]

# for alg in ['skrf', 'skrfp', 'sket', 'sketp']:
#   for n_estimators in [500]:
#     for max_features in [0.35]:   # 0.4
#       for max_depth in [11, 15]:    # 11
#         mdl_configs.append({'alg': alg, 'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth}) 

# bucket256 = boto3.resource('s3', region_name='us-west-2').Bucket('recom.lambda.m256')
# bucket512 = boto3.resource('s3', region_name='us-west-2').Bucket('recom.lambda.m512')
# bucket1024 = boto3.resource('s3', region_name='us-west-2').Bucket('recom.lambda.m1024')

def drill_grid(df_grid, x_cols, xi, yi, stamp, grid_submit_path, do_blending=True):
  # params init
  best_score = 0
  all_score = []
  Xs, ys = {}, {}
  for m in ['tr', 'te']:
    Xs[m], ys[m], row_id = conv.df2sample(df_grid[m], x_cols)
  scnt = len(ys['tr'])
  
  mdl_path = '/'.join(grid_submit_path.split('/')[:-1])
  dat_file = 'dat_%i_%i.pkl' % (xi, yi)
  sol_file = 'dat_%i_%i.sol.%s' % (xi, yi, stamp)

  # if scnt > 1500:
  #   mdl_configs = [
  #     {'alg': 'skrf', 'n_estimators': 500, 'max_features': 0.35, 'max_depth': 15},
  #     {'alg': 'skrfp', 'n_estimators': 500, 'max_features': 0.35, 'max_depth': 15},
  #     {'alg': 'sket', 'n_estimators': 500, 'max_features': 0.4, 'max_depth': 15},
  #     {'alg': 'sketp', 'n_estimators': 500, 'max_features': 0.4, 'max_depth': 11},
  #   ]
  # else:
  mdl_configs = raw_mdl_configs

  # prepare data
  cmds = {
    'mdl_configs': mdl_configs,
    'Xs': {k:v.values for k,v in Xs.items()},
    'ys': ys,
    'row_id': row_id.values,
    'sol_file': sol_file,
  }
  dat_path = "%s/%s" % (mdl_path, dat_file)
  try:
    pickle.dump(cmds, open(dat_path, 'wb'), protocol=2) # lambda only has python2, which support latest pickle protocol=2 
  except Exception as e:
    print(e)
    print("ERROR for (%i/%i), cmds: %s" % (xi, yi, cmds))

  # upload to s3 for lambda
  job_done = False

  # bucket = boto3.resource('s3', region_name='us-west-2').Bucket('recom.lambda.m1536')
  session = boto3.session.Session()
  bucket = session.resource('s3', region_name='us-west-2').Bucket('recom.lambda.m1536')

  #-----[use aws lambda]-------------------------------------------
  if True: #scnt < 2500:
    # if scnt > 1000:
    #   bucket = bucket1536
    # elif scnt > 500:
    #   bucket = bucket1024
    # elif scnt > 300:  
    #   bucket = bucket512
    # else:
    #   bucket = bucket256
    try:
      bucket.upload_file(dat_path, dat_file)
    except Exception as e:
      print(e)
      print("when bucket.upload_file", dat_path)
    print("upload dat_file %s of %i tr samples @ %s" % (dat_file, len(ys['tr']), datetime.now()))
    df_grid, Xs, ys, row_id, cmds = [None]*5  # release memory
    
    
    # print("try download %s to %s" % (sol_file, grid_submit_path))
    try_cnt, max_try = 0, 6
    while try_cnt <= max_try:
      try:
        bucket.download_file(sol_file, grid_submit_path)
        job_done = True
        break
      except Exception as e:
        if try_cnt > 4: print("(%i/%i) scnt=%i, waiting %i ... @ %s" % (xi, yi, scnt, try_cnt, datetime.now()))
        try_cnt += 1
        sleep(30)

    # remove tmp files
    bucket.delete_objects(Delete={'Objects': [{'Key': sol_file}], 'Quiet': True,})

    # collect sols
    if job_done:
      try:
        sols = json.load(open(grid_submit_path, 'rt'))
      except Exception as e:
        print(e)
        print("when json try load %s" % grid_submit_path)
      # print(sols[:5])
      sols = pd.DataFrame(sols)
      sols['row_id'] = row_id
      df2submit(sols, grid_submit_path)
      # print("get sols:\n %s \n@ %s" % (sols.head(), datetime.now()))
      print("[drill_grid (%i,%i)] blended @ %s" % (xi, yi, datetime.now()))
    else:
      sols = None
      print("[TIMEOUT] job timeout: (%i/%i)" % (xi, yi))

  #-----[use local machine]-------------------------------------------
  else:
    all_bt_preds = []
    for bcfg in mdl_configs:
      bmdl = get_alg(bcfg['alg'], bcfg)
      bmdl.fit(Xs['tr'], ys['tr'])
      _, bt_preds = drill_eva(bmdl, Xs['te'], ys['te'])
      all_bt_preds.append(bt_preds)
    sols = blending(all_bt_preds)
    sols = pd.DataFrame(sols)
    sols['row_id'] = row_id
    df2submit(sols, grid_submit_path)
    print("[LOCAL] done (%i/%i) locally @ %s" % (xi, yi, datetime.now()))

  return 1.0, sols



def df2submit(df, filename):
  df['place_id'] = df[[0,1,2]].astype(str).apply(lambda x: ' '.join(x), axis=1)
  df[['row_id', 'place_id']].sort_values(by='row_id').to_csv(filename, index=False)


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


#===========================================
#   Analyse model parameter from log
#===========================================
# submit_partial_merge(base="blending_20160621_214954_0.58657.csv.gz", folder="treva_full10")
def submit_partial_merge(base, folder, all_blended=False):
  root_path = '/home/workspace/checkins'
  folder = "%s/submit/%s" % (root_path, folder)
  stamp = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
  output = "%s/submit/treva_overwrite_%s_all_blended_%s.csv" % (root_path, stamp, all_blended)

  if all_blended:
    tfiles = [f for f in listdir(folder) if 'blend' in f]
  else:
    tfiles = [f for f in listdir(folder) if 'blend' not in f]

  # # remove old batch
  # print("tfiles before removing old batch: %i" % len(tfiles))
  # old_partials = [f for f in listdir(root_path + "/submit/treva_merge")]
  # tfiles = [f for f in tfiles if f not in old_partials]
  # print("tfiles after removing old batch: %i" % len(tfiles))

  # concat and merge
  df_treva = [pd.read_csv("%s/%s" % (folder, f)) for f in tfiles]
  df_treva = pd.concat(df_treva).sort_values(by='row_id')
  df_base = pd.read_csv("%s/data/submits/%s" % (root_path, base))

  df_base = df_base[~df_base.row_id.isin(df_treva.row_id.values)]
  df_overwrite = pd.concat([df_base, df_treva]).sort_values(by='row_id')
  df_overwrite[['row_id', 'place_id']].sort_values(by='row_id').to_csv(output, index=False)
  print("ensure dim:", len(df_treva), len(set(df_treva.row_id.values)), len(set(df_overwrite.row_id.values)))
  print("overwrite output written in %s @ %s" % (output, datetime.now()))
  # submiter.submiter().submit(entry=output, message="treva submit_partial_merge with %s and all_blended=%s" % (base, all_blended))


def analysis_params(log_path):
  raw = open(log_path, 'rt')
  cfg_stat = defaultdict(float)
  for line in raw.readlines():
    if 'va_score' in line:
      if 'blending' in line:
        cfg = 'blending'
      else:
        cfg = re.compile('{.*}').findall(line)[0]
      score = float(re.compile('0\.\d+').findall(line)[0])
      # print(score, cfg)
      cfg_stat[cfg] += score
  results = sorted(cfg_stat.items(), key=lambda v: v[1], reverse=True)
  for line in results:
    print(line)

def analysis_best(log_path):
  raw = open(log_path, 'rt')
  cfg_stat = defaultdict(float)
  for line in raw.readlines():
    if 'model is' in line:
      if 'blending' in line:
        cfg = 'blending'
      else:
        cfg = re.compile('{.*}').findall(line)[0]
      cfg_stat[cfg] += 1
  results = sorted(cfg_stat.items(), key=lambda v: v[1], reverse=True)
  for line in results:
    print(line)



if __name__ == '__main__':
    # -----[analyses treva params]-----
    log_path = "/home/workspace/checkins/logs/nohup_treva_all_20160626_090148.log"
    analysis_params(log_path)
    # analysis_best(log_path)

    # # -----[submit treva partial merge]-----
    # submit_partial_merge(base='blending_20160621_214954_0.58657.csv.gz', folder="treva_merge2", all_blended=False)
    # submit_partial_merge(base='blending_20160621_214954_0.58657.csv.gz', folder="treva_merge2", all_blended=True)
