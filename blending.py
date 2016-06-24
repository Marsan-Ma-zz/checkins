import os, sys, time, pickle, gzip, shutil
import pandas as pd
import numpy as np
import multiprocessing as mp
pool_size = mp.cpu_count()

from datetime import datetime
from collections import defaultdict, OrderedDict

from lib import submiter

#----------------------------------------
#   Blending Models
#----------------------------------------
def csv2df(mname, out='df', nrows=None):
  fname = "/home/workspace/checkins/data/submits/%s" % mname
  df = pd.read_csv(fname, skiprows=1, names=['row_id', 0, 1, 2], sep=' |,', engine='python', nrows=nrows)
  # detect null rows
  null_rows = df[df.isnull().any(axis=1)]
  if len(null_rows):
    print("[null_rows from %s]\n" % mname, null_rows)
    df.fillna(0, inplace=True)
  if out == 'dic':
    df = dict(zip(df.row_id.values, df[[0,1,2]].astype(int).to_dict('records')))
  print("%s loaded @ %s" % (mname, datetime.now()))
  return df


def submit_blendings(models, mdl_weights, rank_w, output_fname, rows=None):
  print("[Start]", datetime.now())    
  with open(output_fname, 'wt') as f:
    f.write("row_id,place_id\n")
    mdl_cnt = len(models)
    idx = 0
    while True:
      # collect voting
      stat = defaultdict(float)
      try:
        raw = [(next(models[i]), mdl_weights[i]) for i in range(mdl_cnt)]
      except StopIteration:
        print("generator exhausted, idx=%i @ %s" % (idx, datetime.now()))
        break
      for v,w in raw:
        for rank, pid in v.items():
          stat[pid] += rank_w[rank]*w
        stat[0] = 0 # prevent empty submit
      stat = sorted(stat.items(), key=lambda v: v[1], reverse=True)
      stat = [pid for pid,val in stat][:3]
      f.write("%s,%s %s %s\n" % (idx, stat[0], stat[1], stat[2]))
      idx += 1
      if (rows and (idx >= rows)): break
      # err check
      for i in range(3):
        v[i] = int(v[i] or 0)
        if v[i] < 1000000000: print("[Error] in k=%i, v=%s" % (k,v))
      if (idx % 1e5 == 0): print('%i samples blended @ %s' % (idx, datetime.now()))
  print("[All done]", datetime.now())    


def cal_correlation(ma, mb, rule=2):
  if rule == 1: # faster (15s/model)
    score = sum(sum(ma[[0,1,2]].values == mb[[0,1,2]].values))/len(ma)/3
  elif rule == 2: # slower (40s/model)
    score = sum([len(set(va) & set(vb)) for va, vb in zip(ma[[0,1,2]].values, mb[[0,1,2]].values)])/len(ma)/3
  # print("cal_correlation=%.4f @ %s" % (score, datetime.now()))
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


def file2gen(fname):
  f = gzip.open("/home/workspace/checkins/data/submits/%s" % fname)
  f.readline() # cast header
  for line in f:
    line = line.decode(encoding='UTF-8').replace("\n", '').split(',')
    row_id = line[0]
    place_id = line[1].split(' ')
    yield {i: (int(p) if p.isdigit() else 0) for i, p in enumerate(place_id)}
  f.close()

#===========================================
#   Blendor
#===========================================
class blendor(object):

  def __init__(self, top_w={0: 2.0}, rank_w=[1, 0.6, 0.4], do_corr=True, do_blend=True, do_upload=False):
    self.root = '.'
    self.top_w = top_w
    self.rank_w = rank_w
    self.do_corr = do_corr
    self.do_blend = do_blend
    self.do_upload = do_upload
    #
    self.mdl_names = []
    self.output_fname = ""
    self.do_corr_rows = 100000
    self.do_blend_rows = None


  def init_models(self):
    self.mdl_names = [
        #-----[King]-----
        ('blending_20160621_214954_0.58657.csv.gz'            , 2.0 ),
        # ('lb_marsan_blending_0622_0.58569.csv.gz'             , 2.0 ),
        # ('blending_gs_top_w2_20160619_180546_0.58529.csv.gz'  , 2.0 ),
        # ('lb_blending_20160617_215629_0.58463.csv.gz'       , self.top_w.get(0, 2.0) ),
        # ('lb_marsan_blending_0614_0.58378.csv.gz'           , 2.0 ),
        # ('lb_marsan_blending_0613_0.58299.csv.gz'           , 2.0 ),
        # ('lb_anouymous_0.58018.csv.gz'                        , 1.8 ),
        # ('lb_marsan_blending_0613_0.57664.csv.gz'           , 2.0 ),
        # ('lb_anouymous_0.57842.csv.gz'                        , 1.5 ),
        # ('lb_hamed_0.57946.csv.gz'                            , 1.2 ),
        ('skrf_submit_20160623_224407_0.57349.csv.gz'         , 1.5 ),
        # ('lb_sub_knn_daten-kieker_0.57189.csv.gz'             , 1.2 ),
        ('skrf_submit_full_20160620_000902_0.57114.csv.gz'    , 1.2 ),
        # ('lb_sub_knn_danielspringt_0.57068.csv.gz'          , 1.2 ),
        ('skrfp_submit_full_20160621_0.57054.csv.gz'          , 1.2 ),
        # ('lb_daniel_0.57068.csv.gz'                         , 1.0 ),
        # ('lb_grid_knn_lonely_shepard_0.57004.csv.gz'        , 1.0 ),
        # ('knn_grid_0.8_20160618_081945_0.56999.csv.gz'        , 1.0 ),
        ('skrf_submit_full_20160621_234034_0.56992.csv.gz'    , 1.0 ),
        #-----[Knight]-----
        # ('submit_knn_0.4_grid_20160617_091633_0.56919.csv.gz',0.9 ),
        # ('submit_knn_submit_20160615_230955_0.56815.csv.gz',  0.8 ),
        ('submit_skrf_submit_20160605_195424_0.56552.csv.gz', 0.7 ),
        ('submit_skrf_submit_20160608_174129_0.56533.csv.gz', 0.7 ),
        ('submit_sket_20160622_130909_0.56315.csv.gz'       , 0.7 ),
        ('sketp_submit_full_20160622_180604_0.56173.csv.gz' , 0.7 ),
        ('submit_skrf_submit_20160604_171454_0.56130.csv.gz', 0.5 ),
        ('submit_skrf_submit_20160602_104038_0.56093.csv.gz', 0.7 ),
        ('submit_sketp_20160622_180604_0.56082.csv.gz'      , 0.7 ),
        ('submit_knn_inter_20160616_172918_0.55919.csv.gz'  , 0.6 ),
        ('xgb_submit_full_20160616_0.55615.csv.gz'          , 0.6 ),
        ('submit_skrf_submit_20160612_214750_0.55583.csv.gz', 0.5 ),
        ('submit_xgb_submit_20160604_173333_0.55361.csv.gz' , 0.5 ),
        
        #-----[Ash]-----
        ('submit_skrf_submit_20160530_155226_0.53946.csv.gz', 0.2 ),
        ('submit_skrf_submit_20160530_143859_0.52721.csv.gz', 0.2 ),
        ('submit_skrf_submit_20160611_233627_0.52375.csv.gz', 0.2 ),
        # ('submit_skrf_submit_20160613_213413_0.51390.csv.gz', 0.2 ),
        # ('lb_msb_battle_0.51115.csv.gz'                     , 0.1 ),
        # ('lb_r_try_to_add_hour_rating_0.51115.csv.gz'       , 0.1 ),
        # # ('submit_skrf_submit_20160605_195424_0.50870.csv.gz', 0.1 ),
        # ('submit_xgb_submit_20160614_230100_0.50970.csv.gz' , 0.1 ),
    ]


  def launch(self, stamp=None):
    if not self.mdl_names: self.init_models()
    mdl_weights = [v for k,v in self.mdl_names]
    stamp = stamp or str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    self.output_fname = "%s/submit/blending_%s.csv" % (self.root, stamp)
    print(self.mdl_names)
    
    if self.do_corr:  # check model correlation
      models = load_models(self.mdl_names, out='df', rows=self.do_corr_rows)
      corr_matrix = np.array([[None if i >= j else cal_correlation(models[i], models[j]) for j in range(len(models))] for i in range(len(models))])
      print('-'*10, "[corr_matrix]", '-'*40)
      for c in corr_matrix: print(c)

    if self.do_blend:  # blending
      # models = load_models(self.mdl_names, out='dic', rows=self.do_blend_rows)
      models = [file2gen(fname) for fname, w in self.mdl_names]
      submit_blendings(models, mdl_weights, self.rank_w, self.output_fname, rows=self.do_blend_rows)
      print("[Finished!!] blending results saved in %s @ %s" % (self.output_fname, datetime.now()))
    

  def run(self, cmd=None):
    #---------------------------------------------
    if cmd == 'gs_top_w':
      for top_w in [1.5, 1.7, 1.9]:
        self.top_w = {0: top_w}
        stamp = "gs_top_w%s_%s" % (top_w, str(datetime.now().strftime("%Y%m%d_%H%M%S")))
        self.launch(stamp=stamp)
        print("[RUN] done gs_top_w=%i" % (top_w))
    #---------------------------------------------
    elif cmd == 'gs_rank_w':
      rank_ws = [
        [1, 0.8, 0.6],
        [1, 0.8, 0.4],
        [1, 0.6, 0.4],
        [1, 0.6, 0.1],
        [1, 0.4, 0.2],
      ]
      for rank_w in rank_ws:
        self.rank_w = rank_w
        stamp = "gs_rank_ws_%s_%s" % ("_".join([str(w) for w in rank_w]), str(datetime.now().strftime("%Y%m%d_%H%M%S")))
        self.launch(stamp=stamp)
        print("[RUN] done gs_rank_ws=%s" % (rank_w))
    #---------------------------------------------
    elif cmd == 'debug':
      self.do_corr_rows = 100000
      self.do_blend_rows = 100000
      self.launch()
    elif cmd == 'average':
      self.init_models()
      self.mdl_names = [(k, 1) for k,v in self.mdl_names]
      self.launch()
    else:
      self.launch()
    # auto-submit
    if self.do_upload:
      submit.submitor().submit(entry=self.output_fname, message=self.mdl_names)
    


#===========================================
#   Main Flow
#===========================================
if __name__ == '__main__':
  # blendor(do_blend=True).run('debug')   # cal corr only
  blendor(do_blend=True, do_upload=True).run('average')
  # bla.run()
  # bla.run('submit')
  # bla.run('gs_top_w')
  # bla.run('gs_rank_w')

