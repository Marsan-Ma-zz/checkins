import os, sys, time, pickle, gzip, shutil, gc
import pandas as pd
import numpy as np
import multiprocessing as mp
pool_size = mp.cpu_count()

from itertools import islice
from datetime import datetime
from collections import defaultdict, OrderedDict

from lib import submiter

#----------------------------------------
#   Blending Models
#----------------------------------------
def csv2df(mname, out='df', skiprows=0, nrows=None):
  fname = "/home/workspace/checkins/data/submits/%s" % mname
  df = pd.read_csv(fname, names=['row_id', 0, 1, 2], sep=' |,', engine='python', skiprows=1+skiprows, nrows=nrows)
  # detect null rows
  null_rows = df[df.isnull().any(axis=1)]
  if len(null_rows):
    # print("[null_rows from %s]\n" % mname, null_rows)
    df.fillna(0, inplace=True)
  if out == 'dic':
    df = dict(zip(df.row_id.values, df[[0,1,2]].astype(int).to_dict('records')))
  if skiprows == 0: print("%s loaded @ %s" % (mname, datetime.now()))
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


# def fetch_slice(gen):
#   return list(islice(gen, 10000))


# def raw_slice2all_stat(raw_slices, mdl_weights, ridx, rank_w):
#   all_stat = []
#   raw_slices = [[(raw_slices[j][ridx+i], mdl_weights[j]) for j in range(len(raw_slices))] for i in range(len(raw_slices[0]))]
#   for raw in raw_slices:
#     stat = defaultdict(float)
#     for v,w in raw:
#       for rank, pid in v.items():
#         stat[pid] += rank_w[rank]*w
#       stat[0] = 0 # prevent empty submit
#     stat = sorted(stat.items(), key=lambda v: v[1], reverse=True)
#     stat = [pid for pid,val in stat][:3]
#     all_stat.append(stat)
#   raw_slices = None
#   # gc.collect()
#   return all_stat


# def submit_blendings(mdl_names, mdl_weights, rank_w, output_fname, rows=None, batch=100000):
#   print("[Start]", datetime.now())    
#   mdl_cnt = len(mdl_names)
#   ridx = 0
#   processes = []
#   mp_pool = mp.Pool(pool_size)
#   while True:
#     # collect voting
#     # try:
#     raw_slices = load_models(mdl_names, out='dic', skiprows=ridx, rows=batch)
#     if len(raw_slices[0]) == 0: break
#     p = mp_pool.apply_async(raw_slice2all_stat, (raw_slices, mdl_weights, ridx, rank_w))
#     processes.append(p)
#     print("loaded models from %i - %i @ %s" % (ridx, ridx+batch, datetime.now()))
#     ridx += len(raw_slices[0])
#     # except StopIteration:
#     #   print("generator exhausted, ridx=%i @ %s" % (ridx, datetime.now()))
#     #   break

#   idx = 0
#   with open(output_fname, 'wt') as f:
#     f.write("row_id,place_id\n")
#     while processes:
#       all_stat = processes.pop(0).get()
#       print("start writing %i samples" % len(all_stat))
#       for stat in all_stat:
#         f.write("%s,%s %s %s\n" % (idx, stat[0], stat[1], stat[2]))
#         idx += 1
#         # if (rows and (idx >= rows)): break
#         if (idx % batch == 0): print('%i samples blended @ %s' % (idx, datetime.now()))
#   mp_pool.close()
#   print("[All done]", datetime.now())    


def cal_correlation(ma, mb, rule=2):
  if rule == 1: # faster (15s/model)
    score = sum(sum(ma[[0,1,2]].values == mb[[0,1,2]].values))/len(ma)/3
  elif rule == 2: # slower (40s/model)
    score = sum([len(set(va) & set(vb)) for va, vb in zip(ma[[0,1,2]].values, mb[[0,1,2]].values)])/len(ma)/3
  # print("cal_correlation=%.4f @ %s" % (score, datetime.now()))
  return round(score, 2)


def load_models(mdl_names, out='dic', skiprows=0, rows=None):
  processes = []
  mp_pool = mp.Pool(pool_size)
  models = {}
  for idx, (mname, w) in enumerate(mdl_names):
    p = mp_pool.apply_async(csv2df, (mname, out, skiprows, rows))
    processes.append([p, idx])
  # mp_pool.close()
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
    # ---------- [corr_matrix] ----------------------------------------
    # [None 0.86 0.85 0.84 0.84 0.82 0.78 0.79 0.82 0.78 0.82 0.79]
    # [None None 0.74 0.75 0.75 0.89 0.78 0.83 0.81 0.69 0.79 0.78]
    # [None None None 0.92 0.83 0.73 0.68 0.68 0.72 0.81 0.72 0.71]
    # [None None None None 0.82 0.73 0.68 0.69 0.72 0.81 0.73 0.72]
    # [None None None None None 0.73 0.68 0.69 0.73 0.75 0.74 0.72]
    # [None None None None None None 0.76 0.79 0.74 0.68 0.72 0.72]
    # [None None None None None None None 0.83 0.71 0.63 0.71 0.7]
    # [None None None None None None None None 0.72 0.64 0.71 0.7]
    # [None None None None None None None None None 0.66 0.8 0.86]
    # [None None None None None None None None None None 0.67 0.66]
    # [None None None None None None None None None None None 0.82]
    # [None None None None None None None None None None None None]


    self.mdl_names = [
      #-----[God]-----
      # ('blending_20160706_142342_0.59864.csv.gz'            , 1.5 ),
      ('blending_20160703_094554_0.59140.csv.gz'            , 1.2 ),
      ('skrf_cfeats_800tree_mf0.5_0.59121.csv.gz'           , 0.6 ),
      ('skrf_cfeats_1500tree_m_split_7.csv.gz'              , 0.6 ),

      ('skrfp_skrfp_cfeats_0.58921.csv.gz'                  , 1.0 ),
      ('knn2_blended_20160705_144539_0.58692.csv.gz'        , 0.8 ),
      # ('KNN_submission_sorted_0.58370.csv.gz'               , 1.0 ),
      ('cKNN_20160630_214715_0.58273.csv.gz'                , 1.0 ),

      ('treva_partial_20160629_all_blended_0.57968.csv.gz'  , 0.8 ),
      ('skrf_cfeats_0705_0.57809.csv.gz'                    , 0.8 ),
      ('treva_all_xstep0.6_elite_20160630_0.57802.csv.gz'   , 0.8 ),
      # ('treva_6trees_20160629_0.57756.csv.gz'               , 0.5 ),
      
      #-----[King]-----
      # ('blending_20160703_094554_0.59140.csv.gz'            , 1.0 ),
      # ('blending_20160630_0.58772.csv.gz'                   , 5.0 ),

      # ('blending_20160626_094932_0.58702.csv.gz'            , 4.0 ),
      # ('blending_20160621_214954_0.58657.csv.gz'            , 3.0 ),
      # ('KNN2_20160703_075345_0.58649.csv.gz'                , 3.0 ),  
      # ('lb_marsan_blending_0622_0.58569.csv.gz'             , 2.0 ),
      # ('blending_gs_top_w2_20160619_180546_0.58529.csv.gz'  , 2.0 ),
      # ('lb_blending_20160617_215629_0.58463.csv.gz'         , 2.0 ),
      # ('lb_marsan_blending_0614_0.58378.csv.gz'             , 2.0 ),
      # ('lb_marsan_blending_0613_0.58299.csv.gz'             , 2.0 ),
      
      #-----[Lord]-----
      # ('lb_ldsc_0.58143.csv.gz'                             , 2.0 ),
      # ('lb_ravi_0.58081.csv.gz'                             , 1.8 ),
      # ('lb_jesting_0.58067.csv.gz'                          , 1.8 ),
      # ('lb_anouymous_0.58018.csv.gz'                        , 1.8 ),
      # ('lb_anouymous_0.57842.csv.gz'                        , 1.5 ),
      # ('lb_hamed_0.57946.csv.gz'                            , 1.2 ),
      # ('treva_full10_20160702_154515_0.57686.csv.gz'        , 0.5 ),


      #-----[RedGuard]-----
      # ('treva_elite_20160626_f0.5d11_0.57475.csv.gz'        , 1.5 ),
      # ('skrf_submit_20160623_224407_0.57349.csv.gz'         , 1.5 ),
      # ('lb_sub_knn_daten-kieker_0.57189.csv.gz'             , 1.2 ),
      # ('treva_elite_20160627_f0.5d15_0.57169.csv.gz'        , 1.2 ),
      # ('treva_full10_x8_20160702_141130_0.57157.csv.gz'     , 1.2 ),
      # ('skrf_submit_full_20160620_000902_0.57114.csv.gz'    , 1.2 ),
      # ('lb_sub_knn_danielspringt_0.57068.csv.gz'            , 1.2 ),
      # ('skrfp_submit_full_20160621_0.57054.csv.gz'          , 1.2 ),
      # ('lb_daniel_0.57068.csv.gz'                           , 1.0 ),
      # ('lb_grid_knn_lonely_shepard_0.57004.csv.gz'          , 1.0 ),
      # ('knn_grid_0.8_20160618_081945_0.56999.csv.gz'        , 1.0 ),
      # ('skrf_submit_full_20160621_234034_0.56992.csv.gz'    , 1.0 ),
      
      # #-----[Knight]-----
      # ('treva_20160630_181344_blended_0.56942.csv.gz'       , 0.9 ),
      # ('treva_submit_20160625_105356_0.56680.csv.gz'        , 0.9 ),
      # ('submit_knn_0.4_grid_20160617_091633_0.56919.csv.gz' , 0.9 ),
      # ('submit_knn_submit_20160615_230955_0.56815.csv.gz'   , 0.8 ),
      # ('submit_skrf_submit_20160605_195424_0.56552.csv.gz'  , 0.7 ),
      # ('submit_skrf_submit_20160608_174129_0.56533.csv.gz'  , 0.7 ),
      # ('submit_sket_20160622_130909_0.56315.csv.gz'         , 0.7 ),
      # ('sketp_submit_full_20160622_180604_0.56173.csv.gz'   , 0.7 ),
      # ('submit_skrf_submit_20160604_171454_0.56130.csv.gz'  , 0.5 ),
      # ('submit_skrf_submit_20160602_104038_0.56093.csv.gz'  , 0.7 ),
      # ('submit_sketp_20160622_180604_0.56082.csv.gz'        , 0.7 ),
      # ('xgb_submit_full_20160620_180926_0.56032.csv.gz'     , 0.7 ),
      # ('submit_knn_inter_20160616_172918_0.55919.csv.gz'    , 0.6 ),
      # ('xgb_submit_full_20160616_0.55615.csv.gz'            , 0.6 ),
      # ('submit_skrf_submit_20160612_214750_0.55583.csv.gz'  , 0.5 ),
      # ('submit_xgb_submit_20160604_173333_0.55361.csv.gz'   , 0.5 ),
      
      #-----[Ash]-----
      # ('submit_skrf_submit_20160530_155226_0.53946.csv.gz', 0.2 ),
      # ('skrfp_submit_treva_20160624_163552_0.53893.csv.gz', 0.2 ),
      # ('submit_skrf_submit_20160530_143859_0.52721.csv.gz', 0.2 ),
      # ('submit_skrf_submit_20160611_233627_0.52375.csv.gz', 0.2 ),
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
    elif cmd == 'gs_top_n':
      self.init_models()
      all_mdl_names = [(k, 1) for k,v in self.mdl_names]
      for n in [8, 15, 30]:
        print("[gs_top_n] n=%i" % n)
        self.mdl_names = all_mdl_names[:n]
        self.launch()
        submiter.submiter().submit(entry=self.output_fname, message=self.mdl_names)
      return
    #---------------------------------------------
    elif cmd == 'debug':
      self.do_corr_rows = 100000
      self.do_blend_rows = 100000
      self.launch()
    elif cmd == 'average':
      self.init_models()
      self.mdl_names = [(k, 1) for k,v in self.mdl_names]
      self.launch()
    elif cmd == 'average_but_top':
      self.init_models()
      self.mdl_names = [((k, 1) if idx > 0 else (k, 2)) for idx, (k,v) in enumerate(self.mdl_names)]
      self.launch()
    else:
      self.launch()
    # auto-submit
    if self.do_upload:
      submiter.submiter().submit(entry=self.output_fname, message=self.mdl_names)
    


#===========================================
#   Main Flow
#===========================================
if __name__ == '__main__':
  # blendor(do_blend=True, do_upload=True).run('gs_top_n')
  # blendor(do_blend=False).run('debug')   # cal corr only
  blendor(do_blend=True, do_upload=True).run('average_but_top')
  # blendor(do_blend=True, do_upload=True).run('average')
  # blendor(do_blend=True, do_upload=True).run()
  # bla.run()
  # bla.run('submit')
  # bla.run('gs_top_w')
  # bla.run('gs_rank_w')

