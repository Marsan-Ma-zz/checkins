import os, sys, time, pickle
import pandas as pd
import numpy as np
import multiprocessing as mp
pool_size = mp.cpu_count()

from random import random
from datetime import datetime
from collections import Counter

from lib import conventions as conv
from lib import evaluator, parser, trainer

#===========================================
#   Main Flow
#===========================================
class main(object):

  def __init__(self, root='.', params={}):
    self.timestamp = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    self.all_feats = ['x','y','accuracy', 'hour', 'qday', 'weekday', 'month', 'year']
    self.params = {
      'root'            : root,
      'x_cols'          : self.all_feats,
      'location_shift'  : 2.0,
      'location_cache'  : "./data/cache/location_est.pkl",
      #-----[best parameters]-----
      'x_step'          : 0.04,
      'y_step'          : 0.10,
      #-----[data engineering in parser]-----
      'train_test_split_time'   : 100000,   # confirmed!
      'place_min_last_checkin'  : 600000,   # for submit 
      # 'train_min_time'          : 300000,   # for submit (not good, no use!)
      # 'place_max_first_checkin' : 300000,   # for valid only, not for submit!
      # 'train_max_time'          : 500000,   # for valid only, not for submit!
      #-----[post-processing]-----
      'mdl_weights'             : (0.4, 1.0, 0.4),  # good, could probe further!
      'time_th'                 : 0.003,
      'loc_th_x'                : 1.2,
      'loc_th_y'                : 2.2,
    }
    for k,v in params.items(): self.params[k] = v   # overwrite if setup
    # extra info

    

  def cmd_parse(self, argv):
    if (len(argv) < 3): 
      print(''' 
        [Usage] 
          1. train model: 
            python3 main.py skrf 1.0 0.05
          2. compare all models
            python3 main.py all 1.0 0.05
      ''')
      sys.exit()
    cmds = {idx: c for idx, c in enumerate(sys.argv)}
    self.params['alg']    = cmds.get(1)
    self.params['size']   = float(cmds.get(2))
    self.params['stamp']  = cmds.get(3)

    

  def init_team(self):
    self.params['stamp'] = self.params.get('stamp') or "%s_%s" % (self.params['alg'], self.timestamp)
    self.pas = parser.parser(self.params)
    self.tra = trainer.trainer(self.params)
    self.eva = evaluator.evaluator(self.params)
    print("=====[Start] @ %s=====" % (datetime.now()))
    for k, v in self.params.items(): print("%s = %s" % (k,v))
    print("="*50)
    

  def run(self):
    run_cmd = self.params['alg']
    alg = run_cmd.split('_')[0]
    print("[RUN_CMD] %s" % run_cmd)
    #------------------------------------------
    if run_cmd == 'all':
      for a in ['skrf', 'xgb', 'sklr']:
        self.init_team()
        self.train_alg(a)
    #------------------------------------------
    elif 'skrf_grid_step' in run_cmd:
      for x_step in [0.04, 0.05, 0.1, 0.2]:
        for y_step in [0.04, 0.05, 0.1, 0.2]:
          print("=====[%s for step=(%.2f, %.2f)]=====" % (run_cmd, x_step, y_step))
          self.params['size']   = 1
          self.params['x_step'] = x_step
          self.params['y_step'] = y_step
          self.init_team()
          self.train_alg(alg)
    #------------------------------------------
    elif run_cmd == 'skrf_feats_sel':
      all_feats = self.all_feats
      # baseline
      self.params['x_cols'] = all_feats
      self.init_team()
      self.train_alg(alg)
      # drop 1 feature
      for af in all_feats:
        self.params['x_cols'] = [a for a in all_feats if a != af]
        self.init_team()
        self.train_alg(alg)
    #------------------------------------------
    elif run_cmd == 'skrf_gs_time':
      for mfc in [None, 200000, 250000, 300000]:
        for tmt in [None]: #, 400000, 500000, 600000, 700000]:
          self.params['place_max_first_checkin'] = mfc
          self.params['train_max_time'] = tmt
          self.init_team()
          self.train_alg(alg)
    #------------------------------------------
    elif run_cmd == 'skrf_gs_loc_th':
      for th_y in np.arange(2, 2.4, 0.1):
        for th_x in np.arange(0.6, 2, 0.2):
          print("[SKRF_GS_LOC_TH]: th_x=%s, th_y=%s" % (th_x, th_y))
          self.params['loc_th_x'] = th_x
          self.params['loc_th_y'] = th_y
          self.init_team()
          self.evaluate_model(evaluate=True, submit=False)
    #------------------------------------------
    elif run_cmd == 'skrf_gs_time_th':
      for pth in np.arange(0, 0.005, 0.001):
        self.params['time_th'] = pth
        self.init_team()
        self.evaluate_model(evaluate=True, submit=False)
    #------------------------------------------
    elif run_cmd == 'skrf_gs_params':
      self.init_team()
      for n_estimators in [200, 300, 500, 800, 1000]:
        for max_depth in [12]:
          self.train_alg(alg, params={'n_estimators': n_estimators, 'max_depth': max_depth})
      #
      self.params['size']   = 10.0
      self.params['stamp']  = "%s_%s" % (self.params['alg'], self.timestamp)
      self.init_team()
      self.train_alg(alg, keep_model=True, submit=True)
      self.tra, self.eva, self.pas = None, None, None
    elif run_cmd == 'xgb_gs_params':
      self.init_team()
      for n_estimators in [50, 100, 200, 300, 500]:
        for max_depth in [12]:
          for learning_rate in [0.15]:
            self.train_alg(alg, params={'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate})
      #
      self.params['size']   = 10.0
      self.params['stamp']  = "%s_%s" % (self.params['alg'], self.timestamp)
      self.init_team()
      self.train_alg(alg, keep_model=True, submit=True)
    #------------------------------------------
    elif 'submit' in run_cmd:
      self.init_team()
      self.train_alg(alg, keep_model=True, submit=True)
    elif 'eva_exist' in run_cmd:
      self.init_team()
      self.evaluate_model(evaluate=True, submit=False)
    elif 'smt_exist' in run_cmd:
      self.init_team()
      self.evaluate_model(evaluate=False, submit=True)
    #------------------------------------------
    elif run_cmd == 'parse_extra_info':
      self.pas = parser.parser(self.params)
      df_train, df_valid, df_test = self.pas.get_data()
      self.init_location_table(pd.concat([df_train, df_valid]))
    #------------------------------------------
    else: # single model
      self.init_team()
      self.train_alg(run_cmd)
    #------------------------------------------
    


  #----------------------------------------
  #   Main
  #----------------------------------------
  def train_alg(self, alg, keep_model=False, submit=False, params={}):
    # get data
    start_time = time.time()
    df_train, df_valid, df_test = self.pas.get_data()
      
    # train & test
    print("[train_alg]: %s" % params)
    self.tra.train(df_train, alg=alg, params=params)
    if self.params['size'] <= 1:  # eva.train only when dev.
      _, train_score = self.eva.evaluate(df_train, title='Eva.Train')
    _, valid_score = self.eva.evaluate(df_valid, title='Eva.Test')
    
    # save & clear
    if not keep_model:
      self.eva.clear_meta_files()
    if submit:
      preds_total, _ = self.eva.evaluate(df_test, title='Submit')
      self.eva.gen_submit_file(preds_total, valid_score)
    print("[Finished!] Elapsed time overall for %.2f secs" % (time.time() - start_time))


  def init_location_table(self, df):
    # location estimation
    stat_mean = df[['place_id', 'x', 'y']].groupby('place_id').mean()
    stat_mean = stat_mean.rename(columns = {'x':'x_mean', 'y': 'y_mean'})
    stat_mean.reset_index(inplace=True)
    stat_std = df[['place_id', 'x', 'y']].groupby('place_id').std()
    stat_std = stat_std.rename(columns = {'x':'x_std', 'y': 'y_std'})
    stat_std.reset_index(inplace=True)
    stat_loc = stat_mean.merge(stat_std, on='place_id')
    stat_loc['x_min'] = stat_loc.x_mean - 2*stat_loc.x_std
    stat_loc['x_max'] = stat_loc.x_mean + 2*stat_loc.x_std
    stat_loc['y_min'] = stat_loc.y_mean - 2*stat_loc.y_std
    stat_loc['y_max'] = stat_loc.y_mean + 2*stat_loc.y_std
    # available time estimation
    avail_hours = df.groupby('place_id').hour.apply(lambda x: Counter(x))
    avail_hours = avail_hours.to_dict()
    sum_hours   = df.place_id.value_counts()
    stat_hours  = {(pid, i): v/sum_hours[pid] for (pid, i), v in avail_hours.items()}

    avail_wdays = df.groupby('place_id').weekday.apply(lambda x: Counter(x))
    avail_wdays = avail_wdays.to_dict()
    sum_wdays   = df.place_id.value_counts()
    stat_wdays  = {(pid, i): v/sum_wdays[pid] for (pid, i), v in avail_wdays.items()}

    pickle.dump([stat_loc, stat_wdays, stat_hours], open(self.params['location_cache'], 'wb'))
    print("location cache written in %s" % self.params['location_cache'])

  # skip training, evaluate from existing model
  def evaluate_model(self, evaluate=False, submit=False):
    print("[Evaluate_model] with params=%s" % (self.params))
    start_time = time.time()
    df_train, df_valid, df_test = self.pas.get_data()
    valid_score = 0.0
    if evaluate:
      _, valid_score = self.eva.evaluate(df_valid, title='Test')
    if submit:
      preds_total, _ = self.eva.evaluate(df_test, title='Submit')
      self.eva.gen_submit_file(preds_total, valid_score)
    print("[Finished!] evaluate_model for %.2f secs" % (time.time() - start_time))


#===========================================
#   Main Flow
#===========================================
if __name__ == '__main__':
  m = main()
  m.cmd_parse(sys.argv)
  m.run()

