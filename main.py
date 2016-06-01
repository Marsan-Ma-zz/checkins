import os, sys, time, pickle
import pandas as pd
import numpy as np
import multiprocessing as mp
pool_size = mp.cpu_count()

from random import random
from datetime import datetime
from lib import conventions as conv
from lib import evaluator, parser, trainer

#===========================================
#   Main Flow
#===========================================
class main(object):

  def __init__(self, root='.', params={}):
    self.timestamp = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    self.all_feats = ['x','y','accuracy', 'hour', 'weekday', 'qday', 'month', 'year']
    self.params = {
      'root'    : root,
      'x_cols'  : self.all_feats,
      # data engineering in parser
      'train_test_split_time'   : 150000,   # confirmed!
      'place_min_last_checkin'  : 600000,   # for submit 
      # 'train_min_time'          : 300000,   # for submit 
      # 'place_max_first_checkin' : 300000,   # for valid only, not for submit!
      # 'train_max_time'          : 500000,   # for valid only, not for submit!
    }
    for k,v in params.items(): self.params[k] = v   # overwrite if setup


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
    self.params['alg']  = cmds.get(1)
    self.params['size'] = float(cmds.get(2))
    self.params['step'] = float(cmds.get(3))

    

  def init_team(self):
    self.params['stamp'] = "%s_%s" % (self.params['alg'], self.timestamp)
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
    elif 'grid_step' in run_cmd:
      for step in [0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.8, 1]:
        print("=====[%s for step=%.2f]=====" % (run_cmd, step))
        self.size = 1
        self.step = step
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
    elif run_cmd == 'skrf_gs_params':
      self.init_team()
      for n_estimators in [200, 300, 500]:
        for max_depth in [12, 15, 18, 20]:
          self.train_alg(alg, params={'n_estimators': n_estimators, 'max_depth': max_depth})
    elif run_cmd == 'xgb_gs_params':
      self.init_team()
      for n_estimators in [10, 30, 50, 100, 200, 300, 500]:
        for max_depth in [3, 5, 7, 9, 12, 15, 20]:
          for learning_rate in [0.005, 0.01, 0.05, 0.1, 0.2]:
            self.train_alg(alg, params={'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate})
    #------------------------------------------
    elif 'submit' in run_cmd:
      self.init_team()
      self.train_alg(alg, keep_model=True, submit=True)
    elif 'eva_exist' in run_cmd:
      self.init_team()
      self.evaluate_model(evaluate=False, submit=True)
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
    self.tra.train(df_train, alg=alg, params=params)
    _, train_score = self.eva.evaluate(df_train, title='Eva.Train')
    _, valid_score = self.eva.evaluate(df_valid, title='Eva.Test')
    
    # save & clear
    if not keep_model:
      self.eva.clear_meta_files()
    if submit:
      preds_total, _ = self.eva.evaluate(df_test, title='Submit')
      self.eva.gen_submit_file(preds_total, valid_score)
    print("[Finished!] Elapsed time overall for %.2f secs" % (time.time() - start_time))


  # skip training, evaluate from existing model
  def evaluate_model(self, evaluate=False, submit=False):
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

