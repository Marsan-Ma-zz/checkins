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

  def __init__(self, root='.', valid_ratio=0.2):
    self.root = root
    self.valid_ratio = valid_ratio


  def cmd_parse(self, argv):
    if (len(argv) < 3): 
      print(''' 
        [Usage] 
          1. train model with sample rate: 
            python3 main.py skrf 1 1.0
          2. compare all models
            python3 main.py all 1 1.0
      ''')
      sys.exit()
    cmds = {idx: c for idx, c in enumerate(sys.argv)}
    self.cmd_set(
      alg     = cmds.get(1), 
      srate   = float(cmds.get(2)),
      size    = float(cmds.get(3)), 
      step    = float(cmds.get(4)),
    )
    

  # for ipython notebook
  def cmd_set(self, alg, srate, size, step):
    self.alg    = alg
    self.srate  = srate
    self.size   = size
    self.step   = step


  def init_team(self):
    x_ranges = conv.get_range(self.size, self.step)
    y_ranges = conv.get_range(self.size, self.step)
    self.stamp = self.alg + '_' + str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    self.pas = parser.parser(size=self.size, root=self.root)
    self.tra = trainer.trainer(stamp=self.stamp, size=self.size, root=self.root, x_ranges=x_ranges, y_ranges=y_ranges)
    self.eva = evaluator.evaluator(stamp=self.stamp, size=self.size, root=self.root, x_ranges=x_ranges, y_ranges=y_ranges)
    print("[Start] srate=%.2f, size=%.2f, step=%.2f, stamp=%s @ %s" % (self.srate, self.size, self.step, self.stamp, datetime.now()))


  def run(self):
    #------------------------------------------
    if self.alg == 'all':
      print("[Alg]: compare all algorithms")
      for a in ['skrf', 'xgb', 'sklr']:
        self.init_team()
        self.train_alg(a)
    #------------------------------------------
    elif 'grid_step' in self.alg:
      alg = self.alg.split('_')[0]
      print("[Alg]: Grid Search for %s" % alg)
      for step in [0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.8, 1]:
        print("=====[%s for step=%.2f]=====" % (self.alg, step))
        self.size = 1
        self.step = step
        self.init_team()
        self.train_alg(alg)
    #------------------------------------------
    elif 'current_best_submit':
      self.alg = 'skrf'
      print("[Alg]: %s" % self.alg)
      self.init_team()
      self.train_alg(self.alg, keep_model=True, submit=True)
    #------------------------------------------
    else:
      print("[Alg]: %s" % self.alg)
      self.init_team()
      self.train_alg(self.alg)
    #------------------------------------------
    


  #----------------------------------------
  #   Main
  #----------------------------------------
  def train_alg(self, alg, keep_model=False, submit=False):
    # get data
    start_time = time.time()
    df_train, df_valid, df_test = self.pas.get_data(sample_rate=self.srate)
      
    # train & test
    self.tra.train(df_train, alg=alg)
    _, valid_score = self.eva.evaluate(df_valid)
    
    # save & clear
    if not keep_model:
      self.eva.clear_meta_files()
    if submit:
      preds_total, _ = self.eva.evaluate(df_test)
      self.eva.save_and_clear(preds_total, valid_score, stamp="%s_%s" % (alg, self.stamp))
    print("[Finished!] Elapsed time overall for %.2f secs" % (time.time() - start_time))



#===========================================
#   Main Flow
#===========================================
if __name__ == '__main__':
  m = main()
  m.cmd_parse(sys.argv)
  m.run()

