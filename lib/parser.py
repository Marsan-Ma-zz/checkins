import os, sys, datetime, time, pickle
import pandas as pd
import numpy as np
import multiprocessing as mp
pool_size = mp.cpu_count()

from sklearn.cross_validation import train_test_split

from lib import conventions as conv

#===========================================
#   Evaluator
#===========================================  
class parser(object):

  def __init__(self, size, root, valid_ratio=0.2):
    self.size = size
    self.root = root
    self.valid_ratio = valid_ratio

  #----------------------------------------
  #   Main
  #----------------------------------------
  def get_data(self, sample_rate, overwrite=False):
    start_time = time.time()
    cache_name = "%s/data/cache/cache_get_data_sz%s_sr%s_va%s.pkl" % (self.root, self.size, sample_rate, self.valid_ratio)
    if (os.path.exists(cache_name) and not overwrite):
      df_train, df_valid, df_test = pickle.load(open(cache_name, 'rb'))
      print("[get_samples] from cache: df_train = %s, df_valid = %s, df_test = %s, %.2f secs" % (df_train.shape, df_valid.shape, df_test.shape, time.time() - start_time))
    else:
      df_train = self.parse_data('%s/data/train.csv.zip' % self.root, sample_rate)
      df_test = self.parse_data('%s/data/test.csv.zip' % self.root, sample_rate)
      df_train, df_valid = train_test_split(df_train, test_size=self.valid_ratio)
      pickle.dump([df_train, df_valid, df_test], open(cache_name, 'wb'))
      print("[get_samples] final: train = %s, valid = %s, test = %s, %.2f secs" % (df_train.shape, df_valid.shape, df_test.shape, time.time() - start_time))
    return df_train, df_valid, df_test


  #----------------------------------------
  #   Tasks
  #----------------------------------------
  def parse_data(self, fname, sample_rate):
    df = pd.read_csv(fname)
    if sample_rate < 1: df = self.sampling(df, sample_rate)
    df = self.feature_engineering(df)
    return df


  def sampling(self, df, sample_rate):
    df_size = df.count().max()
    df = df.sample(int(df_size*sample_rate))
    df = df[(df.x >= 0) & (df.x <= self.size) & (df.y >= 0) & (df.y <= self.size)]
    return df


  def feature_engineering(self, df):
    df['hour'] = (df['time']//60)%24+1 # 1 to 24
    df['weekday'] = (df['time']//1440)%7+1
    df['month'] = (df['time']//43200)%12+1 # rough estimate, month = 30 days
    df['year'] = (df['time']//525600)+1
    return df


