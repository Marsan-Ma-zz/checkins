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

  def __init__(self, params):
    print("params", params)
    self.size = params['size']
    self.root = params['root']
    self.sample_rate = 1.0 # this datasets not good for sampling, always use 100% samples.

    # [data engineering]
    # time
    self.train_test_split_time = params['train_test_split_time']
    self.train_max_time = params.get('train_max_time')
    self.train_min_time = params.get('train_min_time')
    # place_id
    self.place_min_checkin = params.get('place_min_checkin', 0)
    self.place_min_last_checkin = params.get('place_min_last_checkin')
    self.place_max_first_checkin = params.get('place_max_first_checkin')


  #----------------------------------------
  #   Main
  #----------------------------------------
  def get_data(self, overwrite=False):
    start_time = time.time()
    cache_name = "%s/data/cache/cache_get_data_split_%i.pkl" % (self.root, self.train_test_split_time)
    if (os.path.exists(cache_name) and not overwrite):
      df_train, df_valid, df_test = pickle.load(open(cache_name, 'rb'))
      print("[get_samples] from cache: df_train = %s, df_valid = %s, df_test = %s, %.2f secs" % (df_train.shape, df_valid.shape, df_test.shape, time.time() - start_time))
    else:
      df_train = self.parse_data('%s/data/train.csv.zip' % self.root)
      df_test = self.parse_data('%s/data/test.csv.zip' % self.root)
      # divide train/valid by time
      df_valid = df_train[df_train.time >= self.train_test_split_time]
      df_train = df_train[df_train.time < self.train_test_split_time]
      pickle.dump([df_train, df_valid, df_test], open(cache_name, 'wb'))
      print("[get_samples] final: train = %s, valid = %s, test = %s, %.2f secs" % (df_train.shape, df_valid.shape, df_test.shape, time.time() - start_time))

    # filter dead place_ids
    if self.place_min_checkin:
      place_cnt = df_train.place_id.value_counts()
      cold_places = place_cnt[place_cnt < self.place_min_checkin].index
      df_train = df_train[~df_train.place_id.isin(cold_places)]
    if self.place_min_last_checkin: 
      df_train = self.filter_min_last_checkin(df_train, th=self.place_min_last_checkin)
    if self.place_max_first_checkin:
      df_train = self.filter_max_first_checkin(df_train, th=self.place_max_first_checkin)
    if self.train_min_time: 
      df_train = df_train[df_train.time >= self.train_min_time]
    if self.train_max_time: 
      df_train = df_train[df_train.time <= self.train_max_time]  
    self.show_info(df_train, df_valid, df_test)
    print("[get_samples] after data engineeing: train = %s, valid = %s, test = %s" % (df_train.shape, df_valid.shape, df_test.shape))
    return df_train, df_valid, df_test


  def show_info(self, df_train, df_valid, df_test):
    if len(df_valid) > 0: 
      print("df_valid time: %i - %i, %i samples" % (df_valid.time.min(), df_valid.time.max(), len(df_valid)))
    print("df_train time: %i - %i, %i samples" % (df_train.time.min(), df_train.time.max(), len(df_train)))
    print("df_test time: %i - %i, %i samples" % (df_test.time.min(), df_test.time.max(), len(df_test)))


  def filter_min_last_checkin(self, df_train, th):   # for submit
    raw_train_cnt = len(df_train)
    last_checkin = df_train[["time", "place_id"]].groupby("place_id").max()
    stopped_place_ids = last_checkin[last_checkin.time < th].index
    df_train = df_train[~df_train.place_id.isin(stopped_place_ids)]
    print("[filter_places] stopped place: %i/%i, alive samples %i/%i" % (len(stopped_place_ids), len(last_checkin), len(df_train), raw_train_cnt))
    return df_train

  # [WARN!] test only, not for submit
  def filter_max_first_checkin(self, df_train, th):   # for valid
    raw_train_cnt = len(df_train)
    first_checkin = df_train[["time", "place_id"]].groupby("place_id").min()
    stopped_place_ids = first_checkin[first_checkin.time > th].index
    df_train = df_train[~df_train.place_id.isin(stopped_place_ids)]
    print("[filter_places] stopped place: %i/%i, alive samples %i/%i" % (len(stopped_place_ids), len(first_checkin), len(df_train), raw_train_cnt))
    return df_train

  #----------------------------------------
  #   Tasks
  #----------------------------------------
  def parse_data(self, fname):
    df = pd.read_csv(fname)
    if self.sample_rate < 1: df = self.sampling(df, sample_rate)
    df = self.feature_engineering(df)
    return df


  def sampling(self, df, sample_rate):
    df_size = df.count().max()
    df = df.sample(int(df_size*sample_rate))
    df = df[(df.x >= 0) & (df.x <= self.size) & (df.y >= 0) & (df.y <= self.size)]
    return df


  def feature_engineering(self, df):
    df['hour']    = (df['time']//60)%24+1 # 1 to 24
    df['hour2']   = (df['time']//60)%24//2+1 # 1 to 12
    df['hour3']   = (df['time']//60)%24//3+1 # 1 to 8
    df['hour4']   = (df['time']//60)%24//4+1 # 1 to 6
    df['qday']    = (df['time']//60)%24//6+1 # 1 to 4
    df['weekday'] = (df['time']//1440)%7+1
    df['month']   = (df['time']//43200)%12+1 # rough estimate, month = 30 days
    df['month2']   = (df['time']//43200)%12//2+1
    df['month3']   = (df['time']//43200)%12//3+1
    df['month6']   = (df['time']//43200)%12//6+1
    df['year']    = (df['time']//525600)+1
    return df


