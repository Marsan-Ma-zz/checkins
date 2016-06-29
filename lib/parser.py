import os, sys, time, pickle
import pandas as pd
import numpy as np
import multiprocessing as mp
pool_size = mp.cpu_count()

from datetime import datetime
from collections import Counter

from lib import conventions as conv
from lib import grouper

#===========================================
#   Evaluator
#===========================================  
class parser(object):

  def __init__(self, params):
    self.size = params['size']
    self.root = params['root']
    self.sample_rate = 1.0 # this datasets not good for sampling, always use 100% samples.
    self.x_cols = params['x_cols']

    # [data engineering]
    # time
    self.train_test_split_time = params['train_test_split_time']
    self.train_max_time = params.get('train_max_time')
    self.train_min_time = params.get('train_min_time')
    # place_id
    self.place_min_checkin = params.get('place_min_checkin', 0)
    self.place_min_last_checkin = params.get('place_min_last_checkin')
    self.place_max_first_checkin = params.get('place_max_first_checkin')
    # ctrl
    self.remove_distance_outlier = params['remove_distance_outlier']
    # special
    self.grp = grouper.grouper(params)

  #----------------------------------------
  #   Main
  #----------------------------------------
  def get_data(self, overwrite=False):
    start_time = time.time()
    cache_name = "%s/data/cache/cache_get_data_split_%i_size_%.2f_rmol_%.2f_mci_%i.pkl" % (self.root, self.train_test_split_time, self.size, self.remove_distance_outlier, self.place_min_checkin)
    if (os.path.exists(cache_name) and not overwrite):
      df_train, df_valid, df_test = pickle.load(open(cache_name, 'rb'))
      print("[get_samples] from cache: df_train = %s, df_valid = %s, df_test = %s, %.2f secs" % (df_train.shape, df_valid.shape, df_test.shape, time.time() - start_time))
    else:
      df_train = self.parse_data('%s/data/train.csv.zip' % self.root)
      df_test = self.parse_data('%s/data/test.csv.zip' % self.root)
      df_train = df_train[(df_train.x <= self.size) & (df_train.y <= self.size)]
      df_test = df_test[(df_test.x <= self.size) & (df_test.y <= self.size)]
      print("[get_samples] from csv: train = %s, test = %s" % (df_train.shape, df_test.shape))
      # divide train/valid by time
      if self.train_test_split_time > 300000:
        df_valid = df_train[df_train.time > self.train_test_split_time]
        df_train = df_train[df_train.time <= self.train_test_split_time]
      else: # skrf_reverse_valid_split_time
        df_valid = df_train[df_train.time <= self.train_test_split_time]
        df_train = df_train[df_train.time > self.train_test_split_time]
      #-----[Fixed data engineering]-----
      if self.place_min_checkin:
        place_cnt = df_train.place_id.value_counts()
        cold_places = place_cnt[place_cnt < self.place_min_checkin].index
        df_train = df_train[~df_train.place_id.isin(cold_places)]
      
      if self.remove_distance_outlier:
        stat_loc = self.location_estimation(df_train)
        stat_loc = dict(zip(stat_loc.place_id, stat_loc[['x_min', 'x_max', 'y_min', 'y_max']].to_dict('records')))
        outliers = []
        for rid, pid, xv, yv in df_train[['row_id', 'place_id', 'x', 'y']].values.tolist():
          if (xv < stat_loc[pid]['x_min']) | (xv > stat_loc[pid]['x_max']) | (yv < stat_loc[pid]['y_min']) | (yv > stat_loc[pid]['y_max']):
            outliers.append(rid)
          if (rid % 1e7 == 0): print("remove_distance_outlier:", rid, datetime.now())
        df_train = df_train[~df_train.row_id.isin(outliers)]
        print("[remove_distance_outlier] remove %i outliers, len(df_train)=%i @ %s" % (len(outliers), len(df_train), datetime.now()))
      #----------------------------------
      # tsne
      # df_train, df_valid, df_test = self.grp.add_grp(df_train, df_valid, df_test)
      pickle.dump([df_train, df_valid, df_test], open(cache_name, 'wb'))
      print("[get_samples] final: train = %s, valid = %s, test = %s, %.2f secs" % (df_train.shape, df_valid.shape, df_test.shape, time.time() - start_time))
    
    print("data ex: df_train.head():\n", df_train.head())

    # filter dead place_ids
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
    print("df_train time: %i - %i, %i samples" % (df_train.time.min(), df_train.time.max(), len(df_train)))
    if len(df_valid) > 0: 
      print("df_valid time: %i - %i, %i samples" % (df_valid.time.min(), df_valid.time.max(), len(df_valid)))
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
    initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]') 
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(t), 'm') for t in df.time.values)
    
    # hour < day < week < month < year
    df['hour']      = (d_times.hour + d_times.minute/60).astype(float)
    df['weekday']   = d_times.weekday.astype(int)
    df['yearday']   = d_times.dayofyear.astype(int)
    df['month']     = d_times.month.astype(int)
    df['year']      = (d_times.year - 2013).astype(int)

    df['qday']        = ((df.hour)//6).astype(int)
    df['dayofmonth']  = [int(d.strftime("%d")) for d in d_times]
    df['weekofyear']  = (d_times.dayofyear//7).astype(int)
    df['season']      = d_times.quarter.astype(int)
    df['logacc']      = np.log(df.accuracy.values).astype(float)

    # df['hour']      = (df['time']//60) % 24
    # df['qday']      = (df['hour']//6)
    # df['day']       = (df['time']//60//24)
    # df['weekday']   = (df['day'] % 7)
    # df['monthday']  = (df['day'] % 30)
    # df['yearday']   = (df['day'] % 365)
    # df['week']      = (df['day']//7) % 52
    # df['month']     = (df['day']//30) % 12 # rough estimate, month = 30 days
    # df['season']    = (df['month']//4)
    # df['year']      = (df['month']//12)
    # df['logacc']    = np.log(df.accuracy.values).astype(float)

    # df['hour']    = (df['time']//60)%24+1 # 1 to 24
    # df['hour']    = (df['time']/60)%24+1 # 1 to 24
    # df['hour2']   = (df['time']/60)%24%2+1
    # df['hour3']   = (df['time']/60)%24%3+1
    # df['hour4']   = (df['time']/60)%24%4+1
    # df['hour6']   = (df['time']/60)%24%6+1
    # df['qday']    = (df['time']/60)%24/6+1 # 1 to 4
    # df['weekday'] = (df['time']/1440)%7+1
    # df['month']   = (df['time']/43200)%12+1 # rough estimate, month = 30 days
    # df['year']    = (df['time']/525600)+1
    
    # df['day']     = (df['time']/60/24)
    # df['monthday']= (df['day'] % 30)
    # df['season']  = (df['time']/43200/3)%4+1
    # df['logacc']  = np.log(df.accuracy.values).astype(float)

    #-----
    # df['hour']    = (df['time']/60)%24+1 # 1 to 24
    # # df['hour2']   = (df['time']/60)%24%2+1 # 1 to 12
    # # df['hour3']   = (df['time']/60)%24%3+1 # 1 to 8
    # # df['hour4']   = (df['time']/60)%24%4+1 # 1 to 6
    # df['qday']    = (df['time']/60)%24%6+1 # 1 to 4
    # # df['p720']    = (df['time'])%720
    # # df['p1260']   = (df['time'])%1260
    # # df['p1440']   = (df['time'])%1440   # equal to df['hour']
    # # df['p1680']   = (df['time'])%1680
    # df['weekday'] = (df['time']/1440)%7+1
    # df['month']   = (df['time']/43200)%12+1 # rough estimate, month = 30 days
    # df['year']    = (df['time']//525600)+1
    
    # # df['day']     = (df['time']//60//24)
    # # df['monthday']= (df['day'] % 30)
    # # df['season']  = (df['time']//43200//3)%4+1
    # df['logacc']  = np.log(df.accuracy.values).astype(float)
    return df


  #----------------------------------------
  #   PreProcessing
  #----------------------------------------
  def init_data_cache(self, df, params):
    # ----- collect & save -----
    print("[init_data_cache] start @ %s" % conv.now('full'))
    stat_loc        = None #self.location_estimation(df)
    stat_wdays      = self.agg_avail_wdays(df)
    stat_hours      = self.agg_avail_hours(df)
    stat_popular    = None #self.popularity(df, params)
    grid_candidates = None #self.get_grid_candidates(df, stat_loc, params['max_cands'], params)
    # info for post-processing
    pickle.dump([stat_loc, stat_wdays, stat_hours, stat_popular, grid_candidates], open(params['data_cache'], 'wb'))
    print("[init_data_cache] written in %s @ %s" % (params['data_cache'], conv.now('full')))


  # ----- location estimation -----
  def location_estimation(self, df, th_x=3, th_y=2):
    stat_mean = df[['place_id', 'x', 'y']].groupby('place_id').mean()
    stat_mean = stat_mean.rename(columns = {'x':'x_mean', 'y': 'y_mean'})
    stat_mean.reset_index(inplace=True)
    stat_std = df[['place_id', 'x', 'y']].groupby('place_id').std()
    stat_std = stat_std.rename(columns = {'x':'x_std', 'y': 'y_std'})
    stat_std.reset_index(inplace=True)
    stat_loc = stat_mean.merge(stat_std, on='place_id')
    stat_loc['x_min'] = stat_loc.x_mean - th_x*stat_loc.x_std
    stat_loc['x_max'] = stat_loc.x_mean + th_y*stat_loc.x_std
    stat_loc['y_min'] = stat_loc.y_mean - th_x*stat_loc.y_std
    stat_loc['y_max'] = stat_loc.y_mean + th_y*stat_loc.y_std
    return stat_loc

  # ----- available time: hours -----
  def agg_avail_hours(self, df):
    df['hour_int'] = df.hour.astype(int)
    avail_hours = df.groupby('place_id').hour_int.apply(lambda x: Counter(x))
    avail_hours = avail_hours.to_dict()
    sum_hours   = df.place_id.value_counts()
    stat_hours  = {(pid, i): v/sum_hours[pid] for (pid, i), v in avail_hours.items()}
    return stat_hours
  
  # ----- available time: wdays -----
  def agg_avail_wdays(self, df):
    df['weekday_int'] = df.weekday.astype(int)
    avail_wdays = df.groupby('place_id').weekday_int.apply(lambda x: Counter(x))
    avail_wdays = avail_wdays.to_dict()
    sum_wdays   = df.place_id.value_counts()
    stat_wdays  = {(pid, i): v/sum_wdays[pid] for (pid, i), v in avail_wdays.items()}
    return stat_wdays

  # ----- place popularity -----
  def popularity(self, df, params):
    x_ranges = conv.get_range(params['size'], params['x_step'], params['x_inter'])
    y_ranges = conv.get_range(params['size'], params['y_step'], params['y_inter'])
    stat_popular = {}
    for x_idx, (x_min, x_max) in enumerate(x_ranges):
      x_min, x_max = conv.trim_range(x_min, x_max, params['size'])
      df_row = df[(df.x >= x_min) & (df.x < x_max)]
      for y_idx, (y_min, y_max) in enumerate(y_ranges): 
        y_min, y_max = conv.trim_range(y_min, y_max, params['size'])
        df_grid = df_row[(df_row.y >= y_min) & (df_row.y < y_max)]
        total = len(df_grid)
        stat_popular[(x_idx, y_idx)] = {k: v/total for k,v in Counter(df_grid.place_id.values.tolist()).items()}
    return stat_popular

  # ----- grid cadidates -----
  def get_grid_candidates(self, df, stat_loc, max_cands, params):
    x_ranges = conv.get_range(params['size'], params['x_step'], params['x_inter'])
    y_ranges = conv.get_range(params['size'], params['y_step'], params['y_inter'])
    grid_candidates = {}
    for x_idx, (x_min, x_max) in enumerate(x_ranges):
      x_min, x_max = conv.trim_range(x_min, x_max, params['size'])
      df_row = df[(df.x >= x_min) & (df.x < x_max)]
      for y_idx, (y_min, y_max) in enumerate(y_ranges): 
        y_min, y_max = conv.trim_range(y_min, y_max, params['size'])
        df_grid = df_row[(df_row.y >= y_min) & (df_row.y < y_max)]
        # grid_candidates[(x_idx, y_idx)] = df_grid.place_id[df_grid.place_id.value_counts().values > 1].values
        grid_candidates[(x_idx, y_idx)] = df_grid.place_id.value_counts()[:max_cands].index.tolist()
    return grid_candidates
