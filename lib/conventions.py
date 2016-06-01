import os, sys, datetime, time, pickle
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

#===========================================
#   Convension tasks
#===========================================  
ts_today = int(datetime.now().timestamp())
ts_hour = 60*60
ts_day = 24*ts_hour

def now(fmt='str', offset=0):
  if fmt == 'int':
    return time.time() + offset*ts_day
  elif fmt == 'str':
    return (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d")
  elif fmt == 'full':
    return (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d %H:%M:%S")

def trim_range(mi, ma, size):
  mi, ma = round(mi, 4), round(ma, 4)
  if ma == size: ma = ma + 0.001
  return mi, ma

def df2sample(df, x_min, x_max, y_min, y_max, x_cols):
  if x_max is None:
    df = df[(df.y >= y_min) & (df.y < y_max)]
  elif y_max is None:
    df = df[(df.x >= x_min) & (df.x < x_max)]
  else:
    df = df[(df.x >= x_min) & (df.x < x_max) & (df.y >= y_min) & (df.y < y_max)]
  row_id = df['row_id'].reset_index(drop=True)
  x = df[x_cols]
  y = None if ('place_id' not in df.columns) else df[['place_id']].values.ravel()
  return x, y, row_id


def get_range(size, step):
  return list(zip(np.arange(0, size, step), np.arange(step, step + size, step)))

