import os, sys, datetime, time, pickle
import pandas as pd
import numpy as np


#===========================================
#   Convension tasks
#===========================================  
def trim_range(mi, ma, size):
  mi, ma = round(mi, 4), round(ma, 4)
  if ma == size: ma = ma + 0.001
  return mi, ma

def df2sample(df, x_min, x_max, y_min, y_max):
  df = df[(df.x >= x_min) & (df.x < x_max) & (df.y >= y_min) & (df.y < y_max)]
  row_id = df['row_id'].reset_index(drop=True)
  x = df[['x','y','accuracy','time', 'hour', 'weekday', 'month', 'year']]
  y = None if 'place_id' not in df.columns else df[['place_id']].values.ravel()
  return x, y, row_id


def get_range(size, step):
  return list(zip(np.arange(0, size, step), np.arange(step, step + size, step)))

