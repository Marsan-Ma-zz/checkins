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

def df2sample(df, x_cols):
  row_id = df['row_id'].reset_index(drop=True)
  x = df[x_cols]
  y = None if ('place_id' not in df.columns) else df[['place_id']].values.ravel()
  return x, y, row_id


def get_range(size, step, interleave):
  return list(zip(np.arange(0, size, step/interleave), np.arange(step, step + size, step/interleave)))


def df_preprocess(mode, df_grid, x_idx, y_idx, LOCATION, AVAIL_WDAYS, AVAIL_HOURS, POPULAR, GRID_CANDS):
  start_time = time.time()
  cand_ids = GRID_CANDS[(x_idx, y_idx)]
  cols_origin = set(df_grid.columns)
  # ----- LOCATION -----
  cands_loc = LOCATION[LOCATION.place_id.isin(cand_ids)]
  for i, xm, ym in (cands_loc[['place_id', 'x_mean', 'y_mean']].values):
    if 'X' in mode:
      df_grid.loc[:, "dist_x_%i" % i] = abs(df_grid.x - xm)
    if 'Y' in mode:
      df_grid.loc[:, "dist_y_%i" % i] = abs(df_grid.y - ym)
  # ----- AVAIL_WDAYS -----
  popu = POPULAR.get((x_idx, y_idx))
  for c in cand_ids:
    if 'W' in mode:
      df_grid.loc[:, "avail_wdays_%s" % c] = [AVAIL_WDAYS.get((c, v), 0) for v in df_grid.weekday.values]
    if 'H' in mode:
      df_grid.loc[:, "avail_hours_%s" % c] = [AVAIL_HOURS.get((c, v), 0) for v in df_grid.hour.values]
  # print("preprocessed (%i, %i) df_grid.shape=%s" % (x_idx, y_idx, len(df_grid.columns)))
  cols_extra = list(set(df_grid.columns) - cols_origin)
  if (x_idx + y_idx == 0): 
    print("[df_preprocess] cols_extra = %s, cost %i secs" % (cols_extra, (time.time() - start_time)))
    pickle.dump(df_grid, open("./data/test_grid.pkl", 'wb'))
  return df_grid, cols_extra

