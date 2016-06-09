#------------------------------
#   Performance memo
#------------------------------
  [20160601]
    0. skrf tree max_depth: (n_estimators: 200)
        7: 0.6447
        8: 0.6462
        9: 0.6462
        10: 0.6465
        11: 0.6466    <--- BEST!
        12: 0.6462
    1. feature selection
       all x_cols   => 0.6364
        no x        => 0.6178
        no y        => 0.5788   (!)
        no accuracy => 0.6237
        no hour     => 0.6210
        no weekday  => 0.6238
        no qday     => 0.6353
        no month    => 0.6266
        no year     => 0.6174   (!)
       add time     => 0.6330
    3. try predict with models from surrounding blocks => good! improve ~0.01 with 3-block in x-direction!
        => could try more alternates!
        => x(0, 0, 1, 0, 0)          : 0.6393
        => x(0, 0.2, 1, 0.2, 0)      : 0.6401
        => x(0, 0.3, 1, 0.3, 0)      : 0.6418  
        => x(0, 0.4, 1, 0.4, 0)      : 0.6427   <--BEST!
        => x(0, 0.45, 1, 0.45, 0)    : 0.6426   
        => x(0, 0.5, 1, 0.5, 0)      : 0.6423   
        => x(0.3, 0.5, 1, 0.5, 0.3)  : 0.6396
        => y(0, 0.5, 1, 0.5, 0)      : 0.6331
        => y(0, 0.3, 1, 0.3, 0)      : 0.6385
        => y(0, 0.2, 1, 0.2, 0)      : 0.6392
        ------[after fix train/test split]-----
        x(0.0, 1.0, 0.0) 0.5431
        x(0.2, 1.0, 0.2) 0.5431
        x(0.4, 1.0, 0.4) 0.5426
        x(0.6, 1.0, 0.6) 0.5408
        x(0.8, 1.0, 0.8) 0.5372
        x(1.0, 1.0, 1.0) 0.5336 

        x(0.0, 0, 1.0, 0, 0.0) 0.5416 
        x(0.1, 0, 1.0, 0, 0.1) 0.5416 
        x(0.2, 0, 1.0, 0, 0.2) 0.5410 
        x(0.3, 0, 1.0, 0, 0.3) 0.5394 
        x(0.4, 0, 1.0, 0, 0.4) 0.5363 
        x(0.5, 0, 1.0, 0, 0.5) 0.5318 
        x(0.6, 0, 1.0, 0, 0.6) 0.5262 
        x(0.7, 0, 1.0, 0, 0.7) 0.5195 

    4. filter place_ids by opening time? (weekday, hour)
        => place_th = 0 :   0.6461
        => place_th = 1 :   0.6470
        => place_th = 2 :   0.6471  
        => place_th = 3 :   0.6456
        => place_th = 4 :   0.6441
        => place_th = 5 :   0.6413
        => place_th = 6 :   0.6363
        [change to ratio, time_th]
        => time_th = 0.001, MAP=0.6453
        => time_th = 0.002, MAP=0.6454
        => time_th = 0.003, MAP=0.6455    <--- BEST!
        => time_th = 0.004, MAP=0.6453
        ------[after fix train/test split]-----
        time_th = 0.0   , 0.5328
        time_th = 0.001 , 0.5329
        time_th = 0.002 , 0.5335
        time_th = 0.003 , 0.5342
        time_th = 0.004 , 0.5344   <--- BEST!
        time_th = 0.005 , 0.5339
        time_th = 0.006 , 0.5330
        time_th = 0.008 , 0.5324
    5. smaller grid? x_step/y_step different step? (grid_step running!)
        => (0.04, 0.04) 0.6430 
        => (0.04, 0.05) 0.6436 
        => (0.04, 0.10) 0.6453    <--- BEST!
        => (0.04, 0.20) 0.6419 
        => (0.05, 0.04) 0.6415 
        => (0.05, 0.05) 0.6441 
        => (0.05, 0.10) 0.6440 
        => (0.05, 0.20) 0.6411 
        => (0.10, 0.04) 0.6397 
        => (0.10, 0.05) 0.6417 
        => (0.10, 0.10) 0.6426 
        => (0.10, 0.20) 0.6385 
        => (0.20, 0.04) 0.6345 
        => (0.20, 0.05) 0.6365 
        ------[after fix train/test split]-----
        (0.04, 0.08) 0.5371 
        (0.04, 0.10) 0.5359 
        (0.04, 0.16) 0.5262 
        (0.04, 0.20) 0.5338 
        (0.05, 0.08) 0.5383
        (0.05, 0.10) 0.5378 
        (0.05, 0.16) 0.5304 

        (0.04, 0.08) 0.5396
        (0.05, 0.08) 0.5399
        (0.08, 0.08) 0.5428   <--- BEST!
        (0.10, 0.08) 0.5420
        (0.20, 0.08) 0.5393

    6. place_min_last_checkin => good with proper threshold! could improve ~0.005
        => 500000 : 0.02236
        => 550000 : 0.02235
        => 600000 : 0.02237   <--- BEST!
        => 650000 : 0.02235
        => 700000 : 0.02225
    7. take place_id "popularity" into training feature
        popu_th = 0.0,   MAP=0.7109
        popu_th = 0.001, MAP=0.7115
        popu_th = 0.002, MAP=0.7117
        popu_th = 0.003, MAP=0.7122   
        popu_th = 0.004, MAP=0.7118
        popu_th = 0.005, MAP=0.7113
        popu_th = 0.01,  MAP=0.7086
        ------[after fix train/test split]-----
        popu_th = 0.0   , 0.5321
        popu_th = 0.001 , 0.5328
        popu_th = 0.002 , 0.5338
        popu_th = 0.003 , 0.5342
        popu_th = 0.004 , 0.5343
        popu_th = 0.005 , 0.5349  <--- BEST!
        popu_th = 0.01  , 0.5313
        popu_th = 0.015 , 0.5229
        popu_th = 0.02  , 0.5041
        popu_th = 0.025 , 0.4915
    8. train_min_time => submit only place_ids from 1.0x1.0 region, find the best threshold!
        train_min_time = 0      , 0.6636  <--- BEST!
        train_min_time = 150000 , 0.6457
        train_min_time = 200000 , 0.6306
        train_min_time = 250000 , 0.6237
        train_min_time = 300000 , 0.6198
        #---LB test--------------------
        train_min_time = 0      , 0.02238  <--- BEST!
        train_min_time = 150000 , 0.02236
        train_min_time = 250000 , 0.02233
    9. loc_th_x/loc_th_y
        th_x0.5, th_y1   MAP0.4926
        th_x1.0, th_y1   MAP0.5044
        th_x1.5, th_y1   MAP0.5082
        th_x2.0, th_y1   MAP0.5092
        th_x2.5, th_y1   MAP0.5093
        th_x0.5, th_y1.5 MAP0.5287
        th_x1.0, th_y1.5 MAP0.5397
        th_x1.5, th_y1.5 MAP0.5430
        th_x2.0, th_y1.5 MAP0.5443
        th_x2.5, th_y1.5 MAP0.5444
        th_x0.5, th_y2.5 MAP0.5349
        #-----------------------
        th_x2.0, th_y2 MAP0.5494
        th_x2.5, th_y2 MAP0.5496
        th_x3.0, th_y2 MAP0.5497  <--- BEST!
        th_x3.5, th_y2 MAP0.5497
        th_x4.0, th_y2 MAP0.5497
        th_x4.5, th_y2 MAP0.5496
        th_x2.0, th_y2.5 MAP0.5471
        th_x2.5, th_y2.5 MAP0.5473
        th_x3.0, th_y2.5 MAP0.5474
        th_x3.5, th_y2.5 MAP0.5474
        th_x4.0, th_y2.5 MAP0.5474
        th_x4.5, th_y2.5 MAP0.5473
        th_x2.0, th_y3 MAP0.5454
        th_x2.5, th_y3 MAP0.5456
        th_x3.0, th_y3 MAP0.5456
        th_x3.5, th_y3 MAP0.5456
        th_x4.0, th_y3 MAP0.5456
        th_x4.5, th_y3 MAP0.5456
        #------------------------
        th_x2.3, th_y1.7 MAP0.5480
        th_x2.5, th_y1.7 MAP0.5481
        th_x2.7, th_y1.7 MAP0.5482
        th_x2.9, th_y1.7 MAP0.5482
        th_x3.1, th_y1.7 MAP0.5482
        th_x3.3, th_y1.7 MAP0.5482
        th_x3.5, th_y1.7 MAP0.5482
        th_x2.3, th_y1.9 MAP0.5493
        th_x2.5, th_y1.9 MAP0.5493
        th_x2.7, th_y1.9 MAP0.5494
        th_x2.9, th_y1.9 MAP0.5494
        th_x3.1, th_y1.9 MAP0.5494
        th_x3.3, th_y1.9 MAP0.5494
        th_x3.5, th_y1.9 MAP0.5494
        th_x2.3, th_y2.1 MAP0.5490
        th_x2.5, th_y2.1 MAP0.5491
    9. split time_th into wdays/hours.
        time_th_wd = 0.0  , 0.5411
        time_th_wd = 0.001, 0.5411
        time_th_wd = 0.002, 0.5413
        time_th_wd = 0.003, 0.5413  <--- BEST!
        time_th_wd = 0.004, 0.5413  
        #------------------------
        time_th_hr = 0.0  , 0.5396
        time_th_hr = 0.001, 0.5397
        time_th_hr = 0.002, 0.5400
        time_th_hr = 0.003, 0.5411
        time_th_hr = 0.004, 0.5418  <--- BEST!
        time_th_hr = 0.005, 0.5402
        time_th_hr = 0.007, 0.5387
    10. drop place fewer than min_checkins
        place_min_checkin = 0, 0.5411
        place_min_checkin = 3, 0.5404
        place_min_checkin = 6, 0.5404
        place_min_checkin = 9, 0.5404
        place_min_checkin = 12, 0.5403
        place_min_checkin = 15, 0.5402
    

[TODO]
    1. time/loc by rate, not hard limits. and maybe add into training-features.
        => try df_preprocess:
          None        : 0.5898
          LOCATION    : 0.5546
          AVAIL_WDAYS : 
          AVAIL_HOURS : 0.7x (!!??)
          POPULAR     : 0.5357
        => by introduce too many features => need more samples, so try reduce candidate place_ids counts

    2. xgboost early stop with validation samples
    3. grid-wise parameter search
    

[COMBINE]
  {'mdl_weights': (0, 1, 0),       'time_th' : -1,    'loc_th_x' : 1000,  'loc_th_y' : 1000 } => tr=0.9925 / te=0.6325
  {'mdl_weights': (0.4, 1.0, 0.4), 'time_th' : -1,    'loc_th_x' : 1000,  'loc_th_y' : 1000 } => tr=       / te=0.6386
  {'mdl_weights': (0.4, 1.0, 0.4), 'time_th' : 0.003, 'loc_th_x' : 1000,  'loc_th_y' : 1000 } => tr=       / te=0.6414
  {'mdl_weights': (0.4, 1.0, 0.4), 'time_th' : 0.003, 'loc_th_x' : 1.1,   'loc_th_y' : 1000 } => tr=       / te=0.6434
  {'mdl_weights': (0.4, 1.0, 0.4), 'time_th' : 0.003, 'loc_th_x' : 1.1,   'loc_th_y' : 2.2  } => tr=0.9026 / te=0.6453

[feature selection]

---[months]---
x_cols = ['month2', 'month3', 'month6', 'accuracy', 'qday', 'month', 'year', 'x', 'y', 'hour', 'weekday']
=====[Eva.Test score] MAP=0.6515 =====
x_cols = ['month3', 'month6', 'accuracy', 'qday', 'month', 'year', 'x', 'y', 'hour', 'weekday']
=====[Eva.Test score] MAP=0.6561 =====
x_cols = ['month2', 'month6', 'accuracy', 'qday', 'month', 'year', 'x', 'y', 'hour', 'weekday']
=====[Eva.Test score] MAP=0.6534 =====
x_cols = ['month2', 'month3', 'accuracy', 'qday', 'month', 'year', 'x', 'y', 'hour', 'weekday']
=====[Eva.Test score] MAP=0.6523 =====
x_cols = ['month2', 'month3', 'month6', 'qday', 'month', 'year', 'x', 'y', 'hour', 'weekday']
=====[Eva.Test score] MAP=0.6356 =====
x_cols = ['month2', 'month3', 'month6', 'accuracy', 'month', 'year', 'x', 'y', 'hour', 'weekday']
=====[Eva.Test score] MAP=0.6528 =====
x_cols = ['month2', 'month3', 'month6', 'accuracy', 'qday', 'year', 'x', 'y', 'hour', 'weekday']
=====[Eva.Test score] MAP=0.6552 =====
x_cols = ['month2', 'month3', 'month6', 'accuracy', 'qday', 'month', 'x', 'y', 'hour', 'weekday']
=====[Eva.Test score] MAP=0.6282 =====


x_cols = ['accuracy', 'qday', 'month', 'year', 'x', 'y', 'hour', 'weekday']
=====[Eva.Test score] MAP=0.6374 =====
x_cols = ['accuracy', 'qday', 'year', 'x', 'y', 'hour', 'weekday']
=====[Eva.Test score] MAP=0.6292 =====


---[general]---
x_cols = ['accuracy', 'qday', 'month', 'year', 'x', 'y', 'hour', 'weekday']
=====[Eva.Test score] MAP=0.6570 =====
x_cols = ['qday', 'month', 'year', 'x', 'y', 'hour', 'weekday']
=====[Eva.Test score] MAP=0.6436 =====
x_cols = ['accuracy', 'month', 'year', 'x', 'y', 'hour', 'weekday']
=====[Eva.Test score] MAP=0.6591 =====
x_cols = ['accuracy', 'qday', 'year', 'x', 'y', 'hour', 'weekday']
=====[Eva.Test score] MAP=0.6442 =====
x_cols = ['accuracy', 'qday', 'month', 'x', 'y', 'hour', 'weekday']
=====[Eva.Test score] MAP=0.6386 =====


---[hours]---
x_cols = ['hour2', 'hour3', 'hour4', 'x', 'y', 'accuracy', 'hour', 'qday', 'weekday', 'month', 'year']
=====[Eva.Test score] MAP=0.6486 =====
x_cols = ['hour3', 'hour4', 'x', 'y', 'accuracy', 'hour', 'qday', 'weekday', 'month', 'year']
=====[Eva.Test score] MAP=0.6512 =====
x_cols = ['hour2', 'hour4', 'x', 'y', 'accuracy', 'hour', 'qday', 'weekday', 'month', 'year']
=====[Eva.Test score] MAP=0.6511 =====
x_cols = ['hour2', 'hour3', 'x', 'y', 'accuracy', 'hour', 'qday', 'weekday', 'month', 'year']
=====[Eva.Test score] MAP=0.6506 =====
x_cols = ['hour2', 'hour3', 'hour4', 'y', 'accuracy', 'hour', 'qday', 'weekday', 'month', 'year']


#------------------------------
#   Ideas
#------------------------------

[Pre Processing]
  #-----[TODO]-------------------
  1. highter weights for samples closer to test samples.
  1. fix location X-Y of training data, by the estimated central of location (normal distribution)
  #-----[DONE]-------------------
  0. remove closed places (last checkin < time=600000)
  0. Lookup table for b) place accurate position
  0. Lookup table for a) place opening time
  2. use model from neighbor blocks to decide together.
  


[Features] to train the model
  User vector: [location, time] 
  Item vector: [distance, onsale, score]

  1. place distance to the mean X,Y as features
  2. place checkins at this time (wday/hour)


  0. [Time (cat)] timestamp(min) => hour(important, by community), weekday, month
  1. [Distance (num)] build item locations first, add distance as feature.
  2. [OnSale (cat)] build item onsale time period first, this could be flag open/closed.
  3. [Score (num)] like yelp score, how hot this item in this region.
  4. add negative samples (at same region but not choosed) to make binary classification problem.
  5. divide into spatial/time grid models, blending models of nearby grids with weights


#------------------------------
#   Links
#------------------------------
[Explore]
  1. Visualise check-ins over time (gif)
    https://www.kaggle.com/peatle/facebook-v-predicting-check-ins/visualise-check-ins-over-time-gif/output
  2. Animated check-ins
    https://www.kaggle.com/sakvaua/facebook-v-predicting-check-ins/animated-check-ins/code
  3. Data exploration and visualisations
    https://www.kaggle.com/beyondbeneath/facebook-v-predicting-check-ins/data-exploration-and-visualisations/discussion
  4. Exploratory Data Analysis (on X)
    https://www.kaggle.com/msjgriffiths/facebook-v-predicting-check-ins/exploratory-data-analysis/discussion
  4. Timestamp structure
    https://www.kaggle.com/andersonk/facebook-v-predicting-check-ins/timestamps-structure/discussion
  5. 1 timestamp = 1 minute
    https://www.kaggle.com/radustoicescu/facebook-v-predicting-check-ins/10080-timestamps-in-a-day-78-days/discussion
  6. Accuracy (higher the better)
    https://www.kaggle.com/kennmyers/facebook-v-predicting-check-ins/exploring-accuracy/discussion
  7. Hypothesis on Timing of Drop
    https://www.kaggle.com/zeraes/facebook-v-predicting-check-ins/hypothesis-on-timing-of-drops/discussion
  8. Time
    https://www.kaggle.com/senorcampos/facebook-v-predicting-check-ins/on-weeks/discussion
  #-----[seen, just backup]-----


[Train]
  0. Mad scripts battle / with validation (LB: 0.45146) 
    https://www.kaggle.com/zfturbo/facebook-v-predicting-check-ins/mad-scripts-battle/code
    https://www.kaggle.com/breakfastpirate/facebook-v-predicting-check-ins/msb-with-validation-v2-0/code
    https://www.kaggle.com/breakfastpirate/facebook-v-predicting-check-ins/msb-with-validation-v3-0/output
    https://www.kaggle.com/justdoit/facebook-v-predicting-check-ins/msb-with-validation-v2-0/comments
    https://www.kaggle.com/breakfastpirate/facebook-v-predicting-check-ins/mad-scripts-battle-v2/comments
    https://www.kaggle.com/tunguz/facebook-v-predicting-check-ins/msb-with-validation-v3-0/code
  #-----[seen, just backup]-----
  3. Random Forest on a Few Blocks
    https://www.kaggle.com/apapiu/facebook-v-predicting-check-ins/random-forest-on-a-few-blocks/discussion
  4. 1-NN Benchmark
    https://www.kaggle.com/sbykau/facebook-v-predicting-check-ins/1-nn-benchmark/code
  6. naive kNN
    https://www.kaggle.com/ntuloser/facebook-v-predicting-check-ins/naive-knn/code

