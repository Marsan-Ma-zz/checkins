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
        ----------------------
        all x_cols     => 0.9884 ,0.5300 
        no yearday     => 0.9885 ,0.5343  (-)
        no weekofyear  => 0.9887 ,0.5333  (-)
        no month       => 0.9885 ,0.5342  (?)  > add back: 0.5357
        no dayofmonth  => 0.9880 ,0.5331  (-)
        no season      => 0.9883 ,0.5325  (-)
        no accuracy    => 0.9884 ,0.5306 
        no logacc      => 0.9884 ,0.5303 
        no qday        => 0.9887 ,0.5278 
        no weekday     => 0.9857 ,0.5244 
        no hour        => 0.9858 ,0.5156 
        no year        => 0.9871 ,0.5082 
        no x           => 0.9824 ,0.5012 
        no y           => 0.9775 ,0.4272
        ----------------------
        all x_cols    => 0.5371
        no dayofmonth => 0.5396 (-)
        no season     => 0.5373
        no logacc     => 0.5370
        no qday       => 0.5360
        no month      => 0.5354
        no accuracy   => 0.5378
        no weekday    => 0.5310

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
        place_min_checkin = 0, 0.9400/0.5362
        place_min_checkin = 1, 0.9400/0.5363
        place_min_checkin = 2, 0.9400/0.5363
        place_min_checkin = 3, 0.9400/0.5363
        place_min_checkin = 4, 0.9400/0.5363
        -------------------------
        place_min_checkin = 6, 0.5404
        place_min_checkin = 9, 0.5404
        place_min_checkin = 12, 0.5403
        place_min_checkin = 15, 0.5402
    11. time/loc by rate, not hard limits. and maybe add into training-features.
        => try df_preprocess:
          None        : 0.5898
          10L 0.9006 ,0.4633 
          20L 0.8958 ,0.4524 
          30L 0.8941 ,0.4519 
          40L 0.8933 ,0.4503 
          50L 0.8908 ,0.4471 
          60L 0.8904 ,0.4481 
          70L 0.8892 ,0.4438 
          80L 0.8886 ,0.4454 
          90L 0.8880 ,0.4479 

          10P 0.9093 ,0.4578 
          20P 0.9074 ,0.4624 
          30P 0.9066 ,0.4640 
          40P 0.9075 ,0.4464 
          50P 0.9065 ,0.4528 
          60P 0.9068 ,0.4481
          70P 0.9061 ,0.4503
          
          10P 0.9415  0.5343
          20P 0.9420  0.5308
          30P 0.9416  0.5307
          40P 0.9425  0.5273
          50P 0.9425  0.5273
          60P 0.9422  0.5262
          70P 0.9421  0.5285
          80P 0.9423  0.5264
          90P 0.9424  0.5253

          10X 0.9395  0.5116
          20X 0.9383  0.5032
          30X 0.9367  0.4971
          40X 0.9349  0.4857
          50X 0.9343  0.4874
          60X 0.9344  0.4845
          70X 0.9326  0.4788
          80X 0.9325  0.4780
          90X 0.9315  0.4693

          10Y 0.9379  0.5137
          20Y 0.9362  0.5073
          30Y 0.9349  0.4997
          40Y 0.9324  0.4926
          50Y 0.9317  0.4904
        => by introduce too many features => need more samples, so try reduce candidate place_ids counts
    12. xgboost
          'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 5, 0.7432, 0.5223
          'max_depth': 7, 'learning_rate': 0.1, 'n_estimators': 5, 0.7582, 0.5200
          'max_depth': 7, 'learning_rate': 0.15, 'n_estimators': 5, 0.7680, 0.5232
          'max_depth': 9, 'learning_rate': 0.05, 'n_estimators': 5, 0.7446, 0.5215
          'max_depth': 9, 'learning_rate': 0.1, 'n_estimators': 5, 0.7621, 0.5197
          'max_depth': 9, 'learning_rate': 0.15, 'n_estimators': 5, 0.7720, 0.5233
          'max_depth': 11, 'learning_rate': 0.05, 'n_estimators': 5, 0.7452, 0.5220
          'max_depth': 11, 'learning_rate': 0.1, 'n_estimators': 5, 0.7635, 0.5201
          'max_depth': 11, 'learning_rate': 0.15, 'n_estimators': 5, 0.7743, 0.5240
          'max_depth': 13, 'learning_rate': 0.05, 'n_estimators': 5, 0.7454, 0.5219
          'max_depth': 13, 'learning_rate': 0.1, 'n_estimators': 5, 0.7640, 0.5200
          'max_depth': 13, 'learning_rate': 0.15, 'n_estimators': 5, 0.7749, 0.5233
          'max_depth': 15, 'learning_rate': 0.05, 'n_estimators': 5, 0.7454, 0.5219
          'max_depth': 15, 'learning_rate': 0.1, 'n_estimators': 5, 0.7640, 0.5201
          'max_depth': 15, 'learning_rate': 0.15, 'n_estimators': 5, 0.7750, 0.5236

          'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 10, 0.7699, 0.5300
          'max_depth': 7, 'learning_rate': 0.1, 'n_estimators': 10, 0.7961, 0.5309
          'max_depth': 7, 'learning_rate': 0.15, 'n_estimators': 10, 0.8152, 0.5294
          'max_depth': 9, 'learning_rate': 0.05, 'n_estimators': 10, 0.7738, 0.5302
          'max_depth': 9, 'learning_rate': 0.1, 'n_estimators': 10, 0.8060, 0.5316
          'max_depth': 9, 'learning_rate': 0.15, 'n_estimators': 10, 0.8259, 0.5283
          'max_depth': 11, 'learning_rate': 0.05, 'n_estimators': 10, 0.7754, 0.5303
          'max_depth': 11, 'learning_rate': 0.1, 'n_estimators': 10, 0.8097, 0.5312
          'max_depth': 11, 'learning_rate': 0.15, 'n_estimators': 10, 0.8321, 0.5300
          'max_depth': 13, 'learning_rate': 0.05, 'n_estimators': 10, 0.7758, 0.5306
          'max_depth': 13, 'learning_rate': 0.1, 'n_estimators': 10, 0.8117, 0.5306
          'max_depth': 13, 'learning_rate': 0.15, 'n_estimators': 10, 0.8342, 0.5293
          'max_depth': 15, 'learning_rate': 0.05, 'n_estimators': 10, 0.7760, 0.5306
          'max_depth': 15, 'learning_rate': 0.1, 'n_estimators': 10, 0.8125, 0.5310
          'max_depth': 15, 'learning_rate': 0.15, 'n_estimators': 10, 0.8353, 0.5300

          'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 15, 0.7893, 0.5356   <-- BEST!!!
          'max_depth': 7, 'learning_rate': 0.1, 'n_estimators': 15, 0.8240, 0.5334
          'max_depth': 7, 'learning_rate': 0.15, 'n_estimators': 15, 0.8499, 0.5327
          'max_depth': 9, 'learning_rate': 0.05, 'n_estimators': 15, 0.7955, 0.5348
          'max_depth': 9, 'learning_rate': 0.1, 'n_estimators': 15, 0.8361, 0.5336
          'max_depth': 9, 'learning_rate': 0.15, 'n_estimators': 15, 0.8635, 0.5316
          'max_depth': 11, 'learning_rate': 0.05, 'n_estimators': 15, 0.7984, 0.5351
          'max_depth': 11, 'learning_rate': 0.1, 'n_estimators': 15, 0.8425, 0.5338
    13. skrf
          'max_depth': 5, 'n_estimators': 200 ,0.7532 ,0.5039
          'max_depth': 7, 'n_estimators': 200 ,0.8640 ,0.5247
          'max_depth': 9, 'n_estimators': 200 ,0.9475 ,0.5325
          'max_depth': 11, 'n_estimators': 200 ,0.9807 ,0.5355
          'max_depth': 13, 'n_estimators': 200 ,0.9910 ,0.5354
          'max_depth': 15, 'n_estimators': 200 ,0.9941 ,0.5340

          'max_depth': 5, 'n_estimators': 300 ,0.7535 ,0.5047
          'max_depth': 7, 'n_estimators': 300 ,0.8638 ,0.5254
          'max_depth': 9, 'n_estimators': 300 ,0.9477 ,0.5328
          'max_depth': 11, 'n_estimators': 300 ,0.9808 ,0.5361
          'max_depth': 13, 'n_estimators': 300 ,0.9912 ,0.5360
          'max_depth': 15, 'n_estimators': 300 ,0.9941 ,0.5360

          'max_depth': 5, 'n_estimators': 500 ,0.7532 ,0.5039
          'max_depth': 7, 'n_estimators': 500 ,0.8644 ,0.5268
          'max_depth': 9, 'n_estimators': 500 ,0.9477 ,0.5346
          'max_depth': 11, 'n_estimators': 500 ,0.9808 ,0.5356
          'max_depth': 13, 'n_estimators': 500 ,0.9911 ,0.5363  <--- 2nd BEST!
          'max_depth': 15, 'n_estimators': 500 ,0.9942 ,0.5366

          'max_depth': 5, 'n_estimators': 800 ,0.7537 ,0.5043
          'max_depth': 7, 'n_estimators': 800 ,0.8644 ,0.5263
          'max_depth': 9, 'n_estimators': 800 ,0.9479 ,0.5355
          'max_depth': 11, 'n_estimators': 800 ,0.9811 ,0.5363
          'max_depth': 13, 'n_estimators': 800 ,0.9912 ,0.5380  <--- BEST!
          'max_depth': 15, 'n_estimators': 800 ,0.9941 ,0.5374

          'max_depth': 5, 'n_estimators': 1000 ,0.7536 ,0.5037
          'max_depth': 7, 'n_estimators': 1000 ,0.8642, 0.5261
          'max_depth': 9, 'n_estimators': 1000 ,0.9479, 0.5350
          'max_depth': 11, 'n_estimators': 1000 ,0.9812, 0.5369
          'max_depth': 13, 'n_estimators': 1000 ,0.9911, 0.5377
          'max_depth': 15, 'n_estimators': 1000 ,0.9941, 0.5369
    14. KNN
          -----[x, y weights]-----
          x=300, y=600 0.5634
          x=300, y=800 0.5622
          x=300, y=1000 0.5627
          x=300, y=1200 0.5594
          x=300, y=1400 0.5575
          x=500, y=600 0.5674
          x=500, y=800 0.5690
          x=500, y=1000 0.5686
          x=500, y=1200 0.5660
          x=500, y=1400 0.5660
          x=700, y=600 0.5633
          x=700, y=800 0.5677
          x=700, y=1000 0.5693
          x=700, y=1200 0.5693  <--- BEST!
          x=700, y=1400 0.5683
          x=900, y=600 0.5569
          x=900, y=800 0.5635
          x=900, y=1000 0.5640
          x=900, y=1200 0.5642
    15. remove_distance_outlier: remove outlier from training set which (x, y) too far from average of certain place
          None    => (0.9402/0.5360?)
          std1.0  => 0.9863/0.5176
          std1.5  => 0.9828/0.5320
          std2.0  => 0.9808/0.5372  <--- BEST!
          std2.5  => 0.9661/0.5353
    16. KNN 
          0.5218 for no x/y_inter, 
          tab1: 'x_inter'/'y_inter' = 2/2, mdl_weights = (0.4, 1, 0.4)   => 0.5153
          tab2: 'x_inter'/'y_inter' = 1/1, mdl_weights = (0.4, 1, 0.4)   => 0.5045
          tab3: blending for rank_w = [1, 0.4, 0.2]   => improve 0.00019 only.
    17. SVM
    18. Blending:
          (rank_ws_1_0.6_0.4) weight more on kings? seems should not to ...
            gs_top_w1.5 => 
            gs_top_w1.7 => 
            gs_top_w1.9 => 
            gs_top_w2 => 0.58529  <--- BEST?
            gs_top_w3 => 0.58513
            gs_top_w5 => 0.58489
          (top_w2)
            gs_rank_ws_1_0.8_0.6 => 0.58479
            gs_rank_ws_1_0.8_0.4 => 0.58492
            gs_rank_ws_1_0.6_0.4 => 0.58529
            gs_rank_ws_1_0.6_0.1 =>
            gs_rank_ws_1_0.4_0.2 =>
    19. [TODO]
          => Diego: remove ash, mod weight to [1, 0.6, 0.4]
          => blending: by same models but drop 1 feature 
          => blending meterials from different model setup (tree depth, knn distance type, ...)
          => check correlation with small grids first.
          => xgboost early stop with validation samples
          => use latter sample (near test samples) only
          => remove place_min_last_checkin
    

    Markus: "I think the basic ideas are out there already: learning various joint or individual distributions for the space and time variables for each place_id (and the overall relative popularities). The challenge is doing this in a computationally efficient manner without losing too much information."

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


0       0  0.7941  9.0809        54  470702  8523065625    22     4        5   
1       1  5.9567  4.7968        13  186555  1757726713    14     3        4   
2       2  8.3078  7.0407        74  322648  1137537235     2     1        1   
5       5  3.8099  1.9586        75  178065  6289802927    16     3        5   
6       6  6.3336  4.3720        13  666829  9931249544     2     1        2   

   month  year  
0     11     1  
1      5     1  
2      8     1  
5      5     1  
6      4     2  

data ex: df_train.head()    row_id       x       y  accuracy    time    place_id       hour  weekday  \
0       0  0.7941  9.0809        54  470702  8523065625  14.050000        6   
1       1  5.9567  4.7968        13  186555  1757726713   6.266667        5   
2       2  8.3078  7.0407        74  322648  1137537235  18.483333        1   
5       5  3.8099  1.9586        75  178065  6289802927   8.766667        6   
6       6  6.3336  4.3720        13  666829  9931249544  18.833333        2   

   yearday  month  year  qday  dayofmonth  weekofyear  season    logacc  
0      327     11     1     2          23          46       4  3.988984  
1      130      5     1     1          10          18       2  2.564949  
2      224      8     1     3          12          32       3  4.304065  
5      124      5     1     1           4          17       2  4.317488  
6       98      4     2     3           8          14       2  2.564949  

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

