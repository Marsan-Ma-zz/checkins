#------------------------------
#   Performance memo
#------------------------------
  [20160601]
    0. skrf tree max_depth: (n_estimators: 200)
        7: 0.6447
        8: 0.6462
        9: 0.6462
        10: 0.6465
        11: 0.6466
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
    4. filter place_ids by opening time? (weekday, hour)
        => place_th = 0 :   0.6461
        => place_th = 1 :   0.6470
        => place_th = 2 :   0.6471  <---BEST!
        => place_th = 3 :   0.6456
        => place_th = 4 :   0.6441
        => place_th = 5 :   0.6413
        => place_th = 6 :   0.6363
        [change to ratio, time_th]
        => time_th = 0.001, MAP=0.6453
        => time_th = 0.002, MAP=0.6454
        => time_th = 0.003, MAP=0.6455
        => time_th = 0.004, MAP=0.6453
    5. smaller grid? x_step/y_step different step? (grid_step running!)
        => (0.04, 0.04) 0.6430 
        => (0.04, 0.05) 0.6436 
        => (0.04, 0.10) 0.6453    <---BEST!
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
    6. [TRY] train_min_time => submit only place_ids from 1.0x1.0 region, find the best threshold!
    7. [TRY] place_max_first_checkin => good with proper threshold! could improve ~0.005
    8. hour2, hour3, hour4
    9. time/loc by rate, not hard limits. and maybe add into training-features.


[COMBINE]
  {'mdl_weights': (0, 1, 0),       'time_th' : -1,    'loc_th_x' : 1000,  'loc_th_y' : 1000 } => tr=0.9925 / te=0.6325
  {'mdl_weights': (0.4, 1.0, 0.4), 'time_th' : -1,    'loc_th_x' : 1000,  'loc_th_y' : 1000 } => tr=       / te=0.6386
  {'mdl_weights': (0.4, 1.0, 0.4), 'time_th' : 0.003, 'loc_th_x' : 1000,  'loc_th_y' : 1000 } => tr=       / te=0.6414
  {'mdl_weights': (0.4, 1.0, 0.4), 'time_th' : 0.003, 'loc_th_x' : 1.1,   'loc_th_y' : 1000 } => tr=       / te=0.6434
  {'mdl_weights': (0.4, 1.0, 0.4), 'time_th' : 0.003, 'loc_th_x' : 1.1,   'loc_th_y' : 2.2  } => tr=0.9026 / te=0.6453


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

  0. [Time (cat)] timestamp(min) => hour(important, by community), weekday, month
  1. [Distance (num)] build item locations first, add distance as feature.
  2. [OnSale (cat)] build item onsale time period first, this could be flag open/closed.
  3. [Score (num)] like yelp score, how hot this item in this region.
  4. add negative samples (at same region but not choosed) to make binary classification problem.
  5. divide into spatial/time grid models, blending models of nearby grids with weights


[Post Processing] after model prediction
  0. filter out items too far from current location
  1. filter out non-popular items
  2. filter out closed items


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

[Tool Usage]
  1. XGBoost Ranking Question
    https://www.kaggle.com/c/facebook-v-predicting-check-ins/forums/t/21142/xgboost-ranking-question
