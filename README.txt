#------------------------------
#   Performance memo
#------------------------------
  [20160601]
    1. all x_cols   => 0.6364
        no x        => 0.6178
        no y        => 0.5788   (!)
        no accuracy => 0.6237
        no hour     => 0.6210
        no weekday  => 0.6238
        no qday     => 0.6353
        no month    => 0.6266
        no year     => 0.6174   (!)
    2. add time     => 0.6330
    3. train_min_time => don't use, more samples always better
    4. place_max_first_checkin => good with proper threshold! could improve ~0.005


#------------------------------
#   Ideas
#------------------------------

[Pre Processing]
  #-----[DONE]-------------------
  0. remove closed places (last checkin < time=600000)
  #-----[TODO]-------------------
  1. highter weights for samples closer to test samples.
  2. use model from neighbor blocks to decide together.
  0. Lookup table for a) place opening time, b) place accurate position
  1. fix location X-Y of training data, by the estimated central of location
  2. fix X by fitting normal distribution?


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
