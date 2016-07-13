## Kaggle challenge: predicting checkins

The goal of this competition is to predict which place a person would like to check into. For the purposes of this competition, Facebook released 40M check-in data of a small city, and our task is to predict the most likely check-in places of 8M user samples.

Since the dataset of this competition is in a huge scale, it's far more than just pushing predicting the ability of our machine learning algorithm. It also challenging us about how to deal with such a huge scale data. I've developed a lot of new tricks on dividing the question size without losing too much predicting accuracy, and speed-up, parallelize every code detail.

## modules

### The main modules
1. **parser.py**  
  parse in the raw data, split into training/validation/testing, also doing most of data pre-processing.
2. **trainer.py**  
  training models according to selected algorithm and parameters.
3. **evaluator.py**
  do data post-processing if enabled, evaluate trained models, and generate submittion file.
4. **submiter.py**
  programmatically submit to kaggle website.


### wrappers
1. **main.py**
  wrapper for above modules, all hyper-parameters and experiments are handled here.
2. **blending.py**
  do the blending among best models, generate blending model results.
3. **grouper.py**
   use tsne and knn results as extra inputs of training models.
4. **conventions.py**
   some convention functions handling time format and dataframe

### Tracing the code

1. top script entrance: go_train, everything start here!  

2. main.py being the wrapper, it host all kinds of experiment configuration and kick-off.  

3. all the modules are in folder: ./lib  



