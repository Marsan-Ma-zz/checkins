# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#COMPETITION NAME (the 'Competition'): Facebook V: Predicting Check Ins 
#COMPETITION SPONSOR: Facebook
#COMPETITION WEBSITE: https://www.kaggle.com/c/facebook-v-predicting-check-ins

'''Inspired by several scripts at:
https://www.kaggle.com/c/facebook-v-predicting-check-ins/scripts
Special thanks to
Michael Hartman
https://www.kaggle.com/zeroblue/facebook-v-predicting-check-ins/mad-scripts-battle-z/
Sandro for starting this KNN madness. :-)
https://www.kaggle.com/svpons/facebook-v-predicting-check-ins/grid-plus-classifier
ZFTurbo for the grid concept
https://www.kaggle.com/zfturbo/facebook-v-predicting-check-ins/msb-with-validation
And Kaggle Overfitting Community :)
'''

import numpy as np
import pandas as pd

#print('	Reading train.csv')
df_train = pd.read_csv('../input/train.csv',usecols=['x','y','time','place_id','accuracy'])
#print('	Reading test.csv')
df_test = pd.read_csv('../input/test.csv',usecols=['x','y','time','accuracy'])

#periodic features from 1
minute = 2*np.pi*(((df_train["time"]//5)+1)%288)/288
df_train['minute_sin'] = ((np.sin(minute)+1)* 0.56515).round(4)
df_train['minute_cos'] = ((np.cos(minute)+1)* 0.56515).round(4)
del minute
day = 2*np.pi*(((df_train['time']//1440)+1)%365)/365
df_train['day_of_year_sin'] = ((np.sin(day)+1)*0.32935).round(4)
df_train['day_of_year_cos'] = ((np.cos(day)+1)*0.32935).round(4)
del day
weekday = 2*np.pi*(((df_train['time']//1440)+1)%7)/7
df_train['weekday_sin'] = ((np.sin(weekday)+1)*0.2670).round(4)
df_train['weekday_cos'] = ((np.cos(weekday)+1)*0.2670).round(4)
del weekday
df_train['year'] = (df_train['time']//525600) * 0.51785
df_train.drop(['time'], axis=1, inplace=True)
df_train['accuracy'] = np.log10(df_train['accuracy'])*0.6

minute = 2*np.pi*(((df_test["time"]//5)+1)%288)/288
df_test['minute_sin'] = ((np.sin(minute)+1)* 0.56515).round(4)
df_test['minute_cos'] = ((np.cos(minute)+1)* 0.56515).round(4)
del minute
day = 2*np.pi*(((df_test['time']//1440)+1)%365)/365
df_test['day_of_year_sin'] = ((np.sin(day)+1)*0.32935).round(4)
df_test['day_of_year_cos'] = ((np.cos(day)+1)*0.32935).round(4)
del day
weekday = 2*np.pi*(((df_test['time']//1440)+1)%7)/7
df_test['weekday_sin'] = ((np.sin(weekday)+1)*0.2670).round(4)
df_test['weekday_cos'] = ((np.cos(weekday)+1)*0.2670).round(4)
del weekday
df_test['year'] = (df_test['time']//525600) * 0.51785
df_test.drop(['time'], axis=1, inplace=True)
df_test['accuracy'] = np.log10(df_test['accuracy'])*0.6

print('Generating wrong models. They are just useful to get this job :) ... done')
from sklearn.neighbors import KNeighborsClassifier

def calculate_distance(distances):
    return distances ** -2.2

def process_one_cell(df_cell_train, df_cell_test):
    # Remove infrequent places
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= 5).values
    df_cell_train = df_cell_train.loc[mask].copy()

    df_cell_train['x']=df_cell_train['x']*22
    df_cell_train['y']=df_cell_train['y']*52
    df_cell_test['x']=df_cell_test['x']*22
    df_cell_test['y']=df_cell_test['y']*52
      
    # Store row_ids for test
    row_ids = df_cell_test.index
    
    # Preparing data
    y = df_cell_train.place_id.values
    X = df_cell_train.drop(['place_id'], axis=1).values
    
    #Applying the classifier
    clf = KNeighborsClassifier(n_neighbors=np.floor(np.sqrt(y.size)/5.83).astype(int),
                            weights=calculate_distance, p=1, 
                            n_jobs=2, leaf_size=20)
    clf.fit(X, y)
    y_pred = clf.predict_proba(df_cell_test.values)
    y_pred_labels = np.argsort(y_pred, axis=1)[:,:-4:-1]
    pred_labels = clf.classes_[y_pred_labels]
    cell_pred = np.column_stack((row_ids, pred_labels)).astype(np.int64) 
    
    return cell_pred
    
# From https://www.kaggle.com/zeroblue/facebook-v-predicting-check-ins/mad-scripts-battle-z/
def create_time_dict(t_cuts, time_factor, time_aug):
    
    t_slice = 24 / t_cuts
    time_dict = dict()
    for t in range(t_cuts):
        
        t_min = 2 * np.pi * (t * t_slice * 12 / 288)
        t_max = 2 * np.pi * (((t + 1) * t_slice * 12 - 1) / 288)
        sin_t_start = np.round(np.sin(t_min)+1, 4) * time_factor
        sin_t_stop = np.round(np.sin(t_max)+1, 4) * time_factor
        cos_t_start = np.round(np.cos(t_min)+1, 4) * time_factor
        cos_t_stop = np.round(np.cos(t_max)+1, 4) * time_factor
        sin_t_min = min((sin_t_start, sin_t_stop))
        sin_t_max = max((sin_t_start, sin_t_stop))
        cos_t_min = min((cos_t_start, cos_t_stop))
        cos_t_max = max((cos_t_start, cos_t_stop))

        time_dict[t] = [sin_t_min, sin_t_max, cos_t_min, cos_t_max]
        t_min = 2 * np.pi * ((t * t_slice - time_aug) * 12 / 288)
        t_max = 2 * np.pi * ((((t + 1) * t_slice + time_aug)* 12 - 1) / 288)
        sin_t_start = np.round(np.sin(t_min)+1, 4) * time_factor
        sin_t_stop = np.round(np.sin(t_max)+1, 4) * time_factor
        cos_t_start = np.round(np.cos(t_min)+1, 4) * time_factor
        cos_t_stop = np.round(np.cos(t_max)+1, 4) * time_factor
        sin_t_min = min((sin_t_start, sin_t_stop, sin_t_min))
        sin_t_max = max((sin_t_start, sin_t_stop, sin_t_max))
        cos_t_min = min((cos_t_start, cos_t_stop, cos_t_min))
        cos_t_max = max((cos_t_start, cos_t_stop, cos_t_max))
        time_dict[t] += [sin_t_min, sin_t_max, cos_t_min, cos_t_max]
        
    return time_dict

def process_grid(df_train, df_test):
                         
    # Defining the size of the grid
    x_cuts = 10 # number of cuts along x 
    y_cuts = 25 # number of cuts along y
    t_cuts = 4
    x_border_aug = 0.04 # expansion of x border on train 
    y_border_aug = 0.016 # expansion of y border on train
    time_aug = 2

    preds_list = []
    x_slice = df_train['x'].max() / x_cuts
    y_slice = df_train['y'].max() / y_cuts
    time_max = df_train['minute_sin'].max()
    time_factor = time_max / 2
    time_dict = create_time_dict(t_cuts, time_factor, time_aug)

    for i in range(x_cuts):
        x_min = x_slice * i
        x_max = x_slice * (i+1)
        x_max += int((i+1) == x_cuts) # expand edge at end

        mask = (df_test['x'] >= x_min)
        mask = mask & (df_test['x'] < x_max)      
        df_col_test = df_test[mask]
        x_min -= x_border_aug
        x_max += x_border_aug
        mask = (df_train['x'] >= x_min)
        mask = mask & (df_train['x'] < x_max)
        df_col_train = df_train[mask]

        for j in range(y_cuts):
            y_min = y_slice * j
            y_max = y_slice * (j+1)
            y_max += int((j+1) == y_cuts) # expand edge at end

            mask = (df_col_test['y'] >= y_min)
            mask= mask & (df_col_test['y'] < y_max)
            df_row_test = df_col_test[mask]
            y_min -= y_border_aug
            y_max += y_border_aug
            mask = (df_col_train['y'] >= y_min)
            mask = mask & (df_col_train['y'] < y_max)
            df_row_train = df_col_train[mask]

            for t in range(4):
                t_lim = time_dict[t]
                mask = df_row_test['minute_sin'].between(t_lim[0], t_lim[1])
                mask = mask & df_row_test['minute_cos'].between(t_lim[2], t_lim[3])
                df_cell_test = df_row_test[mask].copy()
                mask = df_row_train['minute_sin'].between(t_lim[4], t_lim[5])
                mask = mask & df_row_train['minute_cos'].between(t_lim[6], t_lim[7])
                df_cell_train = df_row_train[mask].copy()
                cell_pred = process_one_cell(df_cell_train.copy(), 
                                             df_cell_test.copy())
                preds_list.append(cell_pred)
                
    preds = np.vstack(preds_list)
    return preds

# From: https://www.kaggle.com/drarfc/facebook-v-predicting-check-ins/fastest-way-to-write-the-csv
def generate_submission(preds):    
    print('Writing submission file')
    with open('sample_submission.csv', "w") as out:
        out.write("row_id,place_id\n")
        rows = ['']*preds.shape[0]
        for num in range(preds.shape[0]):
            rows[num]='%d,%d %d %d\n' % (preds[num,0],preds[num,1],preds[num,2],preds[num,3])
        out.writelines(rows)

preds = process_grid(df_train, df_test)

generate_submission(preds)