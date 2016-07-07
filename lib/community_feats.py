# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

__author__ = 'Abhi'

'''Contains additions from several excellent scripts and ideas posted on the forum

The script does the following:
- calculates time features for test and train
- calculates knn using a grid. Top 10 probabilities are calculated
- calculates probability lookup tables for main features like hour etc
- multiplies knn probabilities with probabilities from lookup tables to give total probability
- selects top 3 placeids based on total probability and generates submission file
- cross-validation is included (for all grid cells as well as one grid cell)
- takes 30 min  to run and produces a score of 0.5865 lb
'''

import os, sys, pickle, gzip
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import time
from datetime import datetime
import multiprocessing as mp
pool_size = 1 #mp.cpu_count()
from lib import submiter
from collections import OrderedDict, defaultdict
import xgboost as xgb
from sklearn import linear_model, ensemble

def calcgridwisemap3(group): 
    score = ([1/1.0, 1/2.0, 1/3.0]*(np.asarray(group[['ytest']]) == np.asarray(group[['id1','id2','id3']])) ).sum()/group.shape[0]
    return score
    
def makeprobtable(train, feature, threshold):
    table = train.groupby('place_id')[feature].value_counts()
    table = table/train.groupby('place_id')[feature].count()
    table = table.reset_index(level=0, drop=True)  #drop placeid index
    table[0]=0 # all missing indices have zero probability
    table[table < threshold]= threshold  #threshold small probabilities including zeros to threshold value
    return table

    
def getprob(ind, table, nn):
    split = len(ind)*nn//3 # split array operations for memory management
    temp = ind.reshape(-1)
    temp1=np.invert(np.in1d(temp[:split], table.index.values))
    temp2=np.invert(np.in1d(temp[split:split*2], table.index.values))
    temp3=np.invert(np.in1d(temp[split*2:], table.index.values))
    temp[np.concatenate((temp1,temp2,temp3))] = 0 # find indices that are not in lookup and set to zero
    temp1=table[temp[:split]]
    temp2=table[temp[split:split*2]]
    temp3=table[temp[split*2:]]
    prob = np.concatenate((temp1,temp2,temp3))
    prob = prob.reshape(-1,nn)
    return prob

def extendgrid(size_x, size_y, n_cell_x, n_cell_y, extension_x=0.03, extension_y=0.015):
    xmin =np.linspace(0,10-size_x,n_cell_x)    
    xmax =np.linspace(0+size_x,10,n_cell_x)    
    ymin =np.linspace(0,10-size_y,n_cell_y)    
    ymax =np.linspace(0+size_y,10,n_cell_y)    
    
    grid1 = np.tile(xmin,n_cell_y)
    grid2 = np.tile(xmax,n_cell_y)
    grid3 = np.repeat(ymin,n_cell_x)
    grid4 = np.repeat(ymax,n_cell_x)
    grid1 = grid1 - extension_x
    grid2 = grid2 + extension_x
    grid3 = grid3 - extension_y
    grid4 = grid4 + extension_y
    
    grid = np.vstack((grid1,grid2,grid3,grid4)).T
    return grid


def get_data(cross_validation, n_cell_x, n_cell_y):
    cache_name = "./data/cache/cache_knn2_dfs_cv%i_nx%i_ny%i.pkl" % (cross_validation, n_cell_x, n_cell_y)
    if os.path.exists(cache_name):
        print("read data from cache: %s @ %s" % (cache_name, datetime.now()))
        train, test, ytrain, ytest = pickle.load(open(cache_name, 'rb'))
    else:
        print("read data from csv @ %s" % (datetime.now()))
        train = pd.read_csv('./data/train.csv.zip',dtype={'place_id': np.int64}, index_col = 0) 

        train['hour'] = ( (train['time']+120)/60)%24+1 
        train['weekday'] = (train['time']/1440)%7+1 
        train['month'] = ( train['time'] /43800)%12+1 
        train['year'] = (train['time']/525600)+1 
        train['four_hour'] = (train['time']/240)%6+1
        train['acc'] = np.log10(train['accuracy'])

        pd.options.mode.chained_assignment = None
        add_data = train[train.hour<2.5]# add data for periodic time that hit the boundary
        add_data.hour = add_data.hour+24
        add_data2 = train[train.hour>22.5]
        add_data2.hour = add_data2.hour-24
        train = train.append(add_data)
        train = train.append(add_data2)
        del add_data,add_data2

        ytrain, ytest = None, None
        if cross_validation == 1:
            print('Loading cross validation data ...')
            test = train.query('month >=5.0 and year >=2.0')  
            train = train.query('~(month >=5.0 and year >=2.0)')
            ytrain = train['place_id']
            test = test.query('place_id in @ytrain')
            ytest = test['place_id']
            del test['place_id']
            test.reset_index(inplace=True) 
            test['row_id'] = test.index.values                     
            test.set_index('row_id',inplace=True)
        else:    
            print('Loading data ...')
            test = pd.read_csv('./data/test.csv.zip', index_col = 0)
          
            test['hour'] = ((test['time']+120)/60)%24+1 
            test['weekday'] = (test['time']/1440)%7+1 
            test['month'] = (test['time']/43800)%12+1 
            test['year'] = (test['time']/525600)+1 
            test['four_hour'] = (test['time']/240)%6+1
            test['acc'] = np.log10(test['accuracy']) 
        
        #Make grid
        size_x = 10. / n_cell_x
        size_y = 10. / n_cell_y
            
        eps = 0.00001  
        xs = np.where(train.x.values < eps, 0, train.x.values - eps)
        ys = np.where(train.y.values < eps, 0, train.y.values - eps)
        pos_x = (xs / size_x).astype(np.int)
        pos_y = (ys / size_y).astype(np.int)
        train['grid_cell'] = pos_y * n_cell_x + pos_x

        xs = np.where(test.x.values < eps, 0, test.x.values - eps)
        ys = np.where(test.y.values < eps, 0, test.y.values - eps)
        pos_x = (xs / size_x).astype(np.int)
        pos_y = (ys / size_y).astype(np.int)
        test['grid_cell'] = pos_y * n_cell_x + pos_x

        pickle.dump([train, test, ytrain, ytest], open(cache_name, 'wb'))
        print("done read data & pre-processing @ %s" % (datetime.now()))
    return train, test, ytrain, ytest


def calculate_distance(distances):
        return distances ** -2

def calculate_distance2(distances):
        return distances ** -2.2

def process_grid(g_id, grid_train, grid_test, debug=False):
    le = LabelEncoder()
    y = le.fit_transform(grid_train.place_id.values)
    X = grid_train[['x', 'y', 'hour', 'weekday', 'month', 'year', 'acc']].values * weights[g_id][:7]
    X_test = grid_test[['x', 'y', 'hour', 'weekday', 'month', 'year', 'acc']].values * weights[g_id][:7]
    
    ###Applying the knn classifier
    #nearest = (weights[g_id][7]).copy().astype(int)
    

    if alg == 'knn':
        nearest = np.floor(np.sqrt(y.size)/5.1282).astype(int)
        clf = KNeighborsClassifier(n_neighbors=nearest, weights=calculate_distance, metric='cityblock')
    elif alg == 'knnz':
        nearest = np.floor(np.sqrt(y.size)/5.83).astype(int)
        clf = KNeighborsClassifier(n_neighbors=nearest, weights=calculate_distance2, p=1, n_jobs=2, leaf_size=20)
    elif alg == 'skrf':
        clf = ensemble.RandomForestClassifier(
            n_estimators=mdl_config.get('n_estimators', 1500), 
            max_features=mdl_config.get('max_features', 0.5),  
            max_depth=mdl_config.get('max_depth', 15), 
            min_samples_split=mdl_config.get('min_samples_split', 7),
            bootstrap=mdl_config.get('bootstrap', True),
            n_jobs=-1,
        )
    elif alg == 'skrfp':
        clf = ensemble.RandomForestClassifier(
            n_estimators=mdl_config.get('n_estimators', 800), 
            max_features=mdl_config.get('max_features', 0.5),
            max_depth=mdl_config.get('max_depth', 15),
            criterion='entropy',
            n_jobs=-1,
        )
    elif alg =='sket':
        clf = ensemble.ExtraTreesClassifier(
            n_estimators=mdl_config.get('n_estimators', 1500), 
            max_features=mdl_config.get('max_features', 0.5),  
            max_depth=mdl_config.get('max_depth', 15), 
            n_jobs=-1,
        )
    elif alg =='sketp':
        clf = ensemble.ExtraTreesClassifier(
            n_estimators=mdl_config.get('n_estimators', 1500), 
            max_features=mdl_config.get('max_features', 0.5),  
            max_depth=mdl_config.get('max_depth', 11), 
            criterion='entropy', 
            n_jobs=-1,
        )
    elif alg == 'xgb':
        clf = xgb.XGBClassifier(
            n_estimators=mdl_config.get('n_estimators', 30), 
            max_depth=mdl_config.get('max_depth', 7), 
            learning_rate=mdl_config.get('learning_rate', 0.1), 
            objective="multi:softprob", 
            silent=True
        )
    

    if (g_id == grid_onecell): print("clf=%s" % clf)

    if debug:
        y_pred = np.array([[0]*10 for i in range(len(X_test))])
    else:
        clf.fit(X, y)
        y_pred = clf.predict_proba(X_test)
    indices = le.inverse_transform(  np.argsort(y_pred, axis=1)[:,::-1][:,:10]  )  
    knn_prob = np.sort(y_pred, axis=1)[:,::-1][:,:10]
    grid_train, grid_test, X, X_test, y_pred, clf = [None] * 6
    return indices, knn_prob


def process_all_grids(train, test, repeats, grid, th=8):
    processes = []
    mp_pool = mp.Pool(pool_size)
    tr = train[['x','y']]

    indices = np.zeros((test.shape[0], 10), dtype=np.int64)
    knn_prob = np.zeros((test.shape[0], 10), dtype=np.float64)
    grid_num = np.zeros((test.shape[0], 1), dtype=np.float64)
    
    mdl_path = "./submit/%s" % stamp
    if not os.path.exists(mdl_path): os.mkdir(mdl_path)

    no_cache = (stamp in ['knn_blend', 'knnz_blend'])

    for g_id in repeats:
        if g_id % 100 == 0:
            print("g_id=%i @ %s" % (g_id, datetime.now()))
        
        # if g_id < 600: continue

        cache_name = "%s/gid_%i.pkl" % (mdl_path, g_id)
        if (not grid_onecell) and (not no_cache) and os.path.exists(cache_name):
            print("skip g_id=%i since %s exists" % (g_id, cache_name))
            mp_row_ids, mp_ind, mp_kprob, mp_g_id = pickle.load(open(cache_name, 'rb'))
            indices[mp_row_ids] = mp_ind
            knn_prob[mp_row_ids] = mp_kprob
            grid_num[mp_row_ids] = mp_g_id
        else:
            print("start processing g_id=%i @ %s" % (g_id, datetime.now()))
            #Applying classifier to one grid cell
            xmin, xmax, ymin, ymax =grid[g_id]   
            grid_train = train[(tr.x > xmin) & (tr.x < xmax) & (tr.y > ymin) & (tr.y < ymax)]    

            place_counts = grid_train.place_id.value_counts()
            mask = (place_counts[grid_train.place_id.values] >= th).values
            grid_train = grid_train.loc[mask]

            grid_test = test.loc[test.grid_cell == g_id]
            row_ids = grid_test.index
            
            p = mp_pool.apply_async(process_grid, (g_id, grid_train, grid_test))
            processes.append([p, row_ids, g_id])

        while len(processes) > pool_size:
            mp_p, mp_row_ids, mp_g_id = processes.pop(0)
            mp_ind, mp_kprob = mp_p.get()
            indices[mp_row_ids] = mp_ind
            knn_prob[mp_row_ids] = mp_kprob
            grid_num[mp_row_ids] = mp_g_id
            if not no_cache:
                mp_cache_name = "%s/gid_%i.pkl" % (mdl_path, mp_g_id)
                pickle.dump([mp_row_ids, mp_ind, mp_kprob, mp_g_id], open(mp_cache_name, 'wb'))
                print("dump g_id=%i results to %s @ %s" % (mp_g_id, mp_cache_name, datetime.now()))

    mp_pool.close()

    while processes:
        mp_p, mp_row_ids, mp_g_id = processes.pop(0)
        mp_ind, mp_kprob = mp_p.get()
        indices[mp_row_ids] = mp_ind
        knn_prob[mp_row_ids] = mp_kprob
        grid_num[mp_row_ids] = mp_g_id
        if not no_cache:
            mp_cache_name = "%s/gid_%i.pkl" % (mdl_path, mp_g_id)
            pickle.dump([mp_row_ids, mp_ind, mp_kprob, mp_g_id], open(mp_cache_name, 'wb'))
            print("dump g_id=%i results to %s @ %s" % (mp_g_id, cache_name, datetime.now()))

    print('process_all_grids complete @ %s' % datetime.now())
    return indices, knn_prob, grid_num


def treva(train, test, repeats, grid, cross_validation):
    indices, knn_prob, grid_num = process_all_grids(train, test, repeats, grid)
    
    ### create indices for probability lookup tables. For example: (for placeid = 999999999 and weekday = 5, then the index is wkday_ind = 99999999905)
    train['wkday_ind'] = 10*train['place_id']+np.floor(train['weekday']).astype(np.int64)   
    train['hr_ind'] = 100*train['place_id']+np.floor(train['hour']).astype(np.int64)
    train['four_hour_ind'] = 100*train['place_id']+np.floor(train['four_hour']).astype(np.int64)

    table_cache = "./data/cache/feats_table_cv%s.pkl" % cross_validation
    if os.path.exists(table_cache):
        weekday, hour, four_hour = pickle.load(open(table_cache, 'rb'))
    else:
        weekday = makeprobtable(train, 'wkday_ind', 0.001)
        hour = makeprobtable(train, 'hr_ind', 0.001)
        four_hour = makeprobtable(train, 'four_hour_ind', 0.001)
        pickle.dump([weekday, hour, four_hour], open(table_cache, 'wb'))

    nn=10
    wkday_indices=10*indices+np.tile(np.floor(test.weekday[:,None]).astype(np.int64),nn )
    hr_indices=100*indices+np.tile(np.floor(test.hour[:,None]).astype(np.int64),nn )
    four_hour_indices=100*indices+np.tile(np.floor(test.four_hour[:,None]).astype(np.int64),nn )

    weekday_prob = getprob(wkday_indices, weekday, nn)
    hour_prob = getprob(hr_indices, hour, nn)
    four_hour_prob = getprob(four_hour_indices, four_hour, nn)
    total_prob = np.log10(four_hour_prob)*0.1 \
                    + np.log10(knn_prob)*1 \
                    + np.log10(hour_prob)*0.1 \
                    + np.log10(weekday_prob)*0.4

    # total_prob_sorted = np.sort(total_prob)[:,::-1] 
    max3index = np.argsort(-total_prob)
    a = np.indices(max3index.shape)[0]
    max3placeids = indices[a,max3index]
    print('treva complete @ %s' % datetime.now())
    return max3index, max3placeids, total_prob, indices, knn_prob, grid_num


def conclude(cross_validation, test, ytest, max3index, max3placeids, indices):
    if cross_validation==1: 
        indices = ([1/1.0, 1/2.0, 1/3.0]*(ytest[:,None] == indices[:,0:3]) ).sum()/indices[np.nonzero(indices[:,0])].shape[0]
        map3 = ([1/1.0, 1/2.0, 1/3.0]*(ytest[:,None] == max3placeids[:,0:3]) ).sum()/max3placeids[np.nonzero(max3placeids[:,0])].shape[0]
        ## calculation assumes unique values  
        print('indices: %.5f, map@3: %.5f' % (indices, map3))
     
        ## calculate map3 for each grid 
        max3placeids1 = pd.DataFrame({'row_id':test.index.values, 'grid_cell': test['grid_cell'], 'ytest': ytest.values, 'id1':max3placeids[:,0],'id2':max3placeids[:,1],'id3':max3placeids[:,2]} )                  
        gridwisemap3 = max3placeids1.groupby('grid_cell').apply(calcgridwisemap3)
        print("[Finish!] @ %s" % (datetime.now()))
        return indices, map3
    else:
        print('writing submission file...')
        max3placeids = pd.DataFrame({'row_id':test.index.values,'id1':max3placeids[:,0],'id2':max3placeids[:,1],'id3':max3placeids[:,2]} )
        max3placeids['place_id']=max3placeids.id1.astype(str).str.cat([max3placeids.id2.astype(str),max3placeids.id3.astype(str)], sep = ' ')       
        
        sfile = './submit/%s_%s.csv' % (alg, stamp)
        max3placeids[['row_id','place_id']].to_csv(sfile, header=True, index=False)
        print("[Finish!] @ %s" % datetime.now())
        if False:
            submiter.submiter().submit(entry=sfile, message="knn2")
        return None, None


def wrapper(cross_validation, n_cell_x, n_cell_y, wx=490, wy=980, grid_onecell=None, no_cache=False):
    meta_path = "./submit/c%s" % alg
    if not os.path.exists(meta_path): os.mkdir(meta_path)
    cv_file = "%s/c%s_cv%i_x%i_y%i_wx%i_wy%i.pkl" % (meta_path, alg, cross_validation, n_cell_x, n_cell_y, wx, wy)
    if (not no_cache) and os.path.exists(cv_file):
        print("%s exists, skip!" % cv_file)
        return 0, cv_file
    else:
        # 0-x,1-y,2-hour,3-weekday,4-month, 5-year, 6 - accuracy, 7-nearestneighbors 
        train, test, ytrain, ytest = get_data(cross_validation, n_cell_x, n_cell_y)
        size_x, size_y = 10./n_cell_x, 10./n_cell_y
        grid = extendgrid(size_x, size_y, n_cell_x, n_cell_y)
        repeats = [grid_onecell] if grid_onecell else list(range(n_cell_x*n_cell_y))
        max3index, max3placeids, total_prob, indices, knn_prob, grid_num = treva(train, test, repeats, grid, cross_validation)
        score = conclude(cross_validation, test, ytest, max3index, max3placeids, indices)
        # dump info
        results = {
            'max3index'     : max3index,
            'max3placeids'  : max3placeids,
            'total_prob'    : total_prob,
            'indices'       : indices,
            'knn_prob'      : knn_prob,
            'grid_num'      : grid_num,
            'score'         : score,
            'ytest'         : ytest,
        }
        if not no_cache: pickle.dump(results, open(cv_file, 'wb'))
        return score, cv_file
    

#------[blendor]------------------------------------------
def apk(actual, predicted, k=3):
  if len(predicted) > k: 
    predicted = predicted[:k]
  score, num_hits = 0.0, 0.0
  for i,p in enumerate(predicted):
    if p in actual and p not in predicted[:i]:
      num_hits += 1.0
      score += num_hits / (i+1.0)
  if not actual: return 0.0
  return score / min(len(actual), k)


def blendor(preds, mdl_weights, ytest=None):
    print("[blendor] start @ %s" % datetime.now())
    scnt = preds[0]['total_prob'].shape[0]
    blended_bests = []
    for i in range(scnt):
        stat = defaultdict(float)
        for p, w in zip(preds, mdl_weights):
            # for pid, v in zip(p['indices'][i], p['knn_prob'][i]):
            for pid, v in zip(p['indices'][i], p['total_prob'][i]):
                stat[pid] += w*(10**max(v, -10000)) 
        stat = sorted(stat.items(), key=lambda v: v[1], reverse=True)
        stat = [pid for pid,val in stat][:3]
        blended_bests.append(stat)
    print("[blendor] done @ %s" % datetime.now())
    if ytest is not None:
        match = [apk([ans], vals) for ans, vals in zip(ytest, blended_bests)]
        score = sum(match)/len(match)
        print("[valid scoring]: %s @ %s" % (score, datetime.now()))
    return blended_bests


def blending_flow(va_paths, te_paths, top_w=2, submit=False):
    va_preds = [pickle.load(open(path, 'rb')) for path in va_paths]
    te_preds = [pickle.load(open(path, 'rb')) for path in te_paths]
    
    scores = [v['score'][1] for v in va_preds]
    best_mdl = scores.index(max(scores))
    mdl_weights = [(top_w if mi == best_mdl else 1) for mi in range(len(va_preds))]
    print("scores=%s, mdl_weights=%s" % (scores, mdl_weights))

    # blending
    _ = blendor(va_preds, mdl_weights, ytest=va_preds[0]['ytest'])
    blended_submits = blendor(te_preds, mdl_weights, ytest=None)
    
    # output
    output = "./submit/knn2_blended_%s.csv" % (stamp)
    df = pd.DataFrame(blended_submits)
    df['row_id'] = df.index
    df['place_id'] = df[[0,1,2]].astype(str).apply(lambda x: ' '.join(x), axis=1)
    df.drop([0,1,2], axis=1, inplace=True)
    df[['row_id', 'place_id']].sort_values(by='row_id').to_csv(output, index=False)
    print("submit file written in %s @ %s" % (output, datetime.now()))
    if submit:
        submiter.submiter().submit(entry=output, message="knn2")


def main(params):
    global alg, weights, stamp, mdl_config, grid_onecell, pool_size
    alg = params['alg']
    stamp = params['stamp']
    cv = params['cv']
    pool_size = (mp.cpu_count() if ('knn' in alg) else 2)
    grid_onecell = (200 if cv else None)
    nx = 20 #params['nx']
    ny = 40 #params['ny']
    weights = np.tile(np.array([490.0, 980.0, 4.0, 3.1, 2.1, 10.0, 10.0, 36])[:,None],nx*ny).T
    print("alg=%s, stamp=%s, cv=%s, grid_onecell=%s" % (alg, stamp, cv, grid_onecell))

    if 'gs' in stamp:
        print("[Start] doing Grid Search for %s" % alg)
        for grid_onecell in [777]:
            for n_estimators in [800]:
                for max_feats in [0.5]:
                    for max_depth in [15]:
                        for bootstrap in [True, False]: # 7 is best!
                            mdl_config = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_feats, 'bootstrap': bootstrap}
                            print("[GS]:", mdl_config)
                            wrapper(cv, nx, ny, grid_onecell=grid_onecell, no_cache=True)
    elif stamp in ['knn_blend', 'knnz_blend']:
        print("[Start] doing %s for %s" % (stamp, alg))
        params = [
            (20, 20),
            (20, 40),
            (40, 20),
            (40, 40),

            (30, 30),
            (20, 30),
            (30, 20),

            (30, 40),
            (40, 30),
        ]
        va_paths, te_paths = [], []
        for nx, ny in params:
            print("[start] params=%i/%i @ %s" % (nx, ny, datetime.now()))
            weights = np.tile(np.array([490.0, 980.0, 4.0, 3.1, 2.1, 10.0, 10.0, 36])[:,None],nx*ny).T
            score, cv_file_va = wrapper(1, nx, ny)
            _, cv_file_te = wrapper(0, nx, ny)
            va_paths.append(cv_file_va)
            te_paths.append(cv_file_te)
        blending_flow(va_paths, te_paths, top_w=1, submit=False)
    else:
        print("[Start] doing single %s with cv=%s" % (alg, cv))
        mdl_config = {}
        wrapper(cv, nx, ny, grid_onecell=grid_onecell, no_cache=True)

