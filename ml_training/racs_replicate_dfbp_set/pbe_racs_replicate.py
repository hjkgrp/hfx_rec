import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid

np.random.seed(5)
import random
random.seed(5)

#for getting energy errors

from scipy.interpolate import interp1d

csd_sse_df = pd.read_csv('../../data/cleaned_csd76_sse.csv').set_index('Unnamed: 0')

csd_76 = pd.read_csv('../../data/CSD-76.csv').set_index('name')

csd_hfx_df = pd.read_csv('../../data/CSD76targets.csv').set_index('Unnamed: 0')

vss_sse_df = pd.read_csv('../../data/cleaned_vss452_sse.csv').set_index('Unnamed: 0')
names = {}
for name in vss_sse_df.index:
    elems = name.split('/')
    names[name] =elems[-1]
vss_sse_df = vss_sse_df.rename(index=names)

vss_hfx_df = pd.read_csv('../../data/VSS452targets.csv').set_index('Unnamed: 0')
names = {}
for name in vss_hfx_df.index:
    elems = name.split('/')
    names[name] =elems[-1]
vss_hfx_df = vss_hfx_df.rename(index=names)

vss_452 = pd.read_csv('../../data/VSS-452.csv')
vss_452 = vss_452.set_index(vss_452['name'])

#to replicate the PBE0 set

train = pd.read_csv('../DF-BP/pbe0_opt5/BP_predictions_hyperparams-train.csv').set_index('Unnamed: 0')
val = pd.read_csv('../DF-BP/pbe0_opt5/BP_predictions_hyperparams-val.csv').set_index('Unnamed: 0')
test = pd.read_csv('../DF-BP/pbe0_opt5/BP_predictions_hyperparams-test.csv').set_index('Unnamed: 0')

import pandas as pd
import numpy as np
import sklearn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

def save_data(vss_df, csd_df, train_y, train_preds, val_y, val_preds, test_y, test_preds, model, functional, seed):
    train_df = pd.DataFrame(train_y)
    train_df[f'{functional}-{model}-' + str(seed)] = train_preds
    train_df[f'{functional}-{model}-set-' + str(seed)] = 'train'
    
    val_df = pd.DataFrame(val_y)
    val_df[f'{functional}-{model}-' + str(seed)] = val_preds
    val_df[f'{functional}-{model}-set-' + str(seed)] = 'val'

    csd = pd.DataFrame(test_y)
    csd[f'{functional}-{model}-' + str(seed)] = test_preds
    
    csd_df = pd.concat([csd_df, csd.drop('hfx_' + functional, axis=1)], axis=1)
    
    pred_df = pd.concat([train_df, val_df]).drop('hfx_' + functional, axis=1)
    vss_df = pd.concat([vss_df, pred_df], axis=1)
    
    return vss_df, csd_df

vss_racs = pd.read_csv('../../ml_features/vss_racs.csv').set_index('Unnamed: 0')
vss_targets = pd.read_csv('../../data/VSS452targets.csv').set_index('Unnamed: 0')
#rename so that the convention matches the original VSS-452
names = {}
for name in vss_targets.index:
    elems = name.split('/')
    names[name] =elems[-1]
vss_targets = vss_targets.rename(index=names)
csd_racs = pd.read_csv('../../ml_features/csd_racs.csv').set_index('Unnamed: 0')
csd_targets = pd.read_csv('../../data/CSD76targets.csv').set_index('Unnamed: 0')

pbe_vss = pd.concat([vss_targets['hfx_pbe'], vss_racs], axis=1).dropna()
pbe_csd = pd.concat([csd_targets['hfx_pbe'], csd_racs], axis=1).dropna()

vss_output = vss_targets
csd_output = csd_targets

# Getting data from the above into y and X

#clip the targets so they are between 0 and 100
pbe_vss_y = pbe_vss['hfx_pbe'].astype(float).clip(0, 100)
pbe_csd_y = pbe_csd['hfx_pbe'].astype(float).clip(0, 100)

#the features are all remaining columns (everything but the first one)
pbe_vss_X = pbe_vss.iloc[:, 1:].copy().apply(pd.to_numeric)
pbe_csd_X = pbe_csd.iloc[:, 1:].copy().apply(pd.to_numeric)

pbe_X_train = pbe_vss_X.loc[train['PBE Target'].dropna().index]
pbe_X_val = pbe_vss_X.loc[val['PBE Target'].dropna().index]
pbe_X_test = pbe_csd_X
pbe_y_train = pbe_vss_y.loc[train['PBE Target'].dropna().index]
pbe_y_val = pbe_vss_y.loc[val['PBE Target'].dropna().index]
pbe_y_test = pbe_csd_y
    
#drop invariant columns based on train only - this removes the number of equitorial and axial ligands since all monodentate in VSS-452
keep_cols = pbe_X_train.columns[pbe_X_train.nunique(dropna=False) > 1]
pbe_X_train   = pbe_X_train[keep_cols]
pbe_X_val    = pbe_X_val[keep_cols]
pbe_X_test  = pbe_X_test[keep_cols]

print('PBE dataset sizes:', pbe_X_train.shape, pbe_X_val.shape, pbe_X_test.shape)

#normalize based on only the training subset
pbe_train_mean = pbe_X_train.mean(axis=0)
pbe_train_std = pbe_X_train.std(axis=0)

pbe_X_train_scaled = (pbe_X_train-pbe_train_mean)/pbe_train_std
pbe_X_val_scaled = (pbe_X_val-pbe_train_mean)/pbe_train_std
pbe_X_test_scaled = (pbe_X_test-pbe_train_mean)/pbe_train_std
    
param_grid = {
    'n_estimators': [25,50,75,100,200],
    'max_depth': [2,5,10,20,30],
    'min_samples_leaf': [1,2,3,4,5,10]}

best_g_pbe = np.nan
best_val_pbe = np.inf

for g in tqdm(ParameterGrid(param_grid)):
    rf = RandomForestRegressor(random_state=5)
    rf.set_params(**g)
    rf.fit(pbe_X_train_scaled, pbe_y_train)
    pbe_val_pred = rf.predict(pbe_X_val_scaled)

    val_mae = mean_absolute_error(pbe_y_val,pbe_val_pred)
    if val_mae < best_val_pbe:
        best_val_pbe = val_mae
        best_g_pbe = g

rf = RandomForestRegressor(random_state=5)
rf.set_params(**best_g_pbe)
rf.fit(pbe_X_train_scaled, pbe_y_train)
pbe_train_pred = rf.predict(pbe_X_train_scaled)
pbe_val_pred = rf.predict(pbe_X_val_scaled)
pbe_test_pred = rf.predict(pbe_X_test_scaled)

vss_output, csd_output = save_data(vss_output, csd_output, pbe_y_train, pbe_train_pred,
                                   pbe_y_val, pbe_val_pred, pbe_y_test, pbe_test_pred,
                                   'rf', 'pbe', 5)

vss_output.to_csv('pbe-vss_racs_predictions.csv')
csd_output.to_csv('pbe-csd_racs_predictions.csv')
