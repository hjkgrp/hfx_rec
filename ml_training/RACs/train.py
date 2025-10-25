import pandas as pd
import numpy as np
import sklearn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

def error_metrics(test,pred,train):
    """
    This function returns several error metrics
    """
    #Evaluate the model's performance
    mae = mean_absolute_error(test,pred)
    #rmse = root_mean_squared_error(test,pred)
    r2 = r2_score(test, pred)
    smae= mae/(train.max()-train.min())
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Scaled MAE: {smae:.2f}")
    #print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R-squared: {r2:.2f}")

def model_fit_cv(model,param_grid,X_train,y_train,X_test,X_val,X_train2,cv):
    """
    This function carries out grid search and fits the model
    """
    clf=GridSearchCV(model, param_grid, cv=cv)
    clf.fit(X_train, y_train)
    print("Best parameters:", clf.best_params_)
    print("Best estimator:", clf.best_estimator_)
    print("Best score:", clf.best_score_)
    y_pred_test = clf.best_estimator_.predict(X_test)
    y_pred_val = clf.best_estimator_.predict(X_val)
    y_pred_train = clf.best_estimator_.predict(X_train2)
  
    return y_pred_test,y_pred_val,y_pred_train,clf

def save_data(vss_df, csd_df, train_y, train_preds, val_y, val_preds, test_y, test_preds, csd_y, csd_preds, model, functional, seed):
    train_df = pd.DataFrame(train_y)
    train_df[f'{functional}-{model}-' + str(seed)] = train_preds
    train_df[f'{functional}-{model}-set-' + str(seed)] = 'train'
    
    val_df = pd.DataFrame(val_y)
    val_df[f'{functional}-{model}-' + str(seed)] = val_preds
    val_df[f'{functional}-{model}-set-' + str(seed)] = 'val'
    
    test_df = pd.DataFrame(test_y)
    test_df[f'{functional}-{model}-' + str(seed)] = test_preds
    test_df[f'{functional}-{model}-set-' + str(seed)] = 'test'
    
    pred_df = pd.concat([train_df, val_df, test_df]).drop('hfx_' + functional, axis=1)
    vss_df = pd.concat([vss_df, pred_df], axis=1)
    
    csd = pd.DataFrame(csd_y)
    csd[f'{functional}-{model}-' + str(seed)] = csd_preds
    
    csd_df = pd.concat([csd_df, csd.drop('hfx_' + functional, axis=1)], axis=1)
    
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

scan_vss = pd.concat([vss_racs, vss_targets['hfx_scan']], axis=1).dropna()
scan_csd = pd.concat([csd_racs, csd_targets['hfx_scan']], axis=1).dropna()

vss_output = vss_targets
csd_output = csd_targets

for rannum in tqdm([1234, 1337, 9876, 18558, 66464, 91625]):
    # Getting data from the above into y and X

    #clip the targets so they are between 0 and 100
    pbe_vss_y = pbe_vss['hfx_pbe'].astype(float).clip(0, 100)
    pbe_csd_y = pbe_csd['hfx_pbe'].astype(float).clip(0, 100)
    
    scan_vss_y = scan_vss['hfx_scan'].astype(float).clip(0, 100)
    scan_csd_y = scan_csd['hfx_scan'].astype(float).clip(0, 100)
    
    #the features are all remaining columns (everything but the first one)
    pbe_vss_X = pbe_vss.iloc[:, 1:].copy().apply(pd.to_numeric)
    pbe_csd_X = pbe_csd.iloc[:, 1:].copy().apply(pd.to_numeric)
    
    scan_vss_X = scan_vss.iloc[:, :-1].copy().apply(pd.to_numeric)
    scan_csd_X = scan_csd.iloc[:, :-1].copy().apply(pd.to_numeric)
    
    #split VSS-452 into 20% holdout, and then 80/20 on the remaining set (64/16) overall for train/val
    pbe_X_temp, pbe_X_test, pbe_y_temp, pbe_y_test = train_test_split(
        pbe_vss_X, pbe_vss_y, test_size=0.20, random_state=rannum
    )
    
    pbe_X_train, pbe_X_val, pbe_y_train, pbe_y_val = train_test_split(
        pbe_X_temp, pbe_y_temp, test_size=0.20, random_state=rannum
    )
    
    scan_X_temp, scan_X_test, scan_y_temp, scan_y_test = train_test_split(
        scan_vss_X, scan_vss_y, test_size=0.20, random_state=rannum
    )
    
    scan_X_train, scan_X_val, scan_y_train, scan_y_val = train_test_split(
        scan_X_temp, scan_y_temp, test_size=0.20, random_state=rannum
    )
    
    #drop invariant columns based on train only - this removes the number of equitorial and axial ligands since all monodentate in VSS-452
    keep_cols = pbe_X_train.columns[pbe_X_train.nunique(dropna=False) > 1]
    pbe_X_train   = pbe_X_train[keep_cols]
    pbe_X_val    = pbe_X_val[keep_cols]
    pbe_X_test  = pbe_X_test[keep_cols]
    pbe_X_csd = pbe_csd_X[keep_cols]
    
    print('PBE dataset sizes:', pbe_X_train.shape, pbe_X_val.shape, pbe_X_test.shape, pbe_X_csd.shape)
    
    keep_cols = scan_X_train.columns[scan_X_train.nunique(dropna=False) > 1]
    scan_X_train   = scan_X_train[keep_cols]
    scan_X_val    = scan_X_val[keep_cols]
    scan_X_test  = scan_X_test[keep_cols]
    scan_X_csd = scan_csd_X[keep_cols]
    
    print('SCAN dataset sizes:', scan_X_train.shape, scan_X_val.shape, scan_X_test.shape, scan_X_csd.shape)
    
    #normalize based on only the training subset
    pbe_train_mean = pbe_X_train.mean(axis=0)
    pbe_train_std = pbe_X_train.std(axis=0)
    
    pbe_X_train_scaled = (pbe_X_train-pbe_train_mean)/pbe_train_std
    pbe_X_val_scaled = (pbe_X_val-pbe_train_mean)/pbe_train_std
    pbe_X_test_scaled = (pbe_X_test-pbe_train_mean)/pbe_train_std
    pbe_X_csd_scaled = (pbe_X_csd-pbe_train_mean)/pbe_train_std
    
    scan_train_mean = scan_X_train.mean(axis=0)
    scan_train_std = scan_X_train.std(axis=0)
    
    scan_X_train_scaled = (scan_X_train-scan_train_mean)/scan_train_std
    scan_X_val_scaled = (scan_X_val-scan_train_mean)/scan_train_std
    scan_X_test_scaled = (scan_X_test-scan_train_mean)/scan_train_std
    scan_X_csd_scaled = (scan_X_csd-scan_train_mean)/scan_train_std
    
    #Combine Train + Val for GridSearchCV
    pbe_X_combined_scaled = pd.concat([pbe_X_train_scaled, pbe_X_val_scaled], axis=0, ignore_index=True)
    pbe_y_combined        = pd.concat([pbe_y_train, pbe_y_val], axis=0, ignore_index=True)
    
    scan_X_combined_scaled = pd.concat([scan_X_train_scaled, scan_X_val_scaled], axis=0, ignore_index=True)
    scan_y_combined        = pd.concat([scan_y_train, scan_y_val], axis=0, ignore_index=True)
    
    #Create a PredefinedSplit: -1 for training, 0 for validation
    pbe_split_index = [-1] * len(pbe_X_train_scaled) + [0] * len(pbe_X_val_scaled)
    pbe_ps = PredefinedSplit(test_fold=pbe_split_index)
    
    scan_split_index = [-1] * len(scan_X_train_scaled) + [0] * len(scan_X_val_scaled)
    scan_ps = PredefinedSplit(test_fold=scan_split_index)

    #train RF
    param_grid = {
        'n_estimators': [25,50,75,100,200],
        'max_depth': [2,5,10,20,30],
        'min_samples_leaf': [1,2,3,4,5,10]}
    cv=pbe_ps
    rf_model = RandomForestRegressor(random_state=rannum)
    
    pbe_y_pred_rf_test,pbe_y_pred_rf_val,pbe_y_pred_rf_train,pbe_clf_rf= model_fit_cv(
        rf_model,param_grid,pbe_X_combined_scaled,pbe_y_combined,pbe_X_test_scaled,pbe_X_val_scaled,pbe_X_train_scaled,cv)
    pbe_y_pred_rf_csd=pbe_clf_rf.best_estimator_.predict(pbe_X_csd_scaled)
    vss_output, csd_output = save_data(vss_output, csd_output, pbe_y_train, pbe_y_pred_rf_train,
                                       pbe_y_val, pbe_y_pred_rf_val, pbe_y_test, pbe_y_pred_rf_test,
                                       pbe_csd_y, pbe_y_pred_rf_csd, 'rf', 'pbe', rannum)
    
    print("Training errors")
    error_metrics(pbe_y_train,pbe_y_pred_rf_train,pbe_y_train)
    print("\nValidation errors")
    error_metrics(pbe_y_val,pbe_y_pred_rf_val,pbe_y_train)
    print("\nTest errors")
    error_metrics(pbe_y_test,pbe_y_pred_rf_test,pbe_y_train)
    print ("\n CSD-76 Test errors")
    error_metrics(pbe_csd_y,pbe_y_pred_rf_csd,pbe_y_train)
    
    cv=scan_ps
    rf_model = RandomForestRegressor(random_state=rannum)
    
    scan_y_pred_rf_test,scan_y_pred_rf_val,scan_y_pred_rf_train,scan_clf_rf=model_fit_cv(
        rf_model,param_grid,scan_X_combined_scaled,scan_y_combined,scan_X_test_scaled,scan_X_val_scaled,scan_X_train_scaled,cv)
    scan_y_pred_rf_csd=scan_clf_rf.best_estimator_.predict(scan_X_csd_scaled)
    vss_output, csd_output = save_data(vss_output, csd_output, scan_y_train, scan_y_pred_rf_train,
                                       scan_y_val, scan_y_pred_rf_val, scan_y_test, scan_y_pred_rf_test,
                                       scan_csd_y, scan_y_pred_rf_csd, 'rf', 'scan', rannum)
    
    print("Training errors")
    error_metrics(scan_y_train,scan_y_pred_rf_train,scan_y_train)
    print("\nValidation errors")
    error_metrics(scan_y_val,scan_y_pred_rf_val,scan_y_train)
    print("\nTest errors")
    error_metrics(scan_y_test,scan_y_pred_rf_test,scan_y_train)
    print ("\n CSD-76 Test errors")
    error_metrics(scan_csd_y,scan_y_pred_rf_csd,scan_y_train)


    #train MLP
    nlayers = (1, 2, 3, 4, 5)
    nhidden = (8, 16, 32, 64, 128)
    
    sizes = [[x]*y for x in nhidden for y in nlayers]
    
    param_grid = {
        'hidden_layer_sizes': sizes,
        'alpha': [1e-5, 1e-4, 1e-3]}
    cv=pbe_ps
    mlp_model = MLPRegressor(random_state=rannum,solver='adam',max_iter=10000, early_stopping=True)
    
    pbe_y_pred_mlp_test,pbe_y_pred_mlp_val,pbe_y_pred_mlp_train,pbe_clf_mlp=model_fit_cv(
        mlp_model,param_grid,pbe_X_combined_scaled,pbe_y_combined,pbe_X_test_scaled,pbe_X_val_scaled,pbe_X_train_scaled,cv)
    pbe_y_pred_mlp_csd=pbe_clf_mlp.best_estimator_.predict(pbe_X_csd_scaled)
    vss_output, csd_output = save_data(vss_output, csd_output, pbe_y_train, pbe_y_pred_mlp_train,
                                       pbe_y_val, pbe_y_pred_mlp_val, pbe_y_test, pbe_y_pred_mlp_test,
                                       pbe_csd_y, pbe_y_pred_mlp_csd, 'mlp', 'pbe', rannum)
    
    print("Training errors")
    error_metrics(pbe_y_train,pbe_y_pred_mlp_train,pbe_y_train)
    print("\nValidation errors")
    error_metrics(pbe_y_val,pbe_y_pred_mlp_val,pbe_y_train)
    print("\nTest errors")
    error_metrics(pbe_y_test,pbe_y_pred_mlp_test,pbe_y_train)
    print ("\n CSD-76 Test errors")
    error_metrics(pbe_csd_y,pbe_y_pred_mlp_csd,pbe_y_train)
    
    cv=scan_ps
    mlp_model = MLPRegressor(random_state=rannum,solver='adam',max_iter=10000, early_stopping=True)
    
    scan_y_pred_mlp_test,scan_y_pred_mlp_val,scan_y_pred_mlp_train,scan_clf_mlp=model_fit_cv(
        mlp_model,param_grid,scan_X_combined_scaled,scan_y_combined,scan_X_test_scaled,scan_X_val_scaled,scan_X_train_scaled,cv)
    scan_y_pred_mlp_csd=scan_clf_mlp.best_estimator_.predict(scan_X_csd_scaled)
    vss_output, csd_output = save_data(vss_output, csd_output, scan_y_train, scan_y_pred_mlp_train,
                                       scan_y_val, scan_y_pred_mlp_val, scan_y_test, scan_y_pred_mlp_test,
                                       scan_csd_y, scan_y_pred_mlp_csd, 'mlp', 'scan', rannum)
    
    print("Training errors")
    error_metrics(scan_y_train,scan_y_pred_mlp_train,scan_y_train)
    print("\nValidation errors")
    error_metrics(scan_y_val,scan_y_pred_mlp_val,scan_y_train)
    print("\nTest errors")
    error_metrics(scan_y_test,scan_y_pred_mlp_test,scan_y_train)
    print ("\n CSD-76 Test errors")
    error_metrics(scan_csd_y,scan_y_pred_mlp_csd,scan_y_train)


    #XGBoost model
    param_grid = { 'n_estimators': [50,100,200],
                   'max_depth': [3,5,7,10,20],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8,1.0],
                    'colsample_bytree': [0.8,1.0]}
    cv=pbe_ps
    xgb_model=xgb.XGBRegressor(objective='reg:squarederror',random_state=rannum)
    
    pbe_y_pred_xgb_test,pbe_y_pred_xgb_val,pbe_y_pred_xgb_train,pbe_clf_xgb=model_fit_cv(
        xgb_model,param_grid,pbe_X_combined_scaled,pbe_y_combined,pbe_X_test_scaled,pbe_X_val_scaled,pbe_X_train_scaled,cv)
    pbe_y_pred_xgb_csd=pbe_clf_xgb.best_estimator_.predict(pbe_X_csd_scaled)
    vss_output, csd_output = save_data(vss_output, csd_output, pbe_y_train, pbe_y_pred_xgb_train,
                                       pbe_y_val, pbe_y_pred_xgb_val, pbe_y_test, pbe_y_pred_xgb_test,
                                       pbe_csd_y, pbe_y_pred_xgb_csd, 'xgb', 'pbe', rannum)
    
    print("Training errors")
    error_metrics(pbe_y_train,pbe_y_pred_xgb_train,pbe_y_train)
    print("\nValidation errors")
    error_metrics(pbe_y_val,pbe_y_pred_xgb_val,pbe_y_train)
    print("\nTest errors")
    error_metrics(pbe_y_test,pbe_y_pred_xgb_test,pbe_y_train)
    print ("\n CSD-76 Test errors")
    error_metrics(pbe_csd_y,pbe_y_pred_xgb_csd,pbe_y_train)
    
    cv=scan_ps
    xgb_model=xgb.XGBRegressor(objective='reg:squarederror',random_state=rannum)
    
    scan_y_pred_xgb_test,scan_y_pred_xgb_val,scan_y_pred_xgb_train,scan_clf_xgb=model_fit_cv(
        xgb_model,param_grid,scan_X_combined_scaled,scan_y_combined,scan_X_test_scaled,scan_X_val_scaled,scan_X_train_scaled,cv)
    scan_y_pred_xgb_csd=scan_clf_xgb.best_estimator_.predict(scan_X_csd_scaled)
    vss_output, csd_output = save_data(vss_output, csd_output, scan_y_train, scan_y_pred_xgb_train,
                                       scan_y_val, scan_y_pred_xgb_val, scan_y_test, scan_y_pred_xgb_test,
                                       scan_csd_y, scan_y_pred_xgb_csd, 'xgb', 'scan', rannum)
    
    print("Training errors")
    error_metrics(scan_y_train,scan_y_pred_xgb_train,scan_y_train)
    print("\nValidation errors")
    error_metrics(scan_y_val,scan_y_pred_xgb_val,scan_y_train)
    print("\nTest errors")
    error_metrics(scan_y_test,scan_y_pred_xgb_test,scan_y_train)
    print ("\n CSD-76 Test errors")
    error_metrics(scan_csd_y,scan_y_pred_xgb_csd,scan_y_train)

vss_output.to_csv('vss_racs_predictions.csv')
csd_output.to_csv('csd_racs_predictions.csv')
