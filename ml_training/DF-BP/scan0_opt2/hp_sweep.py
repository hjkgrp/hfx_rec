import sys
sys.path.append('dfa_recommender')

import pickle
import numpy as np
import os
import pandas as pd
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

vss_features = pickle.load(open('../../../ml_features/BP_features/scan0-vss452_X.pkl', 'rb'))
csd_features = pickle.load(open('../../../ml_features/BP_features/scan0-csd76_X.pkl', 'rb'))

vss_df = pd.read_csv('../../../ml_features/BP_features/scan0-vss452_structures.csv')
vss_feature_df = pd.DataFrame(index=vss_df['Unnamed: 0'], columns=['features'])
for idx, row in vss_df.iterrows():
    vss_feature_df['features'][row['Unnamed: 0']] = vss_features[idx]

csd_df = pd.read_csv('../../../ml_features/BP_features/scan0-csd76_structures.csv')
csd_feature_df = pd.DataFrame(index=csd_df['Unnamed: 0'], columns=['features'])
for idx, row in csd_df.iterrows():
    csd_feature_df['features'][row['Unnamed: 0']] = csd_features[idx]

vss_df = pd.read_csv('../../../ml_features/BP_features/b3lyp-vss452_structures.csv')
vss_df2 = pd.read_csv('../../../ml_features/BP_features/pbe0-vss452_structures.csv')
vss_df3 = pd.read_csv('../../../ml_features/BP_features/scan0-vss452_structures.csv')

set1 = set(vss_df['Unnamed: 0'].to_list())
set2 = set(vss_df2['Unnamed: 0'].to_list())
set3 = set(vss_df3['Unnamed: 0'].to_list())
vss_common_structs = set1 & set2 & set3

csd_df = pd.read_csv('../../../ml_features/BP_features/b3lyp-csd76_structures.csv')
csd_df2 = pd.read_csv('../../../ml_features/BP_features/pbe0-csd76_structures.csv')
csd_df3 = pd.read_csv('../../../ml_features/BP_features/scan0-csd76_structures.csv')

set1 = set(csd_df['Unnamed: 0'].to_list())
set2 = set(csd_df2['Unnamed: 0'].to_list())
set3 = set(csd_df3['Unnamed: 0'].to_list())
csd_common_structs = set1 & set2 & set3

#load features and targets
csd_features = csd_feature_df.sort_index()
csd_X = csd_features['features'].to_numpy()
csd_targets = pd.read_csv('../../../data/CSD76targets.csv')
csd_targets = csd_targets.set_index(csd_targets['Unnamed: 0'])
for idx, row in csd_targets.iterrows():
    if idx in csd_features.index:
        pass
    else:
        csd_targets = csd_targets.drop([idx])
csd_targets = csd_targets.sort_index()
csd_structs = csd_features.index.to_list()

vss_features = vss_feature_df.sort_index()
vss_X = vss_features['features'].to_numpy()
vss_targets = pd.read_csv('../../../data/VSS452targets.csv')
vss_targets['Unnamed: 0'] = vss_targets['Unnamed: 0'].apply(lambda x: x.split('/')[-1])
vss_targets = vss_targets.set_index(vss_targets['Unnamed: 0'])
for idx, row in vss_targets.iterrows():
    if idx in vss_features.index:
        pass
    else:
        vss_targets = vss_targets.drop([idx])
vss_targets = vss_targets.sort_index()
vss_structs = vss_features.index.to_list()

#remove the structures that do not have targets
csd_pbe_y = csd_targets['hfx_pbe'].to_numpy()
csd_pbe_X = [csd_X[i] for i in range(len(csd_pbe_y)) if (not np.isnan(csd_pbe_y[i])) and (True not in np.isnan(csd_X[i]))]
csd_pbe_structs = [csd_structs[i] for i in range(len(csd_pbe_y)) if (not np.isnan(csd_pbe_y[i])) and (True not in np.isnan(csd_X[i]))]
csd_pbe_y = [[csd_pbe_y[i]] for i in range(len(csd_pbe_y)) if (not np.isnan(csd_pbe_y[i])) and (True not in np.isnan(csd_X[i]))]
for idx, elem in enumerate(csd_pbe_y):
    if elem[0] > 100:
        csd_pbe_y[idx] = [100]
    elif elem[0] < 0:
        csd_pbe_y[idx] = [0]

csd_scan_y = csd_targets['hfx_scan'].to_numpy()
csd_scan_X = [csd_X[i] for i in range(len(csd_scan_y)) if (not np.isnan(csd_scan_y[i])) and (True not in np.isnan(csd_X[i]))]
csd_scan_structs = [csd_structs[i] for i in range(len(csd_scan_y)) if (not np.isnan(csd_scan_y[i])) and (True not in np.isnan(csd_X[i]))]
csd_scan_y = [[csd_scan_y[i]] for i in range(len(csd_scan_y)) if (not np.isnan(csd_scan_y[i])) and (True not in np.isnan(csd_X[i]))]
for idx, elem in enumerate(csd_scan_y):
    if elem[0] > 100:
        csd_scan_y[idx] = [100]
    elif elem[0] < 0:
        csd_scan_y[idx] = [0]

vss_pbe_y = vss_targets['hfx_pbe'].to_numpy()
vss_pbe_X = [vss_X[i] for i in range(len(vss_pbe_y)) if (not np.isnan(vss_pbe_y[i])) and (True not in np.isnan(vss_X[i]))]
vss_pbe_structs = [vss_structs[i] for i in range(len(vss_pbe_y)) if (not np.isnan(vss_pbe_y[i])) and (True not in np.isnan(vss_X[i]))]
vss_pbe_y = [[vss_pbe_y[i]] for i in range(len(vss_pbe_y)) if (not np.isnan(vss_pbe_y[i])) and (True not in np.isnan(vss_X[i]))]
for idx, elem in enumerate(vss_pbe_y):
    if elem[0] > 100:
        vss_pbe_y[idx] = [100]
    elif elem[0] < 0:
        vss_pbe_y[idx] = [0]

vss_scan_y = vss_targets['hfx_scan'].to_numpy()
vss_scan_X = [vss_X[i] for i in range(len(vss_scan_y)) if (not np.isnan(vss_scan_y[i])) and (True not in np.isnan(vss_X[i]))]
vss_scan_structs = [vss_structs[i] for i in range(len(vss_scan_y)) if (not np.isnan(vss_scan_y[i])) and (True not in np.isnan(vss_X[i]))]
vss_scan_y = [[vss_scan_y[i]] for i in range(len(vss_scan_y)) if (not np.isnan(vss_scan_y[i])) and (True not in np.isnan(vss_X[i]))]
for idx, elem in enumerate(vss_scan_y):
    if elem[0] > 100:
        vss_scan_y[idx] = [100]
    elif elem[0] < 0:
        vss_scan_y[idx] = [0]

'''
#replicate the original train and validation sets
vss_pbe_train_idxs = []
vss_pbe_val_idxs = []

for idx, struct in enumerate(vss_pbe_structs):
    if vss_df.loc[vss_df['name'] == struct]['train'].item():
        vss_pbe_train_idxs.append(idx)
    else:
        vss_pbe_val_idxs.append(idx)

vss_scan_train_idxs = []
vss_scan_val_idxs = []

for idx, struct in enumerate(vss_scan_structs):
    if vss_df.loc[vss_df['name'] == struct]['train'].item():
        vss_scan_train_idxs.append(idx)
    else:
        vss_scan_val_idxs.append(idx)
'''

#random train/val split that ensures the same training set between the two
#find structures that are in common between two
in_common1 = set(vss_pbe_structs).intersection(set(vss_scan_structs)) & vss_common_structs
in_common2 = set(vss_pbe_structs).intersection(set(vss_scan_structs)) - vss_common_structs

#get training, testing indices based on the structures in common
np.random.seed(2)
vss_pbe_common_idxs1 = [i for i in range(len(vss_pbe_structs)) if vss_pbe_structs[i] in in_common1]
vss_scan_common_idxs1 = [i for i in range(len(vss_scan_structs)) if vss_scan_structs[i] in in_common1]
vss_pbe_common_idxs2 = [i for i in range(len(vss_pbe_structs)) if vss_pbe_structs[i] in in_common2]
vss_scan_common_idxs2 = [i for i in range(len(vss_scan_structs)) if vss_scan_structs[i] in in_common2]

train_idxs1 = np.random.choice(len(vss_pbe_common_idxs1), int(0.9*len(vss_pbe_common_idxs1)), replace=False)
train_idxs2 = np.random.choice(len(vss_pbe_common_idxs2), int(0.9*len(vss_pbe_common_idxs2)), replace=False)

vss_pbe_train_idxs = [vss_pbe_common_idxs1[i] for i in train_idxs1] + [vss_pbe_common_idxs2[i] for i in train_idxs2]
vss_pbe_val_idxs = [i for i in range(len(vss_pbe_y)) if i not in vss_pbe_train_idxs]
vss_scan_train_idxs = [vss_scan_common_idxs1[i] for i in train_idxs1] + [vss_scan_common_idxs2[i] for i in train_idxs2]
vss_scan_val_idxs = [i for i in range(len(vss_scan_y)) if i not in vss_scan_train_idxs]

#make the (unscaled) datasets
vss_pbe_train_X = [vss_pbe_X[i] for i in vss_pbe_train_idxs]
vss_pbe_val_X = [vss_pbe_X[i] for i in vss_pbe_val_idxs]
vss_scan_train_X = [vss_scan_X[i] for i in vss_scan_train_idxs]
vss_scan_val_X = [vss_scan_X[i] for i in vss_scan_val_idxs]
#note: pbe and scan train features are identical, can just use one
vss_pbe_train_y = [vss_pbe_y[i] for i in vss_pbe_train_idxs]
vss_pbe_val_y = [vss_pbe_y[i] for i in vss_pbe_val_idxs]
vss_scan_train_y = [vss_scan_y[i] for i in vss_scan_train_idxs]
vss_scan_val_y = [vss_scan_y[i] for i in vss_scan_val_idxs]

#scale by VSS-452 training features
pbe_target_scaler = StandardScaler().fit(vss_pbe_train_y)
scan_target_scaler = StandardScaler().fit(vss_scan_train_y)

vss_pbe_train_y = pbe_target_scaler.transform(vss_pbe_train_y)
vss_scan_train_y = scan_target_scaler.transform(vss_scan_train_y)
vss_pbe_val_y = pbe_target_scaler.transform(vss_pbe_val_y)
vss_scan_val_y = scan_target_scaler.transform(vss_scan_val_y)
csd_pbe_y = pbe_target_scaler.transform(csd_pbe_y)
csd_scan_y = scan_target_scaler.transform(csd_scan_y)

vss_pbe_train_y = np.ravel(vss_pbe_train_y)
vss_scan_train_y = np.ravel(vss_scan_train_y)
vss_pbe_val_y = np.ravel(vss_pbe_val_y)
vss_scan_val_y = np.ravel(vss_scan_val_y)
csd_pbe_y = np.ravel(csd_pbe_y)
csd_scan_y = np.ravel(csd_scan_y)

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from dfa_recommender.net import GatedNetwork
from dfa_recommender.dataset import SubsetDataset
from dfa_recommender.sampler import InfiniteSampler
from dfa_recommender.ml_utils import numpy_to_dataset
from dfa_recommender.evaluate import evaluate_regressor

import numpy as np
import pickle
import pandas as pd
import copy

def weighted_mse_loss(input, target, weights = 1):
    out = torch.absolute(input-target) * weights
    loss = out.mean() 
    return loss

#hyperparameter optimization
from hyperopt import tpe, hp, fmin

#for PBE

def objective(params):
    n_out, n_hidden, n_layers, droprate = params['n_out'], params['n_hidden'], params['n_layers'], params['droprate']
    
    X_train = np.array(vss_pbe_train_X)
    X_val = np.array(vss_pbe_val_X)
    X_test = np.array(csd_pbe_X)
    
    y_train = np.array(vss_pbe_train_y)
    y_val = np.array(vss_pbe_val_y)
    y_test = np.array(csd_pbe_y)
    y_scaler = pbe_target_scaler
    
    atoms  = ["X", "H", "C", "N", "O", "F", "Cr", "Mn", "Fe", "Co"]
    
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cpu')
    num_workers = 0
    bz = 16
    stop_iter = 30
    
    data_tr, data_te = numpy_to_dataset(X_train, y_train, regression=True), numpy_to_dataset(X_val, y_val, regression=True)
    tr_l = SubsetDataset(data_tr, list(range(len(data_tr))))
    te_l = SubsetDataset(data_te, list(range(len(data_te))))
    #print("sub labeled dataset length: ", len(tr_l), len(te_l))
    
    # ---build and train--- 
    pbe_cls = GatedNetwork(nin=58*2, n_out=n_out, n_hidden=n_hidden, 
                       n_layers=n_layers, droprate=droprate,
                       elements=list(range(len(atoms)))).to(device)  # vertsse
    #nin variably set to X_org.shape[-1] -1 
    pbe_cls.train()
    optimizer = AdamW(list(pbe_cls.parameters()),
                      lr=2e-4,
                      betas=(0.90, 0.999),
                      weight_decay=1e-2,
                      amsgrad=True,
                      )
    scheduler = ExponentialLR(optimizer, gamma=0.999)
    l_tr_iter = iter(DataLoader(tr_l, bz, num_workers=num_workers,
                                sampler=InfiniteSampler(len(tr_l))))
    l_te_iter = iter(DataLoader(te_l, bz, num_workers=num_workers,
                                sampler=InfiniteSampler(len(te_l))))
    te_loader = DataLoader(te_l, len(te_l), num_workers=num_workers)
    tr_l_loader = DataLoader(tr_l, len(tr_l), num_workers=num_workers)
    
    mae_list, scaled_mae_list, rval_list = [], [], []
    min_scale_mae = 10000
    for epoch in range(2000):
        for niter in range(0, 1 + int(len(data_tr)/bz)):
            l_x, l_y = next(l_tr_iter)
            l_x, l_y = l_x.to(device), l_y.to(device)
    
            sup_reg_loss = weighted_mse_loss(pbe_cls(l_x), l_y, torch.ones(bz))
            
            unsup_reg_loss = sup_reg_loss
            loss = sup_reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    
        mae, scaled_mae, rval = evaluate_regressor(
            pbe_cls, te_loader, device, y_scaler)
        tr_mae, tr_scaled_mae, tr_rval = evaluate_regressor(
            pbe_cls, tr_l_loader, device, y_scaler)
        scaled_mae_list.append(scaled_mae)
        mae_list.append(mae)
        rval_list.append(rval)

        if len(scaled_mae_list) > stop_iter:
            if min(scaled_mae_list[-stop_iter:]) - min(scaled_mae_list[:-stop_iter]) > 0:
                break

    pbe_cls.eval()
    
    pbe_val_preds = []
    pbe_val_labels = []
    with torch.no_grad():
        for x, y in te_loader:
            _pred = pbe_cls(x.to(device))
            pbe_val_preds.append(_pred.cpu().numpy())
            pbe_val_labels.append(y.cpu().numpy())
    
    pbe_val_preds = y_scaler.inverse_transform(pbe_val_preds)[0]
    pbe_val_labels = y_scaler.inverse_transform(pbe_val_labels)[0]

    for idx, elem in enumerate(pbe_val_preds):
        if elem > 100:
            pbe_val_preds[idx] = 100
        elif elem < 0:
            pbe_val_preds[idx] = 0

    pbe_val_r2 = r2_score(pbe_val_labels, pbe_val_preds)
    pbe_val_mae = mean_absolute_error(pbe_val_labels, pbe_val_preds)

    return pbe_val_mae

n_out_options = [5, 8, 10]
n_hidden_options = [8, 16, 32, 64, 128]
n_layers_options = [1, 2, 3, 4, 5]
droprate_options = [0, 0.1, 0.2, 0.3]

search_space = {
    'n_out': hp.choice('n_out', n_out_options),
    'n_hidden': hp.choice('n_hidden', n_hidden_options),
    'n_layers': hp.choice('n_layers', n_layers_options),
    'droprate': hp.choice('droprate', droprate_options)
}

best = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=50, rstate=np.random.default_rng(5))

n_out = n_out_options[best['n_out']]
n_hidden = n_hidden_options[best['n_hidden']]
n_layers = n_layers_options[best['n_layers']]
droprate = droprate_options[best['droprate']]

with open('pbe_optimal_hps.txt', 'w+') as f:
    f.write(
f'''
n_out: {n_out}
n_hidden: {n_hidden}
n_layers: {n_layers}
droprate: {droprate}
'''
    )

#PBE best model

X_train = np.array(vss_pbe_train_X)
X_val = np.array(vss_pbe_val_X)
X_test = np.array(csd_pbe_X)

y_train = np.array(vss_pbe_train_y)
y_val = np.array(vss_pbe_val_y)
y_test = np.array(csd_pbe_y)
y_scaler = pbe_target_scaler

atoms  = ["X", "H", "C", "N", "O", "F", "Cr", "Mn", "Fe", "Co"]

torch.set_num_threads(4)
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cpu')
num_workers = 0
bz = 16
stop_iter = 30

data_tr, data_te = numpy_to_dataset(X_train, y_train, regression=True), numpy_to_dataset(X_val, y_val, regression=True)
tr_l = SubsetDataset(data_tr, list(range(len(data_tr))))
te_l = SubsetDataset(data_te, list(range(len(data_te))))
print("sub labeled dataset length: ", len(tr_l), len(te_l))

# ---build and train--- 
pbe_cls = GatedNetwork(nin=58*2, n_out=n_out, n_hidden=n_hidden, 
                   n_layers=n_layers, droprate=droprate,
                   elements=list(range(len(atoms)))).to(device)  # vertsse
#nin variably set to X_org.shape[-1] -1 
pbe_cls.train()
optimizer = AdamW(list(pbe_cls.parameters()),
                  lr=2e-4,
                  betas=(0.90, 0.999),
                  weight_decay=1e-2,
                  amsgrad=True,
                  )
scheduler = ExponentialLR(optimizer, gamma=0.999)
l_tr_iter = iter(DataLoader(tr_l, bz, num_workers=num_workers,
                            sampler=InfiniteSampler(len(tr_l))))
l_te_iter = iter(DataLoader(te_l, bz, num_workers=num_workers,
                            sampler=InfiniteSampler(len(te_l))))
te_loader = DataLoader(te_l, len(te_l), num_workers=num_workers)
tr_l_loader = DataLoader(tr_l, len(tr_l), num_workers=num_workers)

mae_list, scaled_mae_list, rval_list = [], [], []
min_scale_mae = 10000
for epoch in range(2000):
    for niter in range(0, 1 + int(len(data_tr)/bz)):
        l_x, l_y = next(l_tr_iter)
        l_x, l_y = l_x.to(device), l_y.to(device)

        sup_reg_loss = weighted_mse_loss(pbe_cls(l_x), l_y, torch.ones(bz))
        
        unsup_reg_loss = sup_reg_loss
        loss = sup_reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    mae, scaled_mae, rval = evaluate_regressor(
        pbe_cls, te_loader, device, y_scaler)
    tr_mae, tr_scaled_mae, tr_rval = evaluate_regressor(
        pbe_cls, tr_l_loader, device, y_scaler)
    scaled_mae_list.append(scaled_mae)
    mae_list.append(mae)
    rval_list.append(rval)
        
    print('Iter {} train_mae {:.3} train_scaled_mae {:.3} train_rval {:.3} mae {:.3} scaled_mae {:.3} rval {:.3} SupLoss {:.3} UnsupLoss {:.3}'.format(
        epoch, tr_mae, tr_scaled_mae, tr_rval, mae, scaled_mae, rval, sup_reg_loss.item(), unsup_reg_loss.item()))
    if scaled_mae < min_scale_mae:
        print("copying best model with scaled mae: ", scaled_mae)
        min_scale_mae = scaled_mae
        best_model = copy.deepcopy(pbe_cls)
    if len(scaled_mae_list) > stop_iter:
        if min(scaled_mae_list[-stop_iter:]) - min(scaled_mae_list[:-stop_iter]) > 0:
            print("EarlyStopping.", min(
                scaled_mae_list[-stop_iter:]), min(scaled_mae_list[:-stop_iter]))
            break

# ---save model---
with open("BP_model-pbe.pkl", "wb") as fo:
    pickle.dump(best_model, fo)
pbe_cls = best_model

pbe_cls.eval()
pbe_train_preds = []
pbe_train_labels = []
with torch.no_grad():
    for x, y in tr_l_loader:
        _pred = pbe_cls(x.to(device))
        pbe_train_preds.append(_pred.cpu().numpy())
        pbe_train_labels.append(y.cpu().numpy())

pbe_train_preds = y_scaler.inverse_transform(pbe_train_preds)[0]
pbe_train_labels = y_scaler.inverse_transform(pbe_train_labels)[0]

pbe_val_preds = []
pbe_val_labels = []
with torch.no_grad():
    for x, y in te_loader:
        _pred = pbe_cls(x.to(device))
        pbe_val_preds.append(_pred.cpu().numpy())
        pbe_val_labels.append(y.cpu().numpy())

pbe_val_preds = y_scaler.inverse_transform(pbe_val_preds)[0]
pbe_val_labels = y_scaler.inverse_transform(pbe_val_labels)[0]

data_csd = numpy_to_dataset(X_test, y_test, regression=True)
csd_l = SubsetDataset(data_csd, list(range(len(data_csd))))
csd_loader = DataLoader(csd_l, len(csd_l), num_workers=num_workers)

pbe_test_preds = []
pbe_test_labels = []
with torch.no_grad():
    for x, y in csd_loader:
        _pred = pbe_cls(x.to(device))
        pbe_test_preds.append(_pred.cpu().numpy())
        pbe_test_labels.append(y.cpu().numpy())

pbe_test_preds = y_scaler.inverse_transform(pbe_test_preds)[0]
pbe_test_labels = y_scaler.inverse_transform(pbe_test_labels)[0]

for arr in [pbe_train_preds, pbe_val_preds, pbe_test_preds]:
    for idx, elem in enumerate(arr):
        if elem > 100:
            arr[idx] = 100
        elif elem < 0:
            arr[idx] = 0

pbe_train_r2 = r2_score(pbe_train_labels, pbe_train_preds)
pbe_train_mae = mean_absolute_error(pbe_train_labels, pbe_train_preds)
pbe_val_r2 = r2_score(pbe_val_labels, pbe_val_preds)
pbe_val_mae = mean_absolute_error(pbe_val_labels, pbe_val_preds)
pbe_test_r2 = r2_score(pbe_test_labels, pbe_test_preds)
pbe_test_mae = mean_absolute_error(pbe_test_labels, pbe_test_preds)

with open('pbe_best_model.txt', 'w+') as f:
    f.write(
f"""
pbe_train_r2: {pbe_train_r2}
pbe_train_mae: {pbe_train_mae}
pbe_val_r2: {pbe_val_r2}
pbe_val_mae: {pbe_val_mae}
pbe_test_r2: {pbe_test_r2}
pbe_test_mae: {pbe_test_mae}
"""
           )

#for SCAN

def objective(params):
    n_out, n_hidden, n_layers, droprate = params['n_out'], params['n_hidden'], params['n_layers'], params['droprate']
    
    X_train = np.array(vss_scan_train_X)
    X_val = np.array(vss_scan_val_X)
    X_test = np.array(csd_scan_X)
    
    y_train = np.array(vss_scan_train_y)
    y_val = np.array(vss_scan_val_y)
    y_test = np.array(csd_scan_y)
    y_scaler = scan_target_scaler
    
    atoms  = ["X", "H", "C", "N", "O", "F", "Cr", "Mn", "Fe", "Co"]
    
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cpu')
    num_workers = 0
    bz = 16
    stop_iter = 30
    
    data_tr, data_te = numpy_to_dataset(X_train, y_train, regression=True), numpy_to_dataset(X_val, y_val, regression=True)
    tr_l = SubsetDataset(data_tr, list(range(len(data_tr))))
    te_l = SubsetDataset(data_te, list(range(len(data_te))))
    
    # ---build and train--- 
    scan_cls = GatedNetwork(nin=58*2, n_out=n_out, n_hidden=n_hidden, 
                       n_layers=n_layers, droprate=droprate,
                       elements=list(range(len(atoms)))).to(device)  # vertsse
    #nin variably set to X_org.shape[-1] -1 
    scan_cls.train()
    optimizer = AdamW(list(scan_cls.parameters()),
                      lr=2e-4,
                      betas=(0.90, 0.999),
                      weight_decay=1e-2,
                      amsgrad=True,
                      )
    scheduler = ExponentialLR(optimizer, gamma=0.999)
    l_tr_iter = iter(DataLoader(tr_l, bz, num_workers=num_workers,
                                sampler=InfiniteSampler(len(tr_l))))
    l_te_iter = iter(DataLoader(te_l, bz, num_workers=num_workers,
                                sampler=InfiniteSampler(len(te_l))))
    te_loader = DataLoader(te_l, len(te_l), num_workers=num_workers)
    tr_l_loader = DataLoader(tr_l, len(tr_l), num_workers=num_workers)
    
    mae_list, scaled_mae_list, rval_list = [], [], []
    min_scale_mae = 10000
    for epoch in range(2000):
        for niter in range(0, 1 + int(len(data_tr)/bz)):
            l_x, l_y = next(l_tr_iter)
            l_x, l_y = l_x.to(device), l_y.to(device)
    
            sup_reg_loss = weighted_mse_loss(scan_cls(l_x), l_y, torch.ones(bz))
            
            unsup_reg_loss = sup_reg_loss
            loss = sup_reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    
        mae, scaled_mae, rval = evaluate_regressor(
            scan_cls, te_loader, device, y_scaler)
        tr_mae, tr_scaled_mae, tr_rval = evaluate_regressor(
            scan_cls, tr_l_loader, device, y_scaler)
        scaled_mae_list.append(scaled_mae)
        mae_list.append(mae)
        rval_list.append(rval)

        if len(scaled_mae_list) > stop_iter:
            if min(scaled_mae_list[-stop_iter:]) - min(scaled_mae_list[:-stop_iter]) > 0:
                break

    scan_cls.eval()
    
    scan_val_preds = []
    scan_val_labels = []
    with torch.no_grad():
        for x, y in te_loader:
            _pred = scan_cls(x.to(device))
            scan_val_preds.append(_pred.cpu().numpy())
            scan_val_labels.append(y.cpu().numpy())
    
    scan_val_preds = y_scaler.inverse_transform(scan_val_preds)[0]
    scan_val_labels = y_scaler.inverse_transform(scan_val_labels)[0]

    for idx, elem in enumerate(scan_val_preds):
        if elem > 100:
            scan_val_preds[idx] = 100
        elif elem < 0:
            scan_val_preds[idx] = 0
    
    scan_val_r2 = r2_score(scan_val_labels, scan_val_preds)
    scan_val_mae = mean_absolute_error(scan_val_labels, scan_val_preds)

    return scan_val_mae

n_out_options = [5, 8, 10]
n_hidden_options = [8, 16, 32, 64, 128]
n_layers_options = [1, 2, 3, 4, 5]
droprate_options = [0, 0.1, 0.2, 0.3]

search_space = {
    'n_out': hp.choice('n_out', n_out_options),
    'n_hidden': hp.choice('n_hidden', n_hidden_options),
    'n_layers': hp.choice('n_layers', n_layers_options),
    'droprate': hp.choice('droprate', droprate_options)
}

best = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=50, rstate=np.random.default_rng(5))

n_out = n_out_options[best['n_out']]
n_hidden = n_hidden_options[best['n_hidden']]
n_layers = n_layers_options[best['n_layers']]
droprate = droprate_options[best['droprate']]

with open('scan_optimal_hps.txt', 'w+') as f:
    f.write(
f'''
n_out: {n_out}
n_hidden: {n_hidden}
n_layers: {n_layers}
droprate: {droprate}
'''
    )

#SCAN best model

X_train = np.array(vss_scan_train_X)
X_val = np.array(vss_scan_val_X)
X_test = np.array(csd_scan_X)

y_train = np.array(vss_scan_train_y)
y_val = np.array(vss_scan_val_y)
y_test = np.array(csd_scan_y)
y_scaler = scan_target_scaler

atoms  = ["X", "H", "C", "N", "O", "F", "Cr", "Mn", "Fe", "Co"]

torch.set_num_threads(4)
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cpu')
num_workers = 0
bz = 16
stop_iter = 30

data_tr, data_te = numpy_to_dataset(X_train, y_train, regression=True), numpy_to_dataset(X_val, y_val, regression=True)
tr_l = SubsetDataset(data_tr, list(range(len(data_tr))))
te_l = SubsetDataset(data_te, list(range(len(data_te))))
print("sub labeled dataset length: ", len(tr_l), len(te_l))

# ---build and train--- 
scan_cls = GatedNetwork(nin=58*2, n_out=n_out, n_hidden=n_hidden, 
                   n_layers=n_layers, droprate=droprate,
                   elements=list(range(len(atoms)))).to(device)  # vertsse
#nin variably set to X_org.shape[-1] -1 
scan_cls.train()
optimizer = AdamW(list(scan_cls.parameters()),
                  lr=2e-4,
                  betas=(0.90, 0.999),
                  weight_decay=1e-2,
                  amsgrad=True,
                  )
scheduler = ExponentialLR(optimizer, gamma=0.999)
l_tr_iter = iter(DataLoader(tr_l, bz, num_workers=num_workers,
                            sampler=InfiniteSampler(len(tr_l))))
l_te_iter = iter(DataLoader(te_l, bz, num_workers=num_workers,
                            sampler=InfiniteSampler(len(te_l))))
te_loader = DataLoader(te_l, len(te_l), num_workers=num_workers)
tr_l_loader = DataLoader(tr_l, len(tr_l), num_workers=num_workers)

mae_list, scaled_mae_list, rval_list = [], [], []
min_scale_mae = 10000
for epoch in range(2000):
    for niter in range(0, 1 + int(len(data_tr)/bz)):
        l_x, l_y = next(l_tr_iter)
        l_x, l_y = l_x.to(device), l_y.to(device)

        sup_reg_loss = weighted_mse_loss(scan_cls(l_x), l_y, torch.ones(bz))
        
        unsup_reg_loss = sup_reg_loss
        loss = sup_reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    mae, scaled_mae, rval = evaluate_regressor(
        scan_cls, te_loader, device, y_scaler)
    tr_mae, tr_scaled_mae, tr_rval = evaluate_regressor(
        scan_cls, tr_l_loader, device, y_scaler)
    scaled_mae_list.append(scaled_mae)
    mae_list.append(mae)
    rval_list.append(rval)
        
    print('Iter {} train_mae {:.3} train_scaled_mae {:.3} train_rval {:.3} mae {:.3} scaled_mae {:.3} rval {:.3} SupLoss {:.3} UnsupLoss {:.3}'.format(
        epoch, tr_mae, tr_scaled_mae, tr_rval, mae, scaled_mae, rval, sup_reg_loss.item(), unsup_reg_loss.item()))
    if scaled_mae < min_scale_mae:
        print("copying best model with scaled mae: ", scaled_mae)
        min_scale_mae = scaled_mae
        best_model = copy.deepcopy(scan_cls)
    if len(scaled_mae_list) > stop_iter:
        if min(scaled_mae_list[-stop_iter:]) - min(scaled_mae_list[:-stop_iter]) > 0:
            print("EarlyStopping.", min(
                scaled_mae_list[-stop_iter:]), min(scaled_mae_list[:-stop_iter]))
            break

# ---save model---
with open("BP_model-scan.pkl", "wb") as fo:
    pickle.dump(best_model, fo)
scan_cls = best_model

scan_cls.eval()
scan_train_preds = []
scan_train_labels = []
with torch.no_grad():
    for x, y in tr_l_loader:
        _pred = scan_cls(x.to(device))
        scan_train_preds.append(_pred.cpu().numpy())
        scan_train_labels.append(y.cpu().numpy())

scan_train_preds = y_scaler.inverse_transform(scan_train_preds)[0]
scan_train_labels = y_scaler.inverse_transform(scan_train_labels)[0]

scan_val_preds = []
scan_val_labels = []
with torch.no_grad():
    for x, y in te_loader:
        _pred = scan_cls(x.to(device))
        scan_val_preds.append(_pred.cpu().numpy())
        scan_val_labels.append(y.cpu().numpy())

scan_val_preds = y_scaler.inverse_transform(scan_val_preds)[0]
scan_val_labels = y_scaler.inverse_transform(scan_val_labels)[0]

data_csd = numpy_to_dataset(X_test, y_test, regression=True)
csd_l = SubsetDataset(data_csd, list(range(len(data_csd))))
csd_loader = DataLoader(csd_l, len(csd_l), num_workers=num_workers)

scan_test_preds = []
scan_test_labels = []
with torch.no_grad():
    for x, y in csd_loader:
        _pred = scan_cls(x.to(device))
        scan_test_preds.append(_pred.cpu().numpy())
        scan_test_labels.append(y.cpu().numpy())

scan_test_preds = y_scaler.inverse_transform(scan_test_preds)[0]
scan_test_labels = y_scaler.inverse_transform(scan_test_labels)[0]

for arr in [scan_train_preds, scan_val_preds, scan_test_preds]:
    for idx, elem in enumerate(arr):
        if elem > 100:
            arr[idx] = 100
        elif elem < 0:
            arr[idx] = 0

scan_train_r2 = r2_score(scan_train_labels, scan_train_preds)
scan_train_mae = mean_absolute_error(scan_train_labels, scan_train_preds)
scan_val_r2 = r2_score(scan_val_labels, scan_val_preds)
scan_val_mae = mean_absolute_error(scan_val_labels, scan_val_preds)
scan_test_r2 = r2_score(scan_test_labels, scan_test_preds)
scan_test_mae = mean_absolute_error(scan_test_labels, scan_test_preds)

with open('scan_best_model.txt', 'w+') as f:
    f.write(
f"""
scan_train_r2: {scan_train_r2}
scan_train_mae: {scan_train_mae}
scan_val_r2: {scan_val_r2}
scan_val_mae: {scan_val_mae}
scan_test_r2: {scan_test_r2}
scan_test_mae: {scan_test_mae}
"""
           )

#save results
pbe_preds = pd.DataFrame(index=csd_pbe_structs, data={'pred':pbe_test_preds, 'label':pbe_test_labels})
scan_preds = pd.DataFrame(index=csd_scan_structs, data={'pred':scan_test_preds, 'label':scan_test_labels})

pred_df = pd.DataFrame(index=list(set(csd_pbe_structs + csd_scan_structs)), columns=['PBE Target', 'PBE Prediction', 'SCAN Target', 'SCAN Prediction'])

for _, row in pred_df.iterrows():
    reference_pbe = csd_targets.loc[row.name]['hfx_pbe']
    if reference_pbe > 100:
        reference_pbe = 100
    elif reference_pbe < 0:
        reference_pbe = 0
    reference_scan = csd_targets.loc[row.name]['hfx_scan']
    if reference_scan > 100:
        reference_scan = 100
    elif reference_scan < 0:
        reference_scan = 0
    row['PBE Target'] = reference_pbe
    row['SCAN Target'] = reference_scan
    if np.isclose(pbe_preds['label'][row.name], reference_pbe, atol=1e-5):
        row['PBE Prediction'] = pbe_preds['pred'][row.name]
    if row.name in scan_preds.index and np.isclose(scan_preds['label'][row.name], reference_scan, atol=1e-5):
        row['SCAN Prediction'] = scan_preds['pred'][row.name]

pred_df.to_csv('BP_predictions_hyperparams-test.csv')

pbe_idx = [vss_pbe_structs[i] for i in vss_pbe_val_idxs]
scan_idx = [vss_scan_structs[i] for i in vss_scan_val_idxs]

pbe_preds = pd.DataFrame(index=pbe_idx, data={'pred':pbe_val_preds, 'label':pbe_val_labels})
scan_preds = pd.DataFrame(index=scan_idx, data={'pred':scan_val_preds, 'label':scan_val_labels})

pred_df = pd.DataFrame(index=list(set(pbe_idx + scan_idx)), columns=['PBE Target', 'PBE Prediction', 'SCAN Target', 'SCAN Prediction'])

for _, row in pred_df.iterrows():
    reference_pbe = vss_targets.loc[row.name]['hfx_pbe']
    if reference_pbe > 100:
        reference_pbe = 100
    elif reference_pbe < 0:
        reference_pbe = 0
    reference_scan = vss_targets.loc[row.name]['hfx_scan']
    if reference_scan > 100:
        reference_scan = 100
    elif reference_scan < 0:
        reference_scan = 0
    row['PBE Target'] = reference_pbe
    row['SCAN Target'] = reference_scan
    if row.name in pbe_preds.index and np.isclose(pbe_preds['label'][row.name], reference_pbe, atol=1e-5):
        row['PBE Prediction'] = pbe_preds['pred'][row.name]
    if row.name in scan_preds.index and np.isclose(scan_preds['label'][row.name], reference_scan, atol=1e-5):
        row['SCAN Prediction'] = scan_preds['pred'][row.name]

pred_df.to_csv('BP_predictions_hyperparams-val.csv')

pbe_idx = [vss_pbe_structs[i] for i in vss_pbe_train_idxs]
scan_idx = [vss_scan_structs[i] for i in vss_scan_train_idxs]

pbe_preds = pd.DataFrame(index=pbe_idx, data={'pred':pbe_train_preds, 'label':pbe_train_labels})
scan_preds = pd.DataFrame(index=scan_idx, data={'pred':scan_train_preds, 'label':scan_train_labels})

pred_df = pd.DataFrame(index=list(set(pbe_idx + scan_idx)), columns=['PBE Target', 'PBE Prediction', 'SCAN Target', 'SCAN Prediction'])

for _, row in pred_df.iterrows():
    reference_pbe = vss_targets.loc[row.name]['hfx_pbe']
    if reference_pbe > 100:
        reference_pbe = 100
    elif reference_pbe < 0:
        reference_pbe = 0
    reference_scan = vss_targets.loc[row.name]['hfx_scan']
    if reference_scan > 100:
        reference_scan = 100
    elif reference_scan < 0:
        reference_scan = 0
    row['PBE Target'] = reference_pbe
    row['SCAN Target'] = reference_scan
    if row.name in pbe_preds.index and np.isclose(pbe_preds['label'][row.name], reference_pbe, atol=1e-5):
        row['PBE Prediction'] = pbe_preds['pred'][row.name]
    if row.name in scan_preds.index and np.isclose(scan_preds['label'][row.name], reference_scan, atol=1e-5):
        row['SCAN Prediction'] = scan_preds['pred'][row.name]

pred_df.to_csv('BP_predictions_hyperparams-train.csv')