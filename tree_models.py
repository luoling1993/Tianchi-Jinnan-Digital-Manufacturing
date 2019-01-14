#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold

from selector import Selector

warnings.filterwarnings(action='ignore')

BASE_PATH = os.path.join('data')
ETLDATA_PATH = os.path.join(BASE_PATH, 'EtlData')


def get_data(selector=True):
    if selector:
        data_name = os.path.join(ETLDATA_PATH, 'selector_features.csv')
    else:
        data_name = os.path.join(ETLDATA_PATH, 'features.csv')

    df = pd.read_csv(data_name, header=0)
    return df


def lgb_model(train_data, test_data, params, n_fold=5):
    # boosting model
    columns = train_data.columns

    remove_columns = ['sample_id', 'rate']
    features_columns = [column for column in columns if column not in remove_columns]

    train_labels = train_data['rate']
    train_x = train_data[features_columns]
    test_x = test_data[features_columns]

    kfolder = KFold(n_splits=n_fold, shuffle=True, random_state=2018)
    kfold = kfolder.split(train_x, train_labels)

    preds_list = list()
    lgb_oof = np.zeros(train_data.shape[0])
    for train_index, vali_index in kfold:
        k_x_train = train_x.loc[train_index]
        k_y_train = train_labels.loc[train_index]
        k_x_vali = train_x.loc[vali_index]
        k_y_vali = train_labels.loc[vali_index]

        k_train = lgb.Dataset(k_x_train, k_y_train)
        k_vali = lgb.Dataset(k_x_vali, k_y_vali)

        num_round = 10000
        gbm = lgb.train(params, k_train, valid_sets=[k_train, k_vali], verbose_eval=False,
                        num_boost_round=num_round, early_stopping_rounds=100)

        k_pred = gbm.predict(k_x_vali, num_iteration=gbm.best_iteration)
        lgb_oof[vali_index] = k_pred
        k_mse = mean_squared_error(k_y_vali, k_pred)
        print(f'lgb mse error is {k_mse}')

        preds = gbm.predict(test_x, num_iteration=gbm.best_iteration)
        preds_list.append(preds)

    cv_mse_error = mean_squared_error(train_labels, lgb_oof)
    print(f'lgb cv mse error is {cv_mse_error}')
    preds_columns = ['preds_{id}'.format(id=i) for i in range(n_fold)]
    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_list = list(preds_df.mean(axis=1))
    lgb_prediction = preds_list

    sub_df = pd.DataFrame({'sample_id': test_data['sample_id'],
                           'rate': preds_list})
    sub_df.to_csv('submittion_tree.csv', index=False, header=False)

    return lgb_oof, lgb_prediction


def xgb_model(train_data, test_data, params, n_fold=5):
    # boosting model
    columns = train_data.columns

    remove_columns = ['sample_id', 'rate']
    features_columns = [column for column in columns if column not in remove_columns]

    train_labels = train_data['rate']
    train_x = train_data[features_columns]
    test_x = test_data[features_columns]

    kfolder = KFold(n_splits=n_fold, shuffle=True, random_state=2018)
    kfold = kfolder.split(train_x, train_labels)

    preds_list = list()
    xgb_oof = np.zeros(train_data.shape[0])
    for train_index, vali_index in kfold:
        k_x_train = train_x.loc[train_index]
        k_y_train = train_labels.loc[train_index]
        k_x_vali = train_x.loc[vali_index]
        k_y_vali = train_labels.loc[vali_index]

        k_train = xgb.DMatrix(k_x_train, k_y_train)
        k_vali = xgb.DMatrix(k_x_vali, k_y_vali)

        num_round = 10000
        watch_list = [(k_train, 'train'), (k_vali, 'vali')]
        gbm = xgb.train(params, k_train, evals=watch_list, verbose_eval=False,
                        num_boost_round=num_round, early_stopping_rounds=200)

        k_pred = gbm.predict(xgb.DMatrix(k_x_vali), ntree_limit=gbm.best_ntree_limit)
        xgb_oof[vali_index] = k_pred
        k_mse = mean_squared_error(k_y_vali, k_pred)
        print(f'xgb mse error is {k_mse}')

        preds = gbm.predict(xgb.DMatrix(test_x), ntree_limit=gbm.best_ntree_limit)
        preds_list.append(preds)

    cv_mse_error = mean_squared_error(train_labels, xgb_oof)
    print(f'xgb cv mse error is {cv_mse_error}')
    preds_columns = ['preds_{id}'.format(id=i) for i in range(n_fold)]
    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_list = list(preds_df.mean(axis=1))
    xgb_prediction = preds_list

    sub_df = pd.DataFrame({'sample_id': test_data['sample_id'],
                           'rate': preds_list})
    sub_df.to_csv('submittion_tree.csv', index=False, header=False)

    return xgb_oof, xgb_prediction


def stacking_model(oof_list, prediction_list, labels, sample_ids):
    train_stack = np.vstack(oof_list).transpose()
    test_stack = np.vstack(prediction_list).transpose()

    kfolder = RepeatedKFold(n_splits=5, n_repeats=2, random_state=666)
    kfold = kfolder.split(train_stack, labels)
    preds_list = list()
    stacking_oof = np.zeros(train_stack.shape[0])

    for train_index, vali_index in kfold:
        k_x_train = train_stack[train_index]
        k_y_train = labels.loc[train_index]
        k_x_vali = train_stack[vali_index]
        k_y_vali = labels.loc[vali_index]

        gbm = BayesianRidge(normalize=True)
        gbm.fit(k_x_train, k_y_train)

        k_pred = gbm.predict(k_x_vali)
        stacking_oof[vali_index] = k_pred
        k_mse = mean_squared_error(k_y_vali, k_pred)
        print(f'stacking mse error is {k_mse}')

        preds = gbm.predict(test_stack)
        preds_list.append(preds)

    cv_mse_error = mean_squared_error(labels, stacking_oof)
    print(f'stacking cv mse error is {cv_mse_error}')

    preds_columns = ['preds_{id}'.format(id=i) for i in range(10)]
    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_list = list(preds_df.mean(axis=1))

    sub_df = pd.DataFrame({'sample_id': sample_ids,
                           'rate': preds_list})
    sub_df.to_csv('submittion_tree.csv', index=False, header=False)


def get_selector():
    # selector columns
    dataset = get_data(selector=False)
    train_data = dataset[dataset['rate'] > 0.0]
    test_data = dataset[dataset['rate'] < 0.0]

    train_data.reset_index(inplace=True, drop=True)
    test_data.reset_index(inplace=True, drop=True)

    columns = train_data.columns
    remove_columns = ['sample_id', 'rate']
    features_columns = [column for column in columns if column not in remove_columns]
    x_train = train_data[features_columns]
    y_train = train_data['rate']
    selector = Selector()

    select_columns = selector.get_select_features(x_train, y_train)

    select_columns.extend(remove_columns)
    remove_columns = [column for column in train_data.columns if column not in select_columns]
    train_data = train_data.drop(columns=remove_columns)
    test_data = test_data.drop(columns=remove_columns)

    dataset = pd.concat([train_data, test_data])
    selector_name = os.path.join(ETLDATA_PATH, 'selector_features.csv')
    dataset.to_csv(selector_name, index=False)


def model_main(model, selector=True, force=True):
    lgb_params = {
        'boosting_type': 'gbdt',
        'num_leaves': 32,
        'max_depth': -1,
        'learning_rate': 0.01,
        'max_bin': 425,
        'objective': 'regression',
        'min_child_samples': 30,
        'subsample': 0.9,
        'subsample_freq': 1,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.1,
        'seed': 2018,
        'n_jobs': 5,
        'verbose': -1
    }

    xgb_params = {
        'eta': 0.01,
        'max_depth': 10,
        'max_bin': 425,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': True,
        'nthread': 4,
        'seed': 2018,
        'verbose': -1
    }

    assert model in ['lgb', 'xgb', 'stacking']
    selector_name = os.path.join(ETLDATA_PATH, 'selector_features.csv')
    if selector:
        if force:
            get_selector()
        elif os.path.exists(selector_name):
            pass
        else:
            raise FileNotFoundError(f'selector data file {selector_name} not found!')

        dataset = get_data(selector=True)
    else:
        dataset = get_data(selector=False)

    train_data = dataset[dataset['rate'] > 0.0]
    test_data = dataset[dataset['rate'] < 0.0]

    train_data.reset_index(inplace=True)
    test_data.reset_index(inplace=True)

    # # single model
    if model == 'lgb':
        lgb_model(train_data, test_data, lgb_params)
    elif model == 'xgb':
        xgb_model(train_data, test_data, xgb_params)

    # # stacking model
    else:
        lgb_oof, lgb_prediction = lgb_model(train_data, test_data, lgb_params)
        xgb_oof, xgb_prediction = xgb_model(train_data, test_data, xgb_params)
        sample_ids = test_data['sample_id']
        labels = train_data['rate']

        oof_list = [lgb_oof, xgb_oof]
        prediction_list = [lgb_prediction, xgb_prediction]
        stacking_model(oof_list, prediction_list, labels, sample_ids)


if __name__ == '__main__':
    model_main(model='stacking', selector=True, force=True)
