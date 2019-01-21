#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')


class Selector(object):
    @staticmethod
    def _get_xgb_model(**kwargs):
        xgb_params = {
            'eta': 0.01,
            'max_depth': 10,
            'max_bin': 425,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:linear',
            'eval_metric': 'rmse',
            'silent': True,
            'nthread': 3,
            'seed': 2018,
            'verbose': -1
        }
        for k, v in kwargs.items():
            xgb_params[k] = v
        xgb_model = xgb.XGBRegressor(**xgb_params)

        return xgb_model

    @staticmethod
    def _get_lgb_model(**kwargs):
        lgb_params = {
            "boosting_type": "gbdt",
            "num_leaves": 32,
            "max_depth": -1,
            "learning_rate": 0.05,
            "max_bin": 425,
            "objective": 'regression',
            "min_child_samples": 30,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.1,
            'metric': 'mse',
            "seed": 2018,
            "n_jobs": 5,
            "verbose": -1
        }
        for k, v in kwargs.items():
            lgb_params[k] = v

        lgb_model = lgb.LGBMRegressor(**lgb_params)

        return lgb_model

    @staticmethod
    def _get_importance_features(model, columns, topn=300):
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({'column': columns, 'score': feature_importance})
        importance_df = importance_df.sort_values(by=['score'], ascending=False).reset_index()
        importance_columns = importance_df['column'].loc[:topn].tolist()

        return importance_columns

    def _get_cv_error(self, x_train, y_train):
        model_list = [self._get_xgb_model(), self._get_lgb_model()]

        cv_error = 0.0
        mse = make_scorer(mean_squared_error)
        for model in model_list:
            mse_error = cross_val_score(model, x_train, y_train, scoring=mse, cv=5, n_jobs=5)
            mse_error = np.mean(mse_error)
            cv_error += mse_error

        model_length = len(model_list)
        cv_error = cv_error / model_length
        return cv_error

    def get_select_features(self, x_train, y_train):
        columns = x_train.columns

        importance_columns_list = list()
        for model in [self._get_xgb_model(), self._get_lgb_model()]:
            meta_model = model.fit(x_train, y_train)
            model_importance_columns = self._get_importance_features(meta_model, columns)
            importance_columns_list.extend(model_importance_columns)

        columns_num = 1
        select_columns = list()
        cv_error = 999.0
        importance_columns_set = set()

        for index, column in enumerate(importance_columns_list):
            if column in importance_columns_set:
                # set function will upset importance_columns_list order
                continue
            else:
                importance_columns_set.add(column)

            select_columns.append(column)
            x_train_sample = x_train[select_columns]
            tmp_cv_error = self._get_cv_error(x_train_sample, y_train)
            if tmp_cv_error < cv_error:
                cv_error = tmp_cv_error
                print(f'columns_num:{columns_num}\tindex_num:{index}\tcv_error:{cv_error}\tcolumn:{column}')
                columns_num += 1
            else:
                select_columns.pop()

        return select_columns
