#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.model_selection import KFold


class Stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    # model only support sklearn api
    base_models_ = None
    meta_models_ = None

    def __init__(self, base_models, meta_models, n_folds=5, seed=2018):
        self.base_models = base_models
        self.meta_models = meta_models
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, x_data, y_data):
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_models_ = clone(self.meta_models)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=2018)
        oof_predictions = np.zeros((x_data.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            for train_index, vali_index in kfold.split(x_data, y_data):
                instance = clone(model)

                self.base_models_[i].append(instance)
                instance.fit(x_data[train_index], y_data[train_index])

                y_pred = instance.predict(x_data[vali_index])
                oof_predictions[vali_index, i] = y_pred

        self.meta_models_.fit(oof_predictions, y_data)
        return self

    def predict(self, x_data):
        meta_features_list = list()
        for base_models in self.base_models_:
            base_models_stack_list = list()
            for model in base_models:
                base_models_stack_list.append(model.predict(x_data))
            meta_features_list.append(np.column_stack(base_models_stack_list).mean(axis=1))

        meta_features = np.column_stack(meta_features_list)

        return self.meta_models_.predict(meta_features)
