#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd


def get_data(name):
    assert name in ['automl', 'tree']

    data_name = f'submittion_{name}.csv'
    columns = ['sample_id', f'{name}_rate']
    df = pd.read_csv(data_name, header=-1, names=columns)
    return df


def blending_main(automl_rate=0.6):
    tree_rate = 1 - automl_rate
    automl_sub = get_data(name='automl')
    tree_sub = get_data(name='tree')

    sub = pd.merge(automl_sub, tree_sub, on='sample_id')
    sub['rate'] = sub['automl_rate'] * automl_rate + sub['tree_rate'] * tree_rate
    sub = sub.drop(columns=['automl_rate', 'tree_rate'])

    sub.to_csv('submittion.csv', header=False, index=False)


if __name__ == '__main__':
    blending_main()
