#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import os
import re
import warnings

import pandas as pd

warnings.filterwarnings(action='ignore')

BASE_PATH = os.path.join('data')
RAWDATA_PATH = os.path.join(BASE_PATH, 'RawData')
ETLDATA_PATH = os.path.join(BASE_PATH, 'EtlData')

time_columns = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
time_ts_columns = ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']


class FuckProcessing(object):
    @staticmethod
    def _get_data():
        train_data_name = os.path.join(RAWDATA_PATH, 'train.csv')
        train_data = pd.read_csv(train_data_name, encoding='gb18030', header=0)
        train_data.rename(columns={'样本id': 'sample_id', '收率': 'rate'}, inplace=True)

        train_data = train_data[train_data['rate'] >= 0.87]  # delte low rate recodes

        test_data_name = os.path.join(RAWDATA_PATH, 'test.csv')
        test_data = pd.read_csv(test_data_name, encoding='gb18030', header=0)
        test_data['收率'] = -1
        test_data.rename(columns={'样本id': 'sample_id', '收率': 'rate'}, inplace=True)

        dataset = pd.concat([train_data, test_data], axis=0, ignore_index=True, sort=False)
        return dataset

    @staticmethod
    def _get_clean_time(item):
        if item == -1:
            return -1

        if not item:
            return -1

        if pd.isnull(item):
            return -1

        # specialized processing
        if item == '1900/1/29 0:00':
            return '14:00:00'
        elif item == '1900/1/21 0:00':
            return '21:00:00'
        elif item == '1900/1/22 0:00':
            return '22:00:00'
        elif item == '1900/1/9 7:00':
            return '23:00:00'
        elif item == '700':
            return '7:00:00'
        elif item == ':30:00':
            return '0:30:00'
        elif item == '1900/1/1 2:30':
            return '21:30:00'
        elif item == '1900/1/12 0:00':
            return '12:00:00'
        elif item == '1900/3/13 0:00':
            return '13:00:00'

        true_time_fmt = '%H:%M:%S'
        try:
            item_datetime = datetime.datetime.strptime(item, true_time_fmt)
            return item_datetime.strftime('%H:%M:%S')
        except ValueError:
            raise ValueError(item)

    @staticmethod
    def _get_clean_time_ts(item):
        if item == -1:
            return -1

        if not item:
            return -1

        if pd.isnull(item):
            return -1

        # specialized processing
        if item == '15:00-1600':
            return '15:00-16:00'
        elif item == '19:-20:05':
            return '19:00-20:05'

        item = re.sub(';', ':', item)
        item = re.sub('；', ':', item)
        item = re.sub('"', ':', item)
        item = re.sub(':-', '-', item)
        item = re.sub('::', ':', item)

        true_time_ts_fmt = re.compile('([0-9]{1,2}:[0-9]{1,2}-[0-9]{1,2}:[0-9]{1,2})')
        match = re.search(true_time_ts_fmt, item)

        if match:
            return match.group(0)

        raise ValueError(item)

    @staticmethod
    def _get_clean_num(item):
        if not item:
            return -1

        try:
            float(item)
            return item
        except ValueError:
            return -1

    @staticmethod
    def _get_remove_columns(dataset):
        dataset = dataset.copy()

        dataset = dataset[dataset['rate'] > 0]  # 通过train_data筛选

        remove_columns = ['B3', 'B13', 'A13', 'A18', 'A23']  # 单一类别
        for column in dataset.columns:
            if column in remove_columns:
                continue

            biggest_rate = dataset[column].value_counts(normalize=True, dropna=False).values[0]
            if biggest_rate >= 0.9:
                remove_columns.append(column)

        return remove_columns

    @staticmethod
    def _time_to_num(time_str):
        if time_str == -1:
            return -1

        elif time_str == ':30:00':
            return 0.5
        elif time_str == '1900/1/9 7:00':
            return 7
        elif time_str == '1900/1/1 2:30':
            return 2.5

        try:
            hours, minutes, seconds = time_str.split(':')
        except ValueError:
            return 0

        try:
            hours = int(hours)
            minutes = int(minutes)
            seconds = int(seconds)

            time_num = hours + minutes / 60 + seconds / 3600
            return round(time_num, 2)
        except ValueError:
            return 0.5

    @staticmethod
    def _get_time_duration(time_str):
        if time_str == -1:
            return -1
        elif time_str == '19:-20:05':
            return 1
        elif time_str == '15:00-1600':
            return 1

        pattern = re.compile(r"\d+\.?\d*")
        start_hour, start_minute, end_hour, end_minute = re.findall(pattern, time_str)
        start_time = int(start_hour) + int(start_minute) / 60.0
        end_time = int(end_hour) + int(end_minute) / 60.0

        if start_time > end_time:
            return round((end_time - start_time) + 24.0, 2)
        else:
            return round((end_time - start_time), 2)

    @staticmethod
    def _get_sample_id(item):
        sample_id = item.split('_')[1]
        try:
            sample_id = int(sample_id)
        except ValueError:
            raise ValueError(f'error item:{item}')
        return sample_id

    @staticmethod
    def _get_operation_df(dataset):
        dataset = dataset.copy()

        dataset['helper_sum'] = (
                dataset['A1'] + dataset['A3'] + dataset['A4'] + dataset['A19'] + dataset['B1'] + dataset['B12'])

        # B14/(A1+A3+A4+A19+B1+B12)
        dataset['B14_helper_sum_rate'] = dataset['B14'] / dataset['helper_sum']

        dataset = dataset.drop(columns=['helper_sum'])
        return dataset

    def _data_encoder(self, dataset):
        dataset = dataset.copy()

        dataset['id'] = dataset['sample_id'].apply(self._get_sample_id)
        remove_columns = ['sample_id', 'rate', 'id']
        cate_columns = [column for column in dataset.columns if column not in remove_columns]

        for column in cate_columns:
            mapping_dict = dict(zip(dataset[column].unique(), range(0, dataset[column].nunique())))
            dataset[column] = dataset[column].map(mapping_dict)

        train_data = dataset[dataset['rate'] > 0.0]

        train_data['helper'] = pd.cut(train_data['rate'], 5, labels=False)
        train_data = pd.get_dummies(train_data, columns=['helper'])
        helper_columns = ['helper_0', 'helper_1', 'helper_2', 'helper_3', 'helper_4']

        # rate embedding + B14 embedding
        # 神仙特征
        b14_keys_length = train_data['B14'].nunique()
        for column in cate_columns:
            column_keys_length = train_data[column].nunique()
            if column_keys_length < b14_keys_length:
                # 神仙逻辑:只考虑column的度>=B14列的度的情况
                continue

            biggest_rate = train_data[column].value_counts(normalize=True, dropna=False).values[0]
            if biggest_rate > 0.9:
                continue

            for helper_column in helper_columns:
                b14_column_name = f'B14_{column}_{helper_column}_mean'
                column_df = train_data.groupby(by=[column])[helper_column].agg('mean').reset_index(name='mean')
                column_dict = column_df.set_index(column)['mean'].to_dict()

                dataset[b14_column_name] = dataset['B14'].map(column_dict)

        # One-Hot encoding
        dataset = pd.get_dummies(dataset, columns=cate_columns)

        return dataset

    def get_processing(self):
        dataset = self._get_data()

        # 业务特征
        dataset = self._get_operation_df(dataset)

        remove_columns = self._get_remove_columns(dataset)
        dataset = dataset.drop(columns=remove_columns)
        dataset = dataset.fillna(-1)

        dataset_columns = dataset.columns
        for column in dataset_columns:
            if column in time_ts_columns:
                # time_ts column
                column_name = f'{column}_duration'
                dataset[column_name] = dataset[column].apply(self._get_time_duration)

            if column in time_columns:
                # time column
                time_num_clumn_name = f'{column}_time_num'
                dataset[time_num_clumn_name] = dataset[column].apply(self._time_to_num)

        dataset = dataset.drop(columns=time_columns, errors='ignore')
        dataset = dataset.drop(columns=time_ts_columns, errors='ignore')
        dataset = self._data_encoder(dataset)

        return dataset


def processing_main():
    processing = FuckProcessing()
    dt = processing.get_processing()

    features_name = os.path.join(ETLDATA_PATH, 'features.csv')
    dt.to_csv(features_name, index=False, encoding='utf-8')


if __name__ == '__main__':
    processing_main()
