#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import re
import warnings
from functools import cmp_to_key

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings(action='ignore')

BASE_PATH = os.path.join('data')
RAWDATA_PATH = os.path.join(BASE_PATH, 'RawData')
ETLDATA_PATH = os.path.join(BASE_PATH, 'EtlData')

time_columns = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
time_ts_columns = ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']
material_columns = ['A1', 'A2', 'A3', 'A4', 'A19', 'B1', 'B12', 'B14']


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
        if not item:
            return ''

        if pd.isnull(item):
            return ''

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
        if not item:
            return ''

        if pd.isnull(item):
            return ''

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
            return ''

        try:
            float(item)
            return item
        except ValueError:
            return ''

    def _clear_dataset(self, dataset):
        # fucccccck data

        dataset = dataset.copy()
        dataset_columns = dataset.columns

        # time clear
        for column in time_columns:
            if column not in dataset_columns:
                continue
            dataset[column] = dataset[column].apply(self._get_clean_time)

        # time_ts clear
        for column in time_ts_columns:
            if column not in dataset_columns:
                continue
            dataset[column] = dataset[column].apply(self._get_clean_time_ts)

        # num clear
        num_columns = list()
        for column in dataset.columns:
            if column in ['sample_id', 'rate']:
                continue

            if column in time_columns:
                continue

            if column in time_ts_columns:
                continue

            num_columns.append(column)

        for column in num_columns:
            dataset[column] = dataset[column].apply(self._get_clean_num)

        return dataset

    @staticmethod
    def _my_division(item):
        item1 = item[0]
        item2 = item[1]

        if not item1:
            return 0
        if not item2:
            return -1

        return float(item1) / float(item2)

    def _get_operation_df(self, dataset):
        dataset = dataset.copy()

        # A22-A23 PH差
        dataset['operation_ph'] = dataset[['A22', 'A23']].apply(lambda item: item[0] - item[1])

        dataset['helper_sum'] = dataset['A1'] + dataset['A2'] + dataset['A2'] + dataset['A2']
        # B14/(A1+A2+A3+A4)
        dataset['B14_helper_sum_rate'] = dataset['B14'] / dataset['helper_sum']

        # A1 A2 A3 A4占比
        dataset['A1_helper_sum_rate'] = dataset['A1'] / dataset['helper_sum']
        dataset['A2_helper_sum_rate'] = dataset['A2'] / dataset['helper_sum']
        dataset['A3_helper_sum_rate'] = dataset['A3'] / dataset['helper_sum']
        dataset['A4_helper_sum_rate'] = dataset['A4'] / dataset['helper_sum']

        for index, column1 in enumerate(material_columns):
            for column2 in material_columns[index + 1:]:
                column_name = f'{column1}_division_{column2}'
                dataset[column_name] = dataset[[column1, column2]].apply(self._my_division)
        return dataset

    @staticmethod
    def _get_remove_columns(dataset):
        dataset = dataset.copy()

        remove_columns = list()
        length = dataset.shape[0]

        for column in dataset.columns:
            nunique = dataset[column].nunique()
            if nunique <= 1:
                remove_columns.append(column)
                continue

            null_rate = dataset[column].isnull().sum() / length
            if null_rate >= 0.9:
                remove_columns.append(column)
                continue

            biggest_rate = dataset[column].value_counts(normalize=True, dropna=False).values[0]
            if biggest_rate >= 0.95:
                remove_columns.append(column)

        return remove_columns

    @staticmethod
    def _time_to_num(time_str):
        if not time_str:
            return 0.5

        time_pattern = re.compile('([0-9]{1,2}):([0-9]{1,2})')
        match = time_pattern.search(time_str)

        if match:
            hours = match.group(1)
            minute = match.group(2)

            time_num = int(hours) + int(minute) / 60.0
            return time_num
        else:
            raise ValueError()

    def _get_time_duration(self, time_item, case=0):
        ts_pattern = re.compile('([0-9]{1,2}:[0-9]{1,2})-([0-9]{1,2}:[0-9]{1,2})')

        if case != 1:
            time1, time2 = time_item[0], time_item[1]

            if not time1 or not time2:
                return ''
        else:
            time1 = time_item
            time2 = None

        if case == 0:
            # time - time
            time1, time2 = time_item[0], time_item[1]

            if not time1 or not time2:
                return ''
        elif case == 1:
            # time_ts
            match = ts_pattern.search(time_item)
            if match:
                time1 = match.group(1)
                time2 = match.group(2)
            else:
                return ''
        elif case == 2:
            # time_ts - time
            if not time2:
                return ''
            match = ts_pattern.search(time1)
            if match:
                time1 = match.group(1)
            else:
                return ''
        elif case == 3:
            # time - time_ts
            if not time2:
                return ''
            match = ts_pattern.search(time2)
            if match:
                time2 = match.group(2)
            else:
                return ''
        elif case == 4:
            # time_ts - time_ts
            if not time2:
                return ''

            match1 = ts_pattern.search(time1)
            if match1:
                time1 = match1.group(1)
            else:
                return ''

            match2 = ts_pattern.search(time2)
            if match2:
                time2 = match2.group(2)
            else:
                return ''

        time1_num = self._time_to_num(time1)
        time2_num = self._time_to_num(time2)

        time_duration = time2_num - time1_num
        if time_duration < 0.0:
            time_duration += 24.0

        return time_duration

    @staticmethod
    def cmp(a, b):
        pattern = re.compile('([A|B])([0-9]+)')

        a_match = pattern.search(a)
        a_1 = a_match.group(1)
        a_2 = int(a_match.group(2))

        b_match = pattern.search(b)
        b_1 = b_match.group(1)
        b_2 = int(b_match.group(2))

        if a_1 < b_1:
            return -1

        if a_1 > b_1:
            return 1

        if a_2 < b_2:
            return -1

        if a_2 > b_2:
            return 1
        return 0

    def _sorted_columns(self):
        sorted_columns = time_columns.copy()
        sorted_columns.extend(time_ts_columns)

        sorted_columns = sorted(sorted_columns, key=cmp_to_key(self.cmp))
        return sorted_columns

    @staticmethod
    def _get_case_num(column1, column2):
        if column1 in time_columns:
            if column2 in time_columns:
                case_num = 0
            else:
                case_num = 3
        else:
            if column2 in time_columns:
                case_num = 2
            else:
                case_num = 4
        return case_num

    @staticmethod
    def _get_mapping_mean(item, mapping_dict):
        default_num = mapping_dict['avg_helper']

        return mapping_dict.get(item, default_num)

    @staticmethod
    def _get_sample_id(item):
        sample_id = item.split('_')[1]
        try:
            sample_id = int(sample_id)
        except ValueError:
            raise ValueError(f'error item:{item}')
        return sample_id

    def _data_encoder(self, dataset):
        dataset = dataset.copy()

        dataset['id'] = dataset['sample_id'].apply(self._get_sample_id)
        remove_columns = ['sample_id', 'rate', 'id']
        cate_columns = [column for column in dataset.columns if column not in remove_columns]

        label_encoder = LabelEncoder()
        for column in cate_columns:
            dataset[column] = label_encoder.fit_transform(dataset[column].astype(np.str_))

        train_data = dataset[dataset['rate'] > 0.0]

        train_data['helper'] = pd.cut(train_data['rate'], 5, labels=False)
        train_data = pd.get_dummies(train_data, columns=['helper'])
        helper_columns = [column for column in train_data.columns if 'helper' in column]

        # rate embedding + B14 embedding
        for column in cate_columns:
            biggest_rate = train_data[column].value_counts(normalize=True, dropna=False).values[0]
            if biggest_rate > 0.9:
                continue

            for helper_column in helper_columns:
                helper_column_name = f'{column}_{helper_column}_mean'
                b14_column_name = f'B14_{column}_{helper_column}_mean'
                column_df = train_data.groupby(by=[column])[helper_column].agg('mean').reset_index(name='mean')
                column_dict = column_df.set_index(column)['mean'].to_dict()
                column_dict['avg_helper'] = np.nanmean(train_data[helper_column])

                dataset[helper_column_name] = dataset[column].apply(self._get_mapping_mean, args=(column_dict,))
                dataset[b14_column_name] = dataset['B14'].apply(self._get_mapping_mean, args=(column_dict,))

        # One-Hot encoding
        dataset = pd.get_dummies(dataset, columns=cate_columns)

        return dataset

    def get_processing(self):
        dataset = self._get_data()

        # 业务特征
        dataset = self._get_operation_df(dataset)

        # remove_columns = self._get_remove_columns(dataset)
        # dataset = dataset.drop(columns=remove_columns)

        dataset = self._clear_dataset(dataset)

        sorted_columns = self._sorted_columns()
        length = len(sorted_columns)
        dataset_columns = dataset.columns

        for index, column in enumerate(sorted_columns):
            if column not in dataset_columns:
                continue

            if column in time_ts_columns:
                # time_ts column
                column_name = f'{column}_duration'
                dataset[column_name] = dataset[column].apply(self._get_time_duration, args=(1,))

            if length - index >= 2:
                next_1_column = sorted_columns[index + 1]

                if next_1_column not in dataset_columns:
                    continue

                column_name = f'{column}_{next_1_column}_duration'

                case_num = self._get_case_num(column, next_1_column)
                dataset[column_name] = dataset[[column, next_1_column]].apply(self._get_time_duration,
                                                                              args=(case_num,), axis=1)

            if length - index >= 3:
                next_2_column = sorted_columns[index + 2]

                if next_2_column not in dataset_columns:
                    continue

                column_name = f'{column}_{next_2_column}_duration'

                case_num = self._get_case_num(column, next_2_column)
                dataset[column_name] = dataset[[column, next_2_column]].apply(self._get_time_duration,
                                                                              args=(case_num,), axis=1)

        for column in time_columns:
            if column not in dataset_columns:
                continue

            time_num_clumn_name = f'{column}_time_num'
            dataset[time_num_clumn_name] = dataset[column].apply(self._time_to_num)

        dataset = dataset.drop(columns=sorted_columns, errors='ignore')
        dataset = self._data_encoder(dataset)

        return dataset


def processing_main():
    processing = FuckProcessing()
    dt = processing.get_processing()

    features_name = os.path.join(ETLDATA_PATH, 'features.csv')
    dt.to_csv(features_name, index=False, encoding='utf-8')


if __name__ == '__main__':
    processing_main()
