#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
# import numpy as np
import re
import datetime


def analysis(include_var, data, target_var, bad_val=1):
    """
    双变量分析
    :param include_var: 参与分析的字段
    :param data: 分箱后数据
    :param target_var:标签字段
    :param bad_val:标签值
    :return:双变量分析表
    """
    bivar_ana = pd.DataFrame()

    def fun_bi_var_ana(var):

        cnt = data[var].value_counts().reset_index().rename(columns={'index': 'VAR_VALUE', var: 'CNT'})
        bad_cnt = data[var][data[target_var] == bad_val].value_counts().reset_index().\
            rename(columns={'index': 'VAR_VALUE', var: 'BAD_CNT'})
        bi_var_tab = pd.merge(cnt, bad_cnt, how='left', on='VAR_VALUE')
        bi_var_tab['BAD_RATE'] = bi_var_tab['BAD_CNT'] / bi_var_tab['CNT']
        bi_var_tab['CNT_%'] = bi_var_tab['CNT'] / len(data)
        bi_var_tab['VARIATE'] = var
        return bi_var_tab

    for i in include_var:
        bi_var = fun_bi_var_ana(var=i)
        bi_var.fillna(0, inplace=True)
        bivar_ana = bivar_ana.append(bi_var)

    def bin_min(bins):
        """
        提取每个分箱的最小值
        :param bins: 分箱
        :return: 分箱的最小值
        """
        str_bin = str(bins)
        split_bin = re.sub(r',.*$', '', str_bin).strip('(')
        try:
            split_bin = float(split_bin)
            return split_bin
        except ValueError:
            return split_bin
    # 对双变量分析表按VAR_VALUE进行排序
    bivar_ana['sort'] = bivar_ana['VAR_VALUE'].apply(bin_min)
    bivar_ana = bivar_ana.sort_values(by=['VARIATE', 'sort']).reset_index(drop=True)
    bivar_ana = bivar_ana[['VARIATE', 'VAR_VALUE', 'CNT', 'CNT_%', 'BAD_RATE']]

    return bivar_ana


def bin_merge_fir(data, bi_ana_tab, exclude_var):
    """
    初步合箱函数（针对连续型变量），对第一个箱和最后一个箱BAD_RATE为0或1的进行合箱处理
    :param data:合箱数据
    :param bi_ana_tab:双变量分析表（根据‘VAR_VALUE’字段排序后）
    :param exclude_var:不需合箱的字段（包含非连续型字段）
    :return 分箱合并后的数据；不需要进行合箱的字段列表
    """
    include_var = [c for c in data.columns if c not in exclude_var]
    merge_list, drop_list = [], []
    bi_ana_tab['NEW_VAR_VALUE'] = bi_ana_tab['VAR_VALUE']

    for d in include_var:
        if bi_ana_tab.loc[bi_ana_tab['VARIATE'] == d].shape[0] == 1:
            print(d)
            drop_list.append(d)
            continue

        idx = bi_ana_tab[(bi_ana_tab['VARIATE'] == d) & (bi_ana_tab['VAR_VALUE'] != 'NAN')
                         & (bi_ana_tab['BAD_RATE'].isin([0, 1]))].index

        if any(idx) or idx == 0:
            if bi_ana_tab.loc[bi_ana_tab['VARIATE'] == d, 'BAD_RATE'].value_counts().shape[0] <= 2:
                print(d)
                drop_list.append(d)
                continue
            bins_index = bi_ana_tab.loc[(bi_ana_tab['VARIATE'] == d) & (bi_ana_tab['VAR_VALUE'] != 'NAN'),
                                        'BAD_RATE'].index
            index = bins_index[0]
            bad_rate = bi_ana_tab.loc[index, 'BAD_RATE']
            while (bad_rate == 0 or bad_rate == 1) and (index < bins_index[-1]):
                index += 1
                bad_rate = bi_ana_tab.loc[index, 'BAD_RATE']
            left = str(bi_ana_tab.loc[bins_index[0], 'VAR_VALUE']).split(',')[0]
            right = str(bi_ana_tab.loc[index, 'VAR_VALUE']).split(',')[1]
            bin_value = left + ',' + right
            bi_ana_tab.loc[bins_index[bins_index <= index], 'NEW_VAR_VALUE'] = bin_value

            index = bins_index[-1]
            bad_rate = bi_ana_tab.loc[index, 'BAD_RATE']
            while (bad_rate == 0 or bad_rate == 1) and (index > bins_index[0]):
                index -= 1
                bad_rate = bi_ana_tab.loc[index, 'BAD_RATE']
            left = str(bi_ana_tab.loc[index, 'VAR_VALUE']).split(',')[0]
            right = str(bi_ana_tab.loc[bins_index[-1], 'VAR_VALUE']).split(',')[1]
            bin_value = left + ',' + right
            bi_ana_tab.loc[bins_index[bins_index >= index], 'NEW_VAR_VALUE'] = bin_value
            merge_list.append(d)

    for m in merge_list:
        map_dict = dict(zip(bi_ana_tab['VAR_VALUE'][bi_ana_tab['VARIATE'] == m],
                            bi_ana_tab['NEW_VAR_VALUE'][bi_ana_tab['VARIATE'] == m]))
        data[m] = data[m].map(map_dict)

    return data, drop_list


def bin_merge_sec(data, bi_ana_tab, exclude_var, rate):
    """
    合箱函数（针对连续型变量），对样本量过小的箱进行合并
    :param data:合箱数据
    :param bi_ana_tab:双变量分析表（根据‘VAR_VALUE’字段排序后）
    :param exclude_var:不需要参与分析的字段（包含非连续型字段、drop_list字段）
    :param rate: float, 合箱阈值
    :return 分箱合并后的数据；
    """
    include_var = [c for c in data.columns if c not in exclude_var]
    merge_list, drop_list = [], []
    bi_ana_tab['NEW_VAR_VALUE'] = bi_ana_tab['VAR_VALUE']

    for d in include_var:

        index = bi_ana_tab.loc[(bi_ana_tab['VARIATE'] == d) & (bi_ana_tab['VAR_VALUE'] != 'NAN')
                               & (bi_ana_tab['CNT_%'] <= rate)].index

        if any(index) or index == 0:
            bins_index = bi_ana_tab.loc[(bi_ana_tab['VARIATE'] == d) & (bi_ana_tab['VAR_VALUE'] != 'NAN')].index

            if bi_ana_tab.loc[bi_ana_tab['VARIATE'] == d, 'CNT_%'].max() >= (1 - rate):
                print(d)
                drop_list.append(d)
                continue
            else:
                merge_list.append(d)
                index_l = list(index)
                for i, idx in enumerate(index_l):
                    idx_ = idx
                    while idx_ in index_l:
                        index_l.remove(idx_)
                        idx_ += 1
                    index_l.insert(i, 0)
                    if bi_ana_tab.loc[bins_index[(bins_index >= idx) & (bins_index <= idx_ - 1)],
                                      'CNT_%'].sum() >= rate:
                        left = str(bi_ana_tab.loc[idx, 'VAR_VALUE']).split(',')[0]
                        right = str(bi_ana_tab.loc[idx_ - 1, 'VAR_VALUE']).split(',')[1]
                        bin_value = left + ',' + right
                        bi_ana_tab.loc[bins_index[(bins_index >= idx) & (bins_index <= idx_ - 1)],
                                       'NEW_VAR_VALUE'] = bin_value
                    else:
                        if idx_ in bins_index:
                            left = str(bi_ana_tab.loc[idx, 'VAR_VALUE']).split(',')[0]
                            right = str(bi_ana_tab.loc[idx_, 'VAR_VALUE']).split(',')[1]
                            bin_value = left + ',' + right
                            bi_ana_tab.loc[bins_index[(bins_index >= idx) & (bins_index <= idx_)],
                                           'NEW_VAR_VALUE'] = bin_value
                        else:
                            value = bi_ana_tab.loc[idx - 1, 'NEW_VAR_VALUE']
                            if bi_ana_tab.loc[idx - 1, 'VAR_VALUE'] == value:
                                left = str(bi_ana_tab.loc[idx - 1, 'VAR_VALUE']).split(',')[0]
                                right = str(bi_ana_tab.loc[idx_ - 1, 'VAR_VALUE']).split(',')[1]
                                bin_value = left + ',' + right
                                bi_ana_tab.loc[bins_index[(bins_index >= idx - 1) & (bins_index <= idx_ - 1)],
                                               'NEW_VAR_VALUE'] = bin_value
                            else:
                                idx = bi_ana_tab[(bi_ana_tab['VARIATE'] == d) &
                                                 (bi_ana_tab['NEW_VAR_VALUE'] == value)].index[0]
                                left = str(bi_ana_tab.loc[idx, 'VAR_VALUE']).split(',')[0]
                                right = str(bi_ana_tab.loc[idx_ - 1, 'VAR_VALUE']).split(',')[1]
                                bin_value = left + ',' + right
                                bi_ana_tab.loc[bins_index[(bins_index >= idx) & (bins_index <= idx_ - 1)],
                                               'NEW_VAR_VALUE'] = bin_value
    for m in merge_list:
        map_dict = dict(zip(bi_ana_tab['VAR_VALUE'][bi_ana_tab['VARIATE'] == m],
                            bi_ana_tab['NEW_VAR_VALUE'][bi_ana_tab['VARIATE'] == m]))
        data[m] = data[m].map(map_dict)

    return data, drop_list


def bin_merge_thi(data, bi_ana_tab, exclude_var):
    """
    合箱函数（针对连续型变量）,解决单调性
    :param data:合箱数据
    :param bi_ana_tab:双变量分析表（根据‘VAR_VALUE’字段排序后）
    :param exclude_var:不需要参与h合箱的字段（包含非连续型字段、drop_list中字段、noneed_list中字段）
    :return:分箱合并后的数据；不需要进行合箱的字段
    """
    include_var = [c for c in data.columns if c not in exclude_var]
    merge_list, noneed_list = [], []
    bi_ana_tab['NEW_VAR_VALUE'] = bi_ana_tab['VAR_VALUE']

    for d in include_var:

        bins_index = bi_ana_tab.loc[(bi_ana_tab['VARIATE'] == d) & (bi_ana_tab['VAR_VALUE'] != 'NAN')].index

        if len(bins_index) <= 2:
            noneed_list.append(d)
            continue

        first = bi_ana_tab.loc[bins_index[0], 'BAD_RATE']
        last = bi_ana_tab.loc[bins_index[-1], 'BAD_RATE']

        if first >= last:
            order_index = {idx for idx in bins_index if first >= bi_ana_tab.loc[idx, 'BAD_RATE'] >= last}
        else:
            order_index = {idx for idx in bins_index if first <= bi_ana_tab.loc[idx, 'BAD_RATE'] <= last}

        out_index = set(bins_index) - order_index

        if out_index:
            out_index = sorted(list(out_index))
            for idx in out_index:
                if bi_ana_tab.loc[idx, 'BAD_RATE'] >= max(first, last):
                    idx_l, idx_r = idx - 1, idx
                    while idx_r in out_index:
                        out_index.remove(idx_r)
                        idx_r += 1
                    out_index.insert(0, idx)
                    # 递减趋势
                    if first >= last:
                        # 跟第一个箱合
                        if idx_l == bins_index[0]:
                            left = str(bi_ana_tab.loc[idx_l, 'VAR_VALUE']).split(',')[0]
                            right = str(bi_ana_tab.loc[idx_r - 1, 'VAR_VALUE']).split(',')[1]
                            bin_value = left + ',' + right
                            bi_ana_tab.loc[bins_index[bins_index < idx_r], 'NEW_VAR_VALUE'] = bin_value
                        # 跟BAD_RATE较小的合
                        else:
                            if bi_ana_tab.loc[idx_l, 'BAD_RATE'] > bi_ana_tab.loc[idx_r, 'BAD_RATE']:
                                left = str(bi_ana_tab.loc[idx_l + 1, 'VAR_VALUE']).split(',')[0]
                                right = str(bi_ana_tab.loc[idx_r, 'VAR_VALUE']).split(',')[1]
                                bin_value = left + ',' + right
                                bi_ana_tab.loc[bins_index[(bins_index <= idx_r) & (bins_index > idx_l)],
                                               'NEW_VAR_VALUE'] = bin_value
                            else:
                                left = str(bi_ana_tab.loc[idx_l, 'VAR_VALUE']).split(',')[0]
                                right = str(bi_ana_tab.loc[idx_r - 1, 'VAR_VALUE']).split(',')[1]
                                bin_value = left + ',' + right
                                bi_ana_tab.loc[bins_index[(bins_index < idx_r) & (bins_index >= idx_l)],
                                               'NEW_VAR_VALUE'] = bin_value
                    # 递增趋势
                    else:
                        # 跟最后一个箱合
                        if idx_r == bins_index[-1]:
                            left = str(bi_ana_tab.loc[idx_l + 1, 'VAR_VALUE']).split(',')[0]
                            right = str(bi_ana_tab.loc[idx_r, 'VAR_VALUE']).split(',')[1]
                            bin_value = left + ',' + right
                            bi_ana_tab.loc[bins_index[bins_index > idx_l], 'NEW_VAR_VALUE'] = bin_value
                        # 跟BAD_RATE较小的合
                        else:
                            if bi_ana_tab.loc[idx_l, 'BAD_RATE'] > bi_ana_tab.loc[idx_r, 'BAD_RATE']:
                                left = str(bi_ana_tab.loc[idx_l + 1, 'VAR_VALUE']).split(',')[0]
                                right = str(bi_ana_tab.loc[idx_r, 'VAR_VALUE']).split(',')[1]
                                bin_value = left + ',' + right
                                bi_ana_tab.loc[bins_index[(bins_index <= idx_r) & (bins_index > idx_l)],
                                               'NEW_VAR_VALUE'] = bin_value
                            else:
                                left = str(bi_ana_tab.loc[idx_l, 'VAR_VALUE']).split(',')[0]
                                right = str(bi_ana_tab.loc[idx_r - 1, 'VAR_VALUE']).split(',')[1]
                                bin_value = left + ',' + right
                                bi_ana_tab.loc[bins_index[(bins_index < idx_r) & (bins_index >= idx_l)],
                                               'NEW_VAR_VALUE'] = bin_value
                elif bi_ana_tab.loc[idx, 'BAD_RATE'] <= min(first, last):
                    idx_l, idx_r = idx - 1, idx
                    while idx_r in out_index:
                        out_index.remove(idx_r)
                        idx_r += 1
                    out_index.insert(0, idx)
                    # 递减趋势
                    if first >= last:
                        # 跟最后一个箱合并
                        if idx_r == bins_index[-1]:
                            left = str(bi_ana_tab.loc[idx_l+1, 'VAR_VALUE']).split(',')[0]
                            right = str(bi_ana_tab.loc[idx_r, 'VAR_VALUE']).split(',')[1]
                            bin_value = left + ',' + right
                            bi_ana_tab.loc[bins_index[bins_index > idx_l], 'NEW_VAR_VALUE'] = bin_value
                        # 跟BAD_RATE较大的合
                        else:
                            if bi_ana_tab.loc[idx_l, 'BAD_RATE'] < bi_ana_tab.loc[idx_r, 'BAD_RATE']:
                                left = str(bi_ana_tab.loc[idx_l + 1, 'VAR_VALUE']).split(',')[0]
                                right = str(bi_ana_tab.loc[idx_r, 'VAR_VALUE']).split(',')[1]
                                bin_value = left + ',' + right
                                bi_ana_tab.loc[bins_index[(bins_index <= idx_r) & (bins_index > idx_l)],
                                               'NEW_VAR_VALUE'] = bin_value
                            else:
                                left = str(bi_ana_tab.loc[idx_l, 'VAR_VALUE']).split(',')[0]
                                right = str(bi_ana_tab.loc[idx_r - 1, 'VAR_VALUE']).split(',')[1]
                                bin_value = left + ',' + right
                                bi_ana_tab.loc[bins_index[(bins_index < idx_r) & (bins_index >= idx_l)],
                                               'NEW_VAR_VALUE'] = bin_value
                    # 递增趋势
                    else:
                        # 跟第一个箱合并
                        if idx_l == bins_index[0]:
                            left = str(bi_ana_tab.loc[idx_l, 'VAR_VALUE']).split(',')[0]
                            right = str(bi_ana_tab.loc[idx_r - 1, 'VAR_VALUE']).split(',')[1]
                            bin_value = left + ',' + right
                            bi_ana_tab.loc[bins_index[bins_index < idx_r], 'NEW_VAR_VALUE'] = bin_value
                        # 跟BAD_RATE较大的合并
                        else:
                            if bi_ana_tab.loc[idx_l, 'BAD_RATE'] < bi_ana_tab.loc[idx_r, 'BAD_RATE']:
                                left = str(bi_ana_tab.loc[idx_l + 1, 'VAR_VALUE']).split(',')[0]
                                right = str(bi_ana_tab.loc[idx_r, 'VAR_VALUE']).split(',')[1]
                                bin_value = left + ',' + right
                                bi_ana_tab.loc[bins_index[(bins_index <= idx_r) & (bins_index > idx_l)],
                                               'NEW_VAR_VALUE'] = bin_value
                            else:
                                left = str(bi_ana_tab.loc[idx_l, 'VAR_VALUE']).split(',')[0]
                                right = str(bi_ana_tab.loc[idx_r - 1, 'VAR_VALUE']).split(',')[1]
                                bin_value = left + ',' + right
                                bi_ana_tab.loc[bins_index[(bins_index < idx_r) & (bins_index >= idx_l)],
                                               'NEW_VAR_VALUE'] = bin_value
            merge_list.append(d)
        else:
            noneed_list.append(d)
            continue

    for m in merge_list:
        map_dict = dict(zip(bi_ana_tab['VAR_VALUE'][bi_ana_tab['VARIATE'] == m],
                            bi_ana_tab['NEW_VAR_VALUE'][bi_ana_tab['VARIATE'] == m]))
        data[m] = data[m].map(map_dict)

    return data, noneed_list


def bin_merge_fin(data, bi_ana_tab, exclude_var):
    """
    合箱函数（针对连续型变量），对空值进行处理
    :param data:合箱数据
    :param bi_ana_tab:双变量分析表（根据‘VAR_VALUE’字段排序后）
    :param exclude_var:不需要参与分析的字段（包含非连续型字段、drop_list字段）
    :return 分箱合并后的数据；
    """
    include_var = [c for c in data.columns if c not in exclude_var]
    merge_list = []
    bi_ana_tab['NEW_VAR_VALUE'] = bi_ana_tab['VAR_VALUE']

    for d in include_var:

        index = bi_ana_tab.loc[(bi_ana_tab['VARIATE'] == d) & (bi_ana_tab['VAR_VALUE'] == 'NAN')].index

        if any(index) or index == 0:
            bins_index = bi_ana_tab.loc[bi_ana_tab['VARIATE'] == d].index

            if len(bins_index) <= 2:
                continue
            else:
                value_list = bi_ana_tab.loc[bins_index[:-1], 'BAD_RATE'].values
                first = bi_ana_tab.loc[bins_index[0], 'BAD_RATE']
                last = bi_ana_tab.loc[bins_index[-2], 'BAD_RATE']

                if bi_ana_tab.loc[bins_index[-1], 'BAD_RATE'] >= max(value_list):
                    if first >= last:
                        bi_ana_tab.loc[bins_index[-1], 'NEW_VAR_VALUE'] = bi_ana_tab.loc[bins_index[0],
                                                                                         'VAR_VALUE']
                    else:
                        bi_ana_tab.loc[bins_index[-1], 'NEW_VAR_VALUE'] = bi_ana_tab.loc[bins_index[-2], 'VAR_VALUE']
                elif bi_ana_tab.loc[bins_index[-1], 'BAD_RATE'] <= min(value_list):
                    if first >= last:
                        bi_ana_tab.loc[bins_index[-1], 'NEW_VAR_VALUE'] = bi_ana_tab.loc[bins_index[-2], 'VAR_VALUE']
                    else:
                        bi_ana_tab.loc[bins_index[-1], 'NEW_VAR_VALUE'] = bi_ana_tab.loc[bins_index[0], 'VAR_VALUE']
                else:
                    if len(bins_index) >= 4:
                        diff_dict = {}
                        for idx in bins_index[1:-2]:
                            diff = abs(bi_ana_tab.loc[idx, 'BAD_RATE'] - bi_ana_tab.loc[bins_index[-1], 'BAD_RATE'])
                            diff_dict.update({idx: diff})
                        idx = max(diff_dict, key=diff_dict.get)
                    else:
                        idx = bins_index[-2]
                    bi_ana_tab.loc[bins_index[-1], 'NEW_VAR_VALUE'] = bi_ana_tab.loc[idx, 'VAR_VALUE']
            merge_list.append(d)
        else:
            continue

    for m in merge_list:
        map_dict = dict(zip(bi_ana_tab['VAR_VALUE'][bi_ana_tab['VARIATE'] == m],
                            bi_ana_tab['NEW_VAR_VALUE'][bi_ana_tab['VARIATE'] == m]))
        data[m] = data[m].map(map_dict)

    return data


def bin_merge_dif(data, bi_ana_tab, exclude_var):
    """
    合箱函数（针对离散型变量），bad_rate为0 、1的分箱与表现接近的分箱分并
    :param data:分箱后数据
    :param bi_ana_tab:双变量分析表
    :param exclude_var:不需要参与分析的字段（包含连续型变量）
    :return:分箱合并后的数据
    """
    include_var = [c for c in data.columns if c not in exclude_var]
    discrete_merge_list = []

    bi_ana_tab = bi_ana_tab.sort_values(by=['VARIATE', 'BAD_RATE'], ascending=True)
    bi_ana_tab = bi_ana_tab.reset_index(drop=True)
    bi_ana_tab['NEW_VAR_VALUE'] = bi_ana_tab['VAR_VALUE']
    for d in include_var:
        line1_index = bi_ana_tab[bi_ana_tab['VARIATE'] == d].index[0]
        last_index = bi_ana_tab[bi_ana_tab['VARIATE'] == d].index[-1]
        if bi_ana_tab.loc[line1_index, 'BAD_RATE'] == 0:
            discrete_merge_list.append(d)
            num = (bi_ana_tab[bi_ana_tab['VARIATE'] == d]['BAD_RATE'].values == 0).sum()
            index = bi_ana_tab[bi_ana_tab['VARIATE'] == d].index[:num+1]
            bi_ana_tab.loc[index, 'NEW_VAR_VALUE'] = '_'.join(bi_ana_tab.loc[index,
                                                                             'VAR_VALUE'].apply(lambda x: str(x)))
        if bi_ana_tab.loc[last_index, 'BAD_RATE'] == 1:
            discrete_merge_list.append(d)
            num = (bi_ana_tab[bi_ana_tab['VARIATE'] == d]['BAD_RATE'].values == 1).sum()
            index = bi_ana_tab[bi_ana_tab['VARIATE'] == d].index[-num-1:]
            bi_ana_tab.ix[index, 'NEW_VAR_VALUE'] = '_'.join(bi_ana_tab.ix[index,
                                                                           'VAR_VALUE'].apply(lambda x: str(x)))
        normal_num = bi_ana_tab[(bi_ana_tab['VARIATE'] == d) & (bi_ana_tab['BAD_RATE'] > 0)
                                & (bi_ana_tab['BAD_RATE'] < 1)].index
        if len(normal_num) < 2:
            index = bi_ana_tab[bi_ana_tab['VARIATE'] == d].index
            bi_ana_tab.ix[index, 'NEW_VAR_VALUE'] = '_'.join(bi_ana_tab.loc[index, 'VAR_VALUE'].
                                                             apply(lambda x: str(x)))
    discrete_merge_list = list(set(discrete_merge_list))

    # 更新原数据的分箱
    for m in discrete_merge_list:
        map_dict = dict(zip(bi_ana_tab['VAR_VALUE'][bi_ana_tab['VARIATE'] == m],
                            bi_ana_tab['NEW_VAR_VALUE'][bi_ana_tab['VARIATE'] == m]))
        data[m] = data[m].map(map_dict)
    return data


def bin_merge_dis(data, bi_ana_tab, exclude_var, rate):
    """
    合箱函数（针对离散型变量），样本数小于5%的合箱
    :param data:分箱后数据
    :param bi_ana_tab:双变量分析表
    :param exclude_var:不需要参与分析的字段（包含连续型变量）
    :param rate:float，[0,1], 合箱阈值
    :return:分箱合并后的数据
    """
    include_var = [c for c in data.columns if c not in exclude_var]
    discrete_merge_list = []

    bi_ana_tab = bi_ana_tab.sort_values(by=['VARIATE', 'CNT'])
    bi_ana_tab = bi_ana_tab.reset_index(drop=True)
    bi_ana_tab['NEW_VAR_VALUE'] = bi_ana_tab['VAR_VALUE']
    for var in include_var:
        list_cnt = bi_ana_tab[bi_ana_tab['VARIATE'] == var]['CNT'].values
        num = 1
        n = data.shape[0] * rate
        while (len(list_cnt) > 2) & (list_cnt.min() < n):
            discrete_merge_list.append(var)
            idx = bi_ana_tab[(bi_ana_tab['VARIATE'] == var) & (bi_ana_tab['CNT'] == list_cnt.min())].index[0]
            bi_ana_tab.loc[idx + 1, 'CNT'] = bi_ana_tab.loc[idx + 1, 'CNT'] + bi_ana_tab.loc[idx, 'CNT']
            idx2 = bi_ana_tab[bi_ana_tab['VARIATE'] == var].index[:num+1]
            bi_ana_tab.loc[idx2, 'NEW_VAR_VALUE'] = '_'.join(bi_ana_tab.loc[idx2,
                                                                            'VAR_VALUE'].apply(lambda x: str(x)))
            list_cnt = bi_ana_tab[bi_ana_tab['VARIATE'] == var]['CNT'].values
            list_cnt = list_cnt[num:]
            num += 1
    discrete_merge_list = list(set(discrete_merge_list))

    # 更新原数据的分箱
    for m in discrete_merge_list:
        map_dict = dict(zip(bi_ana_tab['VAR_VALUE'][bi_ana_tab['VARIATE'] == m],
                            bi_ana_tab['NEW_VAR_VALUE'][bi_ana_tab['VARIATE'] == m]))
        data[m] = data[m].map(map_dict)
    return data


if __name__ == '__main__':

    # 开始时间
    start_time = datetime.datetime.now()
    print('开始时间为：%s\n' % start_time)

    # 读取原始数据
    path = r'D:\myproject\设备模型'
    f = open(path + '\\raw_zy_bin.csv', encoding='utf-8')
    raw_data_bin = pd.read_csv(f)

    # 数据分箱
    # raw_data_bin = bins_cut(raw_data, ['tgt'], [])
    # print('分箱完成')
    # 分箱后数据
    # raw_data_bin = pd.read_csv(path + '\\Demo_bin.csv')

    # 基本设置
    exclude_var_f = ['tgt', 'keyid', 'Score']
    include_var_ = [var for var in raw_data_bin.columns if var not in exclude_var_f]
    # discrete_var = [var for var in raw_data.select_dtypes(include='object').columns if var not in exclude_var_f]
    # continuous_var = [var for var in raw_data.select_dtypes(exclude='object').columns if var not in exclude_var_f]
    continuous_var = [var for var in raw_data_bin.select_dtypes(include='object').columns if var not in exclude_var_f]
    discrete_var = []
    target_var_ = 'tgt'

    # 合箱之前的双变量分析
    raw_bi_ana = analysis(include_var_, raw_data_bin, target_var=target_var_, bad_val=1)
    # raw_bi_ana.to_csv('bi_ana_00.csv', index=False)
    print('<' * 50 + '原始分箱双变量分析完成' + '>' * 50)

    # 连续型变量
    # 初步合箱、分析
    exclude_var_ = discrete_var + exclude_var_f
    new_data, drop_list_f = bin_merge_fir(raw_data_bin, raw_bi_ana, exclude_var_)
    exclude_var_ += drop_list_f
    include_var_ = [var for var in new_data.columns if var not in exclude_var_]
    print('连续型变量还有{}个'.format(len(include_var_)))
    new_bi_ana = analysis(include_var_, new_data, target_var=target_var_, bad_val=1)
    print('初步合箱完成\n')
    # new_bi_ana.to_csv('bi_ana_01.csv', index=False)

    # 第二步合箱
    new_data, drop_list_s = bin_merge_sec(new_data, new_bi_ana, exclude_var_, 0.02)
    exclude_var_ += drop_list_s
    include_var_ = [var for var in new_data.columns if var not in exclude_var_]
    print('连续型变量还有{}个'.format(len(include_var_)))
    new_bi_ana = analysis(include_var_, new_data, target_var=target_var_, bad_val=1)
    print('第二步合箱完成\n')
    # new_bi_ana.to_csv('bi_ana_02.csv', index=False)

    # 再次合箱、分析
    while include_var_:
        # 合箱
        new_data, noneed_list_ = bin_merge_thi(new_data, new_bi_ana, exclude_var_)
        exclude_var_ += noneed_list_
        include_var_ = [var for var in new_data.columns if var not in exclude_var_]
        # 双变量分析
        if include_var_:
            new_bi_ana = analysis(include_var_, new_data, target_var=target_var_, bad_val=1)
    drop_list_ = drop_list_f + drop_list_s
    exclude_var_ = drop_list_ + exclude_var_f + discrete_var
    include_var_ = [var for var in raw_data_bin.columns if var not in exclude_var_f]
    bi_ana_03 = analysis(include_var_, new_data, target_var=target_var_, bad_val=1)
    # bi_ana_03.to_excel('bi_ana_03.xlsx', index=False)
    print('第三步合箱完成\n')

    # 连续型最终合箱、分析
    fina_data = bin_merge_fin(new_data, bi_ana_03, exclude_var_)
    for var in drop_list_:
        del new_data[var]
    print('<' * 50 + '连续型变量合箱完成' + '>' * 50)

    # 离散型合箱
    exclude_var_ = continuous_var + exclude_var_f
    include_var_ = [var for var in new_data.columns if var not in exclude_var_]
    if include_var_:
        new_data = bin_merge_dif(fina_data, raw_bi_ana, exclude_var_)
        bi_ana_04 = analysis(include_var_, new_data, target_var=target_var_, bad_val=1)
        fina_data = bin_merge_dis(new_data, bi_ana_04, exclude_var_, 0.05)
        # bi_ana_03.to_csv('bi_ana_03.csv', index=False)
        # bi_ana_04.to_csv('bi_ana_04.csv', index=False)
        print('<' * 50 + '离散型变量合箱完成' + '>' * 50)

    # 最终双变量分析表
    include_var_ = [var for var in fina_data.columns if var not in exclude_var_f]
    final_bivariate_analysis = analysis(include_var_, fina_data, target_var=target_var_, bad_val=1)

    # 输出数据
    fina_data.to_csv('new_data_bin.csv', index=False)
    print('<' * 50 + '输出最终合箱数据' + '>' * 50)
    # 连续型变量缺失值处理表
    missing_value = bi_ana_03[(bi_ana_03['VAR_VALUE'] == 'NAN') &
                              (bi_ana_03['NEW_VAR_VALUE'] != 'NAN')][['VARIATE', 'VAR_VALUE', 'NEW_VAR_VALUE']]
    missing_value.to_excel('missing_value_handling.xlsx', index=False)
    # final_bivariate_analysis.to_csv('final_bivariate_analysis.csv', index=False)
    final_bivariate_analysis.to_excel('final_bivariate_analysis.xlsx', index=False)
    print('<' * 50 + '输出最终双变量分析表' + '>' * 50)

    # 结束时间及耗时
    end_time = datetime.datetime.now()
    take_time = end_time - start_time
    print('结束时间为：%s\n' % end_time)
    print('耗时：%s\n' % take_time)
