# preprocess data
import numpy as np
import re
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


def get_most_common_features(target, all_features, max = 3, min = 3):
    res = []
    main_keys = target.split('_')

    for feature in all_features:
        if target == feature:
            continue

        f_keys = feature.split('_')
        common_key_num = len(list(set(f_keys) & set(main_keys)))

        if common_key_num >= min and common_key_num <= max:
            res.append(feature)

    return res

def build_net(target, all_features):
    # get edge_indexes, and index_feature_map
    main_keys = target.split('_')
    edge_indexes = [
        [],
        []
    ]
    index_feature_map = [target]

    # find closest features(nodes):
    parent_list = [target]
    graph_map = {}
    depth = 2
    
    for i in range(depth):        
        for feature in parent_list:
            children = get_most_common_features(feature, all_features)

            if feature not in graph_map:
                graph_map[feature] = []
            
            # exclude parent
            pure_children = []
            for child in children:
                if child not in graph_map:
                    pure_children.append(child)

            graph_map[feature] = pure_children

            if feature not in index_feature_map:
                index_feature_map.append(feature)
            p_index = index_feature_map.index(feature)
            for child in pure_children:
                if child not in index_feature_map:
                    index_feature_map.append(child)
                c_index = index_feature_map.index(child)

                edge_indexes[1].append(p_index)
                edge_indexes[0].append(c_index)

        parent_list = pure_children

    return edge_indexes, index_feature_map


def construct_data(data, feature_map, labels=0):
    res = []

    for feature in feature_map:
        if feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())
        else:
            print(feature, 'not exist in data')
    # append labels as last
    sample_n = len(res[0])

    if type(labels) == int:
        res.append([labels]*sample_n)
    elif len(labels) == sample_n:
        res.append(labels)

    return res

def build_loc_net(struc, all_features, feature_map=[]):

    index_feature_map = feature_map
    edge_indexes = [
        [],
        []
    ]
    for node_name, node_list in struc.items():
        if node_name not in all_features:
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)
        
        p_index = index_feature_map.index(node_name)
        for child in node_list:
            if child not in all_features:
                continue

            if child not in index_feature_map:
                print(f'error: {child} not in index_feature_map')
                # index_feature_map.append(child)

            c_index = index_feature_map.index(child)
            # edge_indexes[0].append(p_index)
            # edge_indexes[1].append(c_index)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)
        

    
    return edge_indexes


def preprocess_data(train, test, label_column='attack', slide_win=15, slide_stride=5):
    """
    处理训练集和测试集数据,转化为Transformer模型需要的输入格式。
    该函数将处理训练集和测试集，去掉标签列，并进行数据标准化和滑动窗口处理。

    Parameters:
    - train: 训练集,pandas DataFrame
    - test: 测试集,pandas DataFrame
    - label_column: 标签列的名称，默认'attack'
    - slide_win: 滑动窗口大小,默认15
    - slide_stride: 滑动窗口步长,默认5

    Returns:
    - train_data: 处理后的训练集数据
    - test_data: 处理后的测试集数据
    """
    
    # 如果label_column是列表，只取第一个元素
    if isinstance(label_column, list):
        label_column = label_column[0]

    # 去掉标签列，如果存在
    if label_column in train.columns:
        train_data = train.drop(columns=[label_column])  # 去掉标签列
    else:
        train_data = train  # 如果没有标签列，则直接使用所有数据作为输入

    if label_column in test.columns:
        test_data = test.drop(columns=[label_column])  # 去掉标签列
    else:
        test_data = test  # 如果没有标签列，则直接使用所有数据作为输入

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 标准化训练集
    train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
    
    # 标准化测试集（使用训练集的归一化器）
    test_data = pd.DataFrame(scaler.transform(test_data), columns=train_data.columns)  # 关键修改

    # 滑动窗口处理
    def create_subsequences(data, window_size=slide_win, stride=slide_stride):
        subsequences = []

        for start in range(0, len(data) - window_size + 1, stride):
            end = start + window_size
            subsequences.append(data[start:end].values)

        subsequences = torch.tensor(subsequences, dtype=torch.float32)
        return subsequences

    # 训练数据和测试数据的滑动窗口处理
    train_data = create_subsequences(train_data, slide_win, slide_stride)
    test_data = create_subsequences(test_data, slide_win, slide_stride)

    return train_data, test_data


