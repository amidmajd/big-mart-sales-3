import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np

####### fill missing values ######
def item_fat(data):
    data = data.replace({'Item_Fat_Content': {'reg': 'Regular', 'low fat': 'Low Fat', 'LF': 'Low Fat'}})
    return data


def outlet_size(data):
    for inx, item in data.iterrows():
        if pd.isnull(item['Outlet_Size']):
            if item['Outlet_Type'] == 'Grocery Store':
                data.loc[inx, 'Outlet_Size'] = 'Small'
            elif item['Outlet_Type'] == 'Supermarket Type1':
                # siml = data[data.loc[:, 'Outlet_Location_Type'] == 'Tier 2'].iloc[:, 8]  =>>> 'Small'
                data.loc[inx, 'Outlet_Size'] = 'Small'
    return data


def item_weight(data):
    for index, item in data.iterrows():
        if pd.isnull(item[1]):
            for inx, i in data[data.loc[:, 'Item_Identifier'] == item[0]].iterrows():
                if not pd.isnull(i[1]):
                    data.iloc[index, 1] = i[1]
                    break
    w_data = data.iloc[:, 1]
    w_data.fillna(method='ffill', inplace=True)
    data['Item_Weight'] = w_data
    return data


# def item_weight(data):
#     for index, item in data.iterrows():
#         if pd.isnull(item[1]):
#             for inx, i in data[data.loc[:, 'Item_Identifier'] == item[0]].iterrows():
#                 if not pd.isnull(i[1]):
#                     data.iloc[index, 1] = i[1]
#                     break
#
#     if len(data[pd.isnull(data.loc[:, 'Item_Weight'])]) != 0:
#         w_set = data.iloc[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]]  # omitting visibility FT
#         w_train_data = w_set.dropna().iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9]]
#         w_train_target = w_set.dropna().iloc[:, 1]  # weight
#         w_predict_data = w_set[pd.isnull(w_set.iloc[:, 1])].iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9]]
#         w_train_data_encoded = encode(w_train_data)[0]
#         w_predict_data_encoded = encode(w_predict_data)[0]
#
#         w_tree = tree.DecisionTreeRegressor(max_depth=9).fit(w_train_data_encoded, w_train_target)
#         w_predicted = w_tree.predict(w_predict_data_encoded)
#         data.loc[w_predict_data.index, 'Item_Weight'] = w_predicted
#     return data


def item_visibility(data):
    for index, item in data.iterrows():
        if item[3] == 0:
            true_value = 0
            tmp_count = 0
            for inx, i in data[data.loc[:, 'Item_Identifier'] == item[0]].iterrows():
                if i[3] == 0:
                    continue
                else:
                    true_value += i[3]
                    tmp_count += 1
            if tmp_count != 0:
                true_value /= tmp_count
                data.iloc[index, 3] = true_value

    visb_data = data.loc[:, 'Item_Visibility']
    visb_data = visb_data.replace(0, np.nan)
    visb_data.fillna(method='ffill', inplace=True)
    data['Item_Visibility'] = visb_data
    return data


# def item_visibility(data):
#     for index, item in data.iterrows():
#         if item[3] == 0:
#             true_value = 0
#             tmp_count = 0
#             for inx, i in data[data.loc[:, 'Item_Identifier'] == item[0]].iterrows():
#                 if i[3] == 0:
#                     continue
#                 else:
#                     true_value += i[3]
#                     tmp_count += 1
#             if tmp_count != 0:
#                 true_value /= tmp_count
#                 data.iloc[index, 3] = true_value
#
#     visb_data = data.loc[:, 'Item_Visibility']   # change 0 to nan in this FT
#     visb_data = visb_data.replace(np.float(0), np.nan)
#     data['Item_Visibility'] = visb_data
#
#     if len(data[pd.isnull(data.iloc[:, 3])]) != 0:
#         v_train_data = data.dropna().iloc[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]]
#         v_train_target = data.dropna().iloc[:, 3]
#         v_predict_data = data[pd.isnull(data.iloc[:, 3])].iloc[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]]
#         v_train_data_encoded = encode(v_train_data)[0]
#         v_predict_data_encoded = encode(v_predict_data)[0]
#
#         v_tree = tree.DecisionTreeRegressor(max_depth=10).fit(v_train_data_encoded, v_train_target)
#         v_predicted = v_tree.predict(v_predict_data_encoded)
#         data.loc[v_predict_data.index, 'Item_Visibility'] = v_predicted
#
#     return data

####### fill missing values ######


####### Encode ######
def encode(data):
    encode_dict = {}
    for obj in ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
                'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']:
        tmp = {}
        for index, item in enumerate(data.loc[:, obj].unique()):
            tmp[item] = index
        encode_dict[obj] = tmp
    data = data.replace(encode_dict)
    return data, encode_dict
####### Encode ######


# ###### Decode ######
# def decode(data, d):
#     decode_dict = {}
#     for root_k, root_v in d.items():
#         tmp = {}
#         for k, v in root_v.items():
#             tmp[v] = k
#         decode_dict[root_k] = tmp
#     data = data.replace(decode_dict)
#     return data
# ###### Decode ######


data_set = pd.read_csv('Train.csv')
test_data = pd.read_csv('Test.csv')
data = data_set.iloc[:, :11]
target = data_set.iloc[:, 11]

data = item_fat(data)
data = outlet_size(data)
data = item_weight(data)
data = item_visibility(data)
data, data_encode_dict = encode(data)

test_data = item_fat(test_data)
test_data = outlet_size(test_data)
test_data = item_weight(test_data)
test_data = item_visibility(test_data)
test_data, test_encode_dict = encode(test_data)

data_set = pd.concat([data, target], axis=1)
data_set.to_csv('Train_Cleaned.csv', index=False)
test_data.to_csv('Test_Cleaned.csv', index=False)
