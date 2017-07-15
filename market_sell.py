import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np

####### fill missing values ######
def item_fat(data):
    data = data.replace({'Item_Fat_Content': {'reg': 'Regular', 'low fat': 'Low Fat', 'LF': 'Low Fat'}})
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


def outlet_size(data):
    for inx, item in data.iterrows():
        if pd.isnull(item['Outlet_Size']):
            if item['Outlet_Type'] == 'Grocery Store':
                data.loc[inx, 'Outlet_Size'] = 'Small'
            elif item['Outlet_Type'] == 'Supermarket Type1':
                # siml = data[data.loc[:, 'Outlet_Location_Type'] == 'Tier 2'].iloc[:, 8]  =>>> 'Small'
                data.loc[inx, 'Outlet_Size'] = 'Small'
    return data
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


####### Decode ######
# def decode(data, d):
#     decode_dict = {}
#     for root_k, root_v in d.items():
#         tmp = {}
#         for k, v in root_v.items():
#             tmp[v] = k
#         decode_dict[root_k] = tmp
#     data = data.replace(decode_dict)
#     return data
####### Decode ######


# data_set = pd.read_csv('Fixed_Train.csv')
# test_data = pd.read_csv('Fixed_Test.csv')
data_set = pd.read_csv('Train.csv')
test_data_set = pd.read_csv('Test.csv')
test_data = test_data_set
data = data_set.iloc[:, :11]
target = data_set.iloc[:, 11]


data = item_fat(data)
data = item_weight(data)
data = item_visibility(data)
data = outlet_size(data)
data, data_encode_dict = encode(data)

test_data = item_fat(test_data)
test_data = item_weight(test_data)
test_data = item_visibility(test_data)
test_data = outlet_size(test_data)
test_data, test_encode_dict = encode(test_data)


# DATA = pd.concat([data, target], axis=1)
# DATA.to_csv('Fixed_Train.csv', index=False)
# test_data.to_csv('Fixed_Test.csv', index=False)


####### Tree ######
T = tree.DecisionTreeRegressor().fit(data, target)
pre_data = T.predict(test_data)
pre_data = pd.DataFrame(pre_data)
####### Tree ######

simple = pd.read_csv('SampleSubmission.csv')
simple['Item_Outlet_Sales'] = pre_data
simple.to_csv('generated_SampleSubmission.csv')


# x_tr, x_ts, y_tr, y_ts = train_test_split(data, target, test_size=0.3)
# T = tree.DecisionTreeRegressor().fit(x_tr, y_tr)
# print(T.score(x_ts, y_ts) * 100)