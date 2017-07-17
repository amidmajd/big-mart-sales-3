from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import forest
from xgboost import XGBRegressor
# import data_clean


data_set = pd.read_csv('Train_Cleaned.csv')
test_data = pd.read_csv('Test_Cleaned.csv')
data = data_set.iloc[:, :11]
target = data_set.iloc[:, 11]


####### Tree ######
# T = tree.DecisionTreeRegressor(max_depth=11).fit(data, target)
# pre_data = T.predict(test_data)
# pre_data = pd.DataFrame(pre_data)
###### Tree ######

# simple = pd.read_csv('SampleSubmission.csv')
# simple['Item_Outlet_Sales'] = pre_data
# simple.to_csv('generated_SampleSubmission.csv')


x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=.2)
model = tree.DecisionTreeRegressor(max_depth=4)
model.fit(x_train, y_train)
#
# ploter = []
# for i in range(1, 50):
#     model = tree.DecisionTreeRegressor(max_depth=i)
#     model.fit(x_train, y_train)
#     ploter.append((i, mean_squared_error(y_test, model.predict(x_test))))
# plt.plot([x for x, y in ploter], [y for x, y in ploter], c='r', marker='.')
# plt.show()
#
print('model 1:',mean_squared_error(y_test, model.predict(x_test)))



model_2 = forest.RandomForestRegressor(max_depth= 6, n_estimators=46)
model_2.fit(x_train, y_train)
print('model 2:',mean_squared_error(y_test, model_2.predict(x_test)))
# 4,5 = 1235606
# ploter = []
# for i in range(1, 100):
#     model = forest.RandomForestRegressor(max_depth=6, n_estimators=i)
#     model.fit(x_train, y_train)
#     ploter.append((i, mean_squared_error(y_test, model.predict(x_test))))
#     print(i,'/',99)
# plt.plot([x for x, y in ploter], [y for x, y in ploter], c='r', marker='.')
# plt.show()

pre_data = model_2.predict(test_data)
pre_data = pd.DataFrame(pre_data)
###### Tree ######

simple = pd.read_csv('SampleSubmission.csv')
simple['Item_Outlet_Sales'] = pre_data
simple.to_csv('generated_SampleSubmission.csv')







# model_3 = XGBRegressor(max_depth=4, n_estimators=12, learning_rate=.2)
# model_3.fit(x_train, y_train)
# print('model 3:',mean_squared_error(y_test, model_3.predict(x_test)))
# 8,40 = 1324449
# 4, 34 = 1150544

# ploter=[]
# for i in range(1, 80):
#     model = XGBRegressor(max_depth=4, n_estimators=i, learning_rate=.2)
#     model.fit(x_train, y_train)
#     ploter.append((i, mean_squared_error(y_test, model.predict(x_test))))
#     print(i,'/',99)
# plt.plot([x for x, y in ploter], [y for x, y in ploter], c='r', marker='.')
# plt.show()

# print('model 3:',mean_squared_error(y_test, model.predict(x_test)))
