from sklearn import tree
import pandas as pd
import numpy as np
# import data_clean


data_set = pd.read_csv('Train_Cleaned.csv')
test_data = pd.read_csv('Test_Cleaned.csv')
data = data_set.iloc[:, :11]
target = data_set.iloc[:, 11]


####### Tree ######
T = tree.DecisionTreeRegressor(max_depth=11).fit(data, target)
pre_data = T.predict(test_data)
pre_data = pd.DataFrame(pre_data)
###### Tree ######

simple = pd.read_csv('SampleSubmission.csv')
simple['Item_Outlet_Sales'] = pre_data
simple.to_csv('generated_SampleSubmission.csv')
