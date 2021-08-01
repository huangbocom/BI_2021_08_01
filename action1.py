# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from deepctr.models import WDL
from deepctr.inputs import SparseFeat, get_feature_names


# 数据读取


# movielens完整数据集
ratings = pd.read_csv("./movielens/ratings.csv")
movies = pd.read_csv("./movielens/movies.csv")
data = pd.merge(ratings,movies,on='movieId')
sparse_features = ["movieId","userId", "genres","title"]
target = ["rating"]

print(data[target])

# 编码
for feature in sparse_features:
    lbe = LabelEncoder()#
    data[feature] = lbe.fit_transform(data[feature])

fixlen_feature_columns = [SparseFeat(feature, data[feature].nunique()) for feature in sparse_features] #embedding
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

train, test = train_test_split(data, test_size=0.2)
train_model_input = {name:train[name].values for name in feature_names} #dict
test_model_input = {name:test[name].values for name in feature_names}

# Wide&Deep进行训练
model = WDL(linear_feature_columns, dnn_feature_columns, task='regression')
model.compile("adam", "mse", metrics=['mse'], )
history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=1, verbose=True, validation_split=0.2, )
# Wide&Deep进行预测
pred_ans = model.predict(test_model_input, batch_size=256)
# RMSE或MSE
mse = round(mean_squared_error(test[target].values, pred_ans), 4)
rmse = mse ** 0.5
print(" RMSE", rmse)
