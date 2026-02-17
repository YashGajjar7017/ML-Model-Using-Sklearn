import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('delaney_solubility_with_descriptors.csv')
# print(df)

# ===============================================
# Data Separation
y = df['logS']
# print(y)

# ===============================================
# Drop the Logs Function from the dataset
x = df.drop('logS',axis=1)
# print(x)

# ===============================================
# Data Splitiing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# print(x_test)
# print(x_train)

# ===============================================
# Model Building : Linear Regression
# 1. Training The Model
lr = LinearRegression()
lr.fit(x_train, y_train)  #empty LR Data from the following data set
LinearRegression()

# ===============================================
# 2. Applying the model to make the predication
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

# print(y_lr_train_pred) # 80% data
# print(y_lr_test_pred) # 20% data

# ===============================================
# 3. Evalute the model performence
# print(y_train)
# print(y_lr_train_pred)

# ===============================================
from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
#square co-relation & co-effient
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
#square co-relation & co-effient
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# print('=====================')
# print("lr_train_mse :", lr_train_mse)
# print("lr_train_r2 :",lr_train_r2)
# print('=====================')
# print("lr_test_mse :",lr_test_mse)
# print("lr_test_r2 :",lr_test_r2)
# print('=====================')

lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Methods','Training MSE','Training R2','Testing MSE','Testing R2']

# print(lr_results)

# ===============================================

# |----------- Random Forest ---------------|
# 1. Training the model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2,random_state=100)
rf.fit(x_train,y_train)

# ===============================================
# 2. Applying the model too make the predication
y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

# ===============================================
# 3. Evalute the model

from sklearn.metrics import mean_squared_error, r2_score

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
#square co-relation & co-effient
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
#square co-relation & co-effient
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Methods','Training MSE','Training R2','Testing MSE','Testing R2']

# print(lr_results)

# ===============================================
# Model Comparison

df_models = pd.concat([lr_results,rf_results], axis=0)
print(df_models)

df_models.reset_index(drop=True)

# ===============================================
# Data visulization of predication result
import matplotlib.pyplot as plt
import numpy as np

z = np.polyfit(y_train,y_lr_train_pred,1)
p = np.poly1d(z)

plt.scatter(x=y_train, y=y_lr_train_pred, c='#7CAE00',alpha=0.3)
plt.figure(figsize=(10,10))
plt.xlabel("X-axis Label")
plt.ylabel("Y-axis Label")
plt.title("Random Forest & linear Regression")
plt.plot()
plt.show()