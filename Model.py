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
lr = LinearRegression()
lr.fit(x_train, y_train)  #empty LR Data from the following data set
LinearRegression()

# ===============================================
# Applying the model to make the predication
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

# print(y_lr_train_pred) # 80% data
# print(y_lr_test_pred) # 20% data

# ===============================================
#Evalute the model performence
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

print('=====================')
print("lr_train_mse :", lr_train_mse)
print("lr_train_r2 :",lr_train_r2)
print('=====================')
print("lr_test_mse :",lr_test_mse)
print("lr_test_r2 :",lr_test_r2)
print('=====================')

lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Methods','Training MSE','Training R2','Testing MSE','Testing R2']

print(lr_results)

# ===============================================

# Random Forest
# Training the model
