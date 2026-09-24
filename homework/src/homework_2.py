import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
The goal of this homework is to create a regression model for predicting the car fuel efficiency (column 'fuel_efficiency_mpg').

Preparing the dataset
Use only the following columns:

'engine_displacement',
'horsepower',
'vehicle_weight',
'model_year',
'fuel_efficiency_mpg'
'''

df = pd.read_csv('./homework/data/car_fuel_efficiency_2026.csv')
df = df.loc[:,(['engine_displacement', 'horsepower', 'vehicle_weight',
               'model_year', 'fuel_efficiency_mpg'])].copy()

'''
EDA
Look at the fuel_efficiency_mpg variable. Does it have a long tail?
'''
plot = sns.histplot(df['fuel_efficiency_mpg'])
# No long tail

#Q1: There's one column with missing values. What is it?
q1 = list(df.columns[df.isna().any()])[0]

#Q2: What's the median (50% percentile) for variable 'horsepower'?
q2 = df['horsepower'].median()

'''
Prepare and split the dataset
Shuffle the filtered dataset and create the split exactly as in the lecture:
'''

n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

np.random.seed(42)
idx = np.arange(n)
np.random.shuffle(idx)
df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train: n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]

#Q3
'''
We need to deal with missing values for the column from Q1.
We have two options: fill it with 0 or with the mean of this variable.
Try both options. For each, train a linear regression model without 
regularization using the code from the lessons.
For computing the mean, use the training only!
Use the validation dataset to evaluate the models and compare the RMSE of 
each option.
Round the RMSE scores to 3 decimal digits using round(score, 3). This keeps 
the imputation difference visible in this release.
Which option gives better RMSE?
'''

def prepare_array(df, y_label):
    y_df = df[y_label]
    x_df = df.drop(columns=[y_label])

    x = np.array(x_df)
    y = np.array(y_df)
    return x, y

def train_linear_regression(x, y):
    ones = np.ones(x.shape[0])
    x = np.column_stack([ones, x])
    xtx = x.T.dot(x)
    xtx_inv = np.linalg.inv(xtx)
    xtx_inv_xt = xtx_inv.dot(x.T)
    w = xtx_inv_xt.dot(y)
    return w[0], w[1:]

def predict(x, w0, w):
    w0_vector =  np.full((x.shape[0], 1), w0)
    y_pred = w0_vector.T + x.dot(w.T)
    return y_pred

def rmse(y, y_pred):
    rmse = np.sqrt(np.mean((y- y_pred) ** 2))
    return rmse

#Training dataset
df_train_fillna_zero = df_train.copy().fillna(0)
df_mean = df_train.mean()
df_train_fillna_mean = df_train.copy().fillna(df_mean)

x_zero, y_zero = prepare_array(df_train_fillna_zero, 'fuel_efficiency_mpg')
x_mean, y_mean = prepare_array(df_train_fillna_mean, 'fuel_efficiency_mpg')

w0_zero, w_zero = train_linear_regression(x_zero, y_zero)
w0_mean, w_mean = train_linear_regression(x_mean, y_mean)

#Validation dataset
df_val_fillna_zero = df_val.copy().fillna(0)
df_val_fillna_mean = df_val.copy().fillna(df_mean)

x_val_zero, y_val_zero = prepare_array(df_val_fillna_zero, 'fuel_efficiency_mpg')
x_val_mean, y_val_mean = prepare_array(df_val_fillna_mean, 'fuel_efficiency_mpg')

#Predict and rmse
pred_val_zero = predict(x_val_zero, w0_zero, w_zero)
pred_val_mean = predict(x_val_mean, w0_mean, w_mean)

rmse_val_zero = round(rmse(y_val_zero, pred_val_zero), 3)
rmse_val_mean = round(rmse(y_val_mean, pred_val_mean), 3)

if rmse_val_zero == rmse_val_mean:
    q3 = "Both are equally good"
elif rmse_val_zero < rmse_val_mean:
    q3 = "With 0"
else:
    q3 = "With mean"

#Q4
'''
Now let's train a regularized linear regression.
For this question, fill the NAs with 0.
Try different values of r from this list: [0, 0.01, 0.1, 1, 5, 10, 100].
Use RMSE to evaluate the model on the validation dataset.
Round the RMSE scores to 4 decimal digits. This keeps the small but real regularization differences visible instead of turning several choices into a tie.
Which r gives the best RMSE?
'''

def train_linear_regression_reg(x, y, r):
    ones = np.ones(x.shape[0])
    x = np.column_stack([ones, x])
    xtx = x.T.dot(x)
    xtx = xtx + r * np.eye(xtx.shape[0])
    xtx_inv = np.linalg.inv(xtx)
    xtx_inv_xt = xtx_inv.dot(x.T)
    w = xtx_inv_xt.dot(y)
    return w[0], w[1:]

df_train_fillna_zero = df_train.copy().fillna(0)
x_zero, y_zero = prepare_array(df_train_fillna_zero, 'fuel_efficiency_mpg')

r_lst = [0, 0.01, 0.1, 1, 5, 10, 100]
result = []
for r in r_lst:
    w0_zero, w_zero = train_linear_regression_reg(x_zero, y_zero, r)

    #Validation dataset
    df_val_fillna_zero = df_val.copy().fillna(0)
    x_val_zero, y_val_zero = prepare_array(df_val_fillna_zero, 'fuel_efficiency_mpg')

    #Predict and rmse
    pred_val_zero = predict(x_val_zero, w0_zero, w_zero)
    rmse_val_zero = round(rmse(y_val_zero, pred_val_zero), 4)
    result.append([r, float(rmse_val_zero)])

result_np = np.array(result)
min_rmse_combination = np.min(result_np, axis = 0)
q4 = min_rmse_combination[0]

#Q5
'''
We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
For each seed, do the train/validation/test split with 60%/20%/20% distribution.
Fill the missing values with 0 and train a model without regularization.
For each seed, evaluate the model on the validation dataset and collect the RMSE scores.
What's the standard deviation of all the scores? To compute the standard deviation, use np.std.
Round the result to 3 decimal digits (round(std, 3))
'''
rmse_result = []
for i in range(10):
    np.random.seed(i)
    idx = np.arange(n)
    np.random.shuffle(idx)
    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train: n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val:]]

    df_train_fillna_zero = df_train.copy().fillna(0)

    x_zero, y_zero = prepare_array(df_train_fillna_zero, 'fuel_efficiency_mpg')

    w0_zero, w_zero = train_linear_regression(x_zero, y_zero)

    #Validation dataset
    df_val_fillna_zero = df_val.copy().fillna(0)

    x_val_zero, y_val_zero = prepare_array(df_val_fillna_zero, 'fuel_efficiency_mpg')

    #Predict and rmse
    pred_val_zero = predict(x_val_zero, w0_zero, w_zero)

    rmse_val_zero = rmse(y_val_zero, pred_val_zero)
    rmse_result.append(rmse_val_zero)

q5 = round(np.std(rmse_result), 3)

#Q6
'''
Split the dataset like previously, use seed 9.
Combine train and validation datasets.
Fill the missing values with 0 and train a model with r=0.001.
What's the RMSE on the test dataset?
'''

np.random.seed(9)
idx = np.arange(n)
np.random.shuffle(idx)
df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train: n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]



df_train_val = pd.concat([df_train, df_val])
df_train_val_fillna_zero = df_train_val.copy().fillna(0)
x_zero, y_zero = prepare_array(df_train_val_fillna_zero, 'fuel_efficiency_mpg')
w0_zero, w_zero = train_linear_regression_reg(x_zero, y_zero, 0.001)

#Validation dataset
df_test_fillna_zero = df_test.copy().fillna(0)
x_test_zero, y_test_zero = prepare_array(df_test_fillna_zero, 'fuel_efficiency_mpg')

#Predict and rmse
pred_test_zero = predict(x_test_zero, w0_zero, w_zero)
q6 = round(rmse(y_test_zero, pred_test_zero), 3)

answer = [q1, q2, q3, q4, q5, q6]

for i in range(len(answer)):
    print(f'The answer for Q{i + 1} is {answer[i]}')