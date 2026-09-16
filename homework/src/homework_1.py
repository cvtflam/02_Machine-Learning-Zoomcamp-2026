import pandas as pd
import numpy as np
from pathlib import Path

csv = Path('./homework/data/car_fuel_efficiency_2026.csv')
df = pd.read_csv(csv)

# Q1: What version of Pandas did you install?
q1=pd.__version__

#Q2: How many records are in the dataset?
q2 = df.shape[0]

#Q3: How many fuel types are presented in the dataset?
q3 = len(df['fuel_type'].unique())

#Q4: How many columns in the dataset have missing values?
na_count = df.isna().sum()
q4 = len(na_count[na_count > 0])

# Q5: What's the maximum fuel efficiency of cars from Asia?
q5 = df.loc[df['origin'] == 'Asia', 'fuel_efficiency_mpg'].max()

# Q6: 
'''
Find the median value of the horsepower column in the dataset.
Next, calculate the most frequent value of the same horsepower column.
Use the fillna method to fill the missing values in the horsepower column with 
the most frequent value from the previous step.
Now, calculate the median value of horsepower once again.
Has it changed?
'''

median = df['horsepower'].median()
frequent_value = df.groupby('horsepower').agg('size').max()
df['horsepower'].fillna(frequent_value)

median_new = df['horsepower'].median()
q6 = "Changed" if median != median_new else "Not changed"

#Q7:
'''
Select all the cars from Asia
Select only columns vehicle_weight and model_year
Select the first 7 values
Get the underlying NumPy array. Let's call it X.
Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
Invert XTX.
Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
What's the sum of all the elements of the result?
'''

asia = df.loc[df['origin'] == 'Asia', ['vehicle_weight', 'model_year']]
asia_first_7 = asia.head(7)
X = asia_first_7.to_numpy()
XTX = np.dot(X.T, X)
XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = np.dot(np.dot(XTX_inv, X.T), y)
q7 = w.sum()

answer = [q1, q2, q3, q4, q5, q6, q7]

for i in range(len(answer)):
    print(f'The answer for Q{i + 1} is {answer[i]}')