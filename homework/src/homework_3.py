import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression

'''
Data preparation
Check if the missing values are presented in the features.
If there are missing values:
For categorical features, replace them with 'NA'
For numerical features, replace with with 0.0
'''

df = pd.read_csv('./homework/data/course_lead_scoring_2026.csv')

categorical = ['lead_source', 'industry', 'employment_status', 'location']
numerical = [col for col in list(df.columns) if col not in categorical]

df[categorical] = df[categorical].fillna('NA')
df[numerical] = df[numerical].fillna(0.0)


#Q1: What is the most frequent observation (mode) for the column industry?
q1 = df['industry'].mode().values[0]


#Q2: 
'''
Create the correlation matrix for the numerical features of your dataset. 
In a correlation matrix, you compute the correlation coefficient between 
every pair of features.
What are the two features that have the biggest correlation?
'''

corr_matrix = df[numerical].corr()
corr_matrix_no_diag = corr_matrix.replace(1.0, np.nan)
q2 = corr_matrix_no_diag.stack().idxmax()


# Split the data.
# Make sure that the target value converted is not in your dataframe.
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=42
)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

x_train, y_train = df_train.drop(columns=['converted']), df_train['converted']
x_val, y_val = df_val.drop(columns=['converted']), df_val['converted']
x_test, y_test = df_test.drop(columns=['converted']), df_test['converted']

#Q3
'''
Calculate the mutual information score between converted and other 
categorical variables in the dataset. Use the training set only.
Round the scores to 2 decimals using round(score, 2).
Which of these variables has the biggest mutual information score?
'''
mi = {}
for cat in categorical:
    mi[cat] = round(mutual_info_score(x_train[cat], y_train), 2)
q3 = max(mi, key=mi.get)


#Q4
'''
Now let's train a logistic regression.
Remember that we have several categorical variables in the dataset. 
Include them using one-hot encoding.
Fit the model on the training dataset.
To make sure the results are reproducible across different versions of 
Scikit-Learn, fit the model with these parameters:
model = LogisticRegression(
solver='liblinear', C=1.0, max_iter=1000, random_state=42
)
Calculate the accuracy on the validation dataset and round it to 
2 decimal digits.
What accuracy did you get?
'''
dv = DictVectorizer(sparse=False)
model = LogisticRegression(
solver='liblinear', C=1.0, max_iter=1000, random_state=42
)

def logistic_train(x, y):
    x_dict = x.to_dict(orient='records')
    x_ohe = dv.fit_transform(x_dict)

    return model.fit(x_ohe, y)

def logistic_predict(x):
    x_dict = x.to_dict(orient='records')
    x_ohe = dv.fit_transform(x_dict)

    return model.predict(x_ohe)

logistic_train(x_train, y_train)
y_val_pred = logistic_predict(x_val)
q4 = round((y_val == y_val_pred).mean(), 2)


#5
'''
Let's find the least useful feature using the feature elimination technique.
Train a model using the same features and parameters as in Q4 
(without rounding).
Now exclude each feature from this set and train a model without it. 
Record the accuracy for each model.
For each feature, calculate the difference between the original accuracy and 
the accuracy without the feature.
Which of following feature has the smallest difference?
'''

features = list(x_train.columns)
accuracy_dict = {}
base_accuracy = float((y_val == y_val_pred).mean())
for feature in features:
    x_train_drop = x_train.drop(columns=[feature])
    x_val_drop = x_val.drop(columns=[feature])
    logistic_train(x_train_drop, y_train)
    y_val_pred = logistic_predict(x_val_drop)
    accuracy = float((y_val == y_val_pred).mean())
    accuracy_dict[feature] = abs(accuracy - base_accuracy)

filtered_dict = {
    k: v for k, v in accuracy_dict.items() 
    if k in ['lead_source', 'number_of_courses_viewed', 'interaction_count']
}
q5 = min(filtered_dict, key=filtered_dict.get)


#Q6
'''
Now let's train a regularized logistic regression.
Let's try the following values of the parameter 
C: [0.000001, 0.00001, 0.0001, 0.001].
Train models using all the features as in Q4.
Calculate the accuracy on the validation dataset and round it to 
3 decimal digits.
Which of these C leads to the best accuracy on the validation set?
'''

c_lst = [0.000001, 0.00001, 0.0001, 0.001]
accuracy_dict = {}
for c in c_lst:
    model = LogisticRegression(
    solver='liblinear', C=c, max_iter=1000, random_state=42
    )
    logistic_train(x_train, y_train)
    y_val_pred = logistic_predict(x_val)
    accuracy = float(round((y_val == y_val_pred).mean(), 3))
    accuracy_dict[c] = accuracy

q6 = max(accuracy_dict, key=accuracy_dict.get)


#Answer
answer = [q1, q2, q3, q4, q5, q6]

for i in range(len(answer)):
    print(f'The answer for Q{i + 1} is {answer[i]}')