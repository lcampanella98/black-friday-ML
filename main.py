"""
CS 301 Intro to Data Science Project
Project: Black Friday
Group: Nikhileshwarananda Vummadi, Lorenzo Campanella
Date: 4/25/2019
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('black-friday/BlackFriday.csv')


y_col = 'Purchase'

# 1. DATA PREPARATION
#  check which columns have null/NaN values
#  print(df.isna().any())

#  since the only columns with NaN values are Product_Category_2 and Product_Category_3 columns, we can
#  fill them with zero, indicating that the product is not a member of any of those categories
df.fillna(value=0, inplace=True)

#  convert floating type columns to ints
df['Product_Category_2'] = df['Product_Category_2'].astype('int64')
df['Product_Category_3'] = df['Product_Category_3'].astype('int64')


def plot_counts(col, labels=None):
    cts = df[col].value_counts(sort=False).sort_index()
    plt.bar(labels if labels else cts.index, cts.array)
    plt.title(col + ' counts')
    plt.show()

# plot_counts('Gender')
# plot_counts('Age')


#  2. OUTLIER DETECTION
outliers = []

def detect_outlier(data_1):
    threshold = 2.935  # minimum standard deviations away from mean for outliers
    mean_1 = np.mean(data_1)
    std_1 = np.std(data_1)

    for y in data_1:
        z_score = (y - mean_1) / std_1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

#detect_outlier(df['Purchase'].array)
#print('Found {} outliers: {}'.format(len(outliers), outliers))

#  3. FEATURE ENGINEERING

#  stores will not be predicting purchase based on an existing user, or based
#  on a particular product, but on the low-level data like Age, Gender, Marital Status, etc.
#  Therefore, we will train one model with this data, and one model without

#  create a regression dataframe with dummy variables from categorical columns
categorical_cols = [
    'Gender',
    'Age',
    'Occupation',
    'City_Category',
    'Stay_In_Current_City_Years',
    'Marital_Status',
    'Product_Category_1',
    'Product_Category_2',
    'Product_Category_3'
]
df_enc = pd.get_dummies(df, columns=categorical_cols)


#  Regression model with user/product-specific information
label_encoder_userid = LabelEncoder()
df_enc['User_ID'] = label_encoder_userid.fit_transform(df['User_ID'])
label_encoder_productid = LabelEncoder()
df_enc['Product_ID'] = label_encoder_productid.fit_transform(df['Product_ID'])

user_or_product_columns = ['User_ID', 'Product_ID', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3']


# for col in df:
#     print(col + ': {} unique values'.format(df[col].nunique()))
#     print('\t',df[col].unique(), '\n')

X_cols_case1 = list(filter(lambda c: c != y_col, list(df_enc)))
X_cols_case2 = list(filter(lambda c: c != y_col and c not in user_or_product_columns and not c.startswith('Product_Category'), list(df_enc)))

train_models = False
#  4. TRAIN MODELS
if train_models:
    #  Case 1: With user and product specific attributes
    print(('*'*30)+' WITH USER & PRODUCT SPECIFIC INFO '+('*'*30))

    X = df_enc[X_cols_case1]
    y = df_enc[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #  Model 1: Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)
    print('-'*50)
    print('\t\tLinear Regression:')
    print('\t\tRMSE = {}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))
    print('-'*50)
    #scores = cross_val_score(lin_reg, X, y, cv=10)
    #print(scores)
    print()

    #  Model 2: Regression Tree
    tree = DecisionTreeRegressor(max_depth=10)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    #scores = cross_val_score(tree, X, y, cv=10)
    #print(scores)
    print('-'*50)
    print('\t\tRegression Tree:')
    print('\t\tRMSE = {}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))
    print('-'*50)

    #  Case 2: WITHOUT user and product specific attributes
    print()
    print(('*'*30)+' WITHOUT USER & PRODUCT SPECIFIC INFO '+('*'*30))

    #print('X cols: {}'.format(X_cols))
    X = df_enc[X_cols_case2]
    y = df_enc[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


    #  Model 1: Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)
    print('-'*50)
    print('\t\tLinear Regression:')
    print('\t\tRMSE = {}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))
    print('-'*50)
    #scores = cross_val_score(lin_reg, X, y, cv=10)
    #print(scores)
    print()

    #  Model 2: Regression Tree
    tree = DecisionTreeRegressor(max_depth=10)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    #scores = cross_val_score(tree, X, y, cv=10)
    #print(scores)
    print('-'*50)
    print('\t\tRegression Tree:')
    print('\t\tRMSE = {}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))
    print('-'*50)

#  CLUSTERING
kmeans = KMeans()
cluster_cols = X_cols_case2 + [y_col]
X = df_enc[cluster_cols]

#  scale continuous columns to between 0 and 1 to give equal weight to all columns
mms = MinMaxScaler()
mms.fit(X)
X = mms.transform(X)

#  Determine Optimal k
sum_squared_dist = []
K = range(1,25)
for k in K:
    print('k:', k)
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    sum_squared_dist.append(km.inertia_)

plt.plot(K, sum_squared_dist, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method For Optimal k - Case 2')
plt.show()

#  Just run kmeans with optimal k
# best_k = 18
# km = KMeans(n_clusters=best_k)
# km = km.fit(X)