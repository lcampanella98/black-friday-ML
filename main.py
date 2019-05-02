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
import time

df = pd.read_csv('black-friday/BlackFriday.csv')


y_col = 'Purchase'

# 1. DATA PREPARATION & MISSING VALUES
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


def graph_demographics():
    plot_counts('Gender')
    plot_counts('Age')
    plot_counts('Stay_In_Current_City_Years')
    plot_counts('Marital_Status', ['Single', 'Married'])


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


#  detect_outlier(df['Purchase'].array)
#  print('Found {} outliers: {}'.format(len(outliers), outliers))

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


def plot_model_error(title, labels, errs):
    plt.bar(labels, errs)
    plt.ylim(0, 5000)
    plt.title(title)
    plt.show()


# returns tuple of (rmse, running time)
def train_model(model, data, X_cols, y_col, train_size, should_print=True):
    X = data[X_cols]
    y = data[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size)

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    if should_print:
        print('-' * 50)
        print('\t\t{}:'.format(type(model).__name__))

        print('\t\tRMSE = {}'.format(rmse))
        print('\t\tRunning time = {} sec'.format(end-start))
        print('-' * 50)
    return rmse, end-start


#  4. TRAIN MODELS
def train_models():
    model_labels = ['Linear Regression', 'Regression Tree']

    #  Case 1: With user and product specific attributes
    desc = "WITH USER & PRODUCT SPECIFIC INFO"
    print(('*' * 30) + ' {} '.format(desc) + ('*' * 30))

    case1_rmse = []
    rmse, runtime = train_model(LinearRegression(), df_enc, X_cols_case1, y_col, 0.7)
    case1_rmse.append(rmse)

    #  Model 2: Regression Tree
    rmse, runtime = train_model(DecisionTreeRegressor(max_depth=10), df_enc, X_cols_case1, y_col, 0.7)
    case1_rmse.append(rmse)
    plot_model_error('RMSE error WITH user/product specific info', model_labels, case1_rmse)

    #  Case 2: WITHOUT user and product specific attributes
    print()
    desc = "WITH USER & PRODUCT SPECIFIC INFO"
    print(('*' * 30) + ' {} '.format(desc) + ('*' * 30))

    case2_rmse = []
    rmse, runtime = train_model(LinearRegression(), df_enc, X_cols_case2, y_col, 0.7)
    case2_rmse.append(rmse)

    #  Model 2: Regression Tree
    rmse, runtime = train_model(DecisionTreeRegressor(max_depth=10), df_enc, X_cols_case2, y_col, 0.7)
    case2_rmse.append(rmse)
    plot_model_error('RMSE error WITHOUT user/product specific info', model_labels, case2_rmse)


#  train_models()

def run_scalability_analysis(plot=True):
    x_cols = X_cols_case1
    train_size_range = np.linspace(0.1, 1.0, endpoint=False, num=9)

    model = LinearRegression()
    linreg_runtimes = [
        train_model(model, df_enc, x_cols, y_col, train_size, should_print=False)[1]
        for train_size
        in train_size_range
    ]

    model = DecisionTreeRegressor(max_depth=10)
    tree_runtimes = [
        train_model(model, df_enc, x_cols, y_col, train_size, should_print=False)[1]
        for train_size
        in train_size_range
    ]
    if plot:
        plt.title('Scalability Analysis (Running time (secs) by train set size)')
        plt.scatter(train_size_range, linreg_runtimes, c='blue')
        plt.scatter(train_size_range, tree_runtimes, c='orange')
        plt.plot(train_size_range, linreg_runtimes, c='blue')
        plt.plot(train_size_range, tree_runtimes, c='orange')
        plt.legend(('Linear Regression', 'Decision Tree Regression'))
        plt.show()

    return train_size_range, {
        'linreg': linreg_runtimes,
        'tree': tree_runtimes
    }


#  run_scalability_analysis()


def run_robustness_analysis(plot=True):
    X = df_enc[X_cols_case1]
    y = df_enc[y_col]

    lin_reg = LinearRegression()
    tree_reg = DecisionTreeRegressor(max_depth=10)

    krange = range(2, 12)
    lin_err = []
    tree_err = []

    for k in krange:
        scores = cross_val_score(lin_reg, X, y, cv=k)
        lin_err.append(np.mean(scores))
        scores = cross_val_score(tree_reg, X, y, cv=k)
        tree_err.append(np.mean(scores))

    if plot:
        plt.title('Robustness Analysis')
        plt.scatter(krange, lin_err, c='blue')
        plt.scatter(krange, tree_err, c='orange')
        plt.plot(krange, lin_err, c='blue')
        plt.plot(krange, tree_err, c='orange')
        plt.legend(('Linear Regression', 'Decision Tree Regression'))
        plt.show()


#  run_robustness_analysis()


def run_kmeans():

    #  CLUSTERING
    cluster_cols = X_cols_case2 + [y_col]
    X = df_enc[cluster_cols]

    #  scale continuous columns to between 0 and 1 to give equal weight to all columns
    mms = MinMaxScaler()
    mms.fit(X)
    X = mms.transform(X)

    #  Determine Optimal k
    sum_squared_dist = []
    min_k, max_k = 1, 24
    K = range(min_k, max_k+1)
    print('Running k-means for k in range ({}, {})'.format(min_k, max_k))
    for k in K:
        print('k:', k)
        km = KMeans(n_clusters=k)
        km = km.fit(X)
        sum_squared_dist.append(km.inertia_)

    plt.plot(K, sum_squared_dist, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    #  Just run kmeans with optimal k
    # best_k = 18
    # km = KMeans(n_clusters=best_k)
    # km = km.fit(X)


#  run_kmeans()
