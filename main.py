# Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer # to replace 'nan' values to average values
from sklearn.compose import ColumnTransformer # to transform categorical values to binary for machine learning
from sklearn.preprocessing import OneHotEncoder # to transform categorical values to binary for machine learning
from sklearn.preprocessing import LabelEncoder # to transform categorical (Yes/No) to 0 and 1
from sklearn.model_selection import train_test_split # to split dataset for machine learning
from sklearn.preprocessing import StandardScaler # for feature scaling

def data_processing():

    # import data
    data = pd.read_csv('Data.csv')
    print(data.head(3))

    # independent and dependent variables
    independent = data.iloc[:, :-1].values
    dependent = data.iloc[:, -1].values
    print(independent, dependent)

    # replace 'nan' data to average
    impute = SimpleImputer(missing_values=np.nan, strategy='mean') # prepare to replace values to mean
    impute.fit(independent[:, 1:3]) # find the values from column 1 and 2 (independent)
    independent[:, 1:3] = impute.transform(independent[:, 1:3]) # transform return the values, and if there is no independent[:, 1:3] = then, indepedent values would miss [Coutry] and [Purchased] columns
    print(independent)

    # to transform categorical values to binary
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough') # treanform=[( 'name', class, column/s ), reminder = keep the values]
    independent = np.array(ct.fit_transform(independent)) # using np.array is for machine learning
    print(independent)

    # to transform Yes/No to 1/0
    le = LabelEncoder() # does not need to specify the values as 1/0 in this case (0 to n-1 values)
    dependent = le.fit_transform(dependent) # return the values to dependent table
    print(dependent)

    # split dataset into train and test for machine learning (train 80, 67, 50: test 20, 33, 50)
    # It can be used for classification or regression problems and can be used for any supervised learning algorithm.
    # the parts are usually 4 parts which are x_train, x_test, y_train, and y_test

    # Train Dataset: Used to fit the machine learning model.
    # Test Dataset: Used to evaluate the fit machine learning model.

    x_train, x_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.2, random_state=1)
    # test-size = 0.2 is [train:test 8:2],
    # Another important consideration is that rows are assigned to the train and test sets randomly

    print(x_train)
    print(x_test)
    print(y_train)
    print(y_test)

    # feature scaling for machine learning
    sc = StandardScaler()
    x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
    x_test[:, 3:] = sc.transform(x_test[:, 3:])

    print(x_train)
    print(x_test)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_processing()
