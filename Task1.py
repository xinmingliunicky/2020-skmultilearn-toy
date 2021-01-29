import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

class Task1:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 1================")
        return

    def model_1_run(self):
        print("Model 1:")
        # Train the model 1 with your best hyper parameters (if have) and features on training data.
        #data_train = pd.read_csv('students_train.txt', sep='\t', header=None, names=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','edusupport','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G3'])
        data_train = pd.read_csv('students_train.txt', sep='\t', header=None, names=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','edusupport','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G3'], dtype={'school':'category','sex':'category','address':'category','famsize':'category','Pstatus':'category','Mjob':'category','Fjob':'category','reason':'category','guardian':'category','edusupport':'category','nursery':'category','higher':'category','internet':'category','romantic':'category'})
        X_train = data_train.drop('G3', axis='columns')

        # Drop features of your choice
        #X_train = X_train.drop('school', axis='columns')
        #X_train = X_train.drop('sex', axis='columns')
        #X_train = X_train.drop('age', axis='columns')
        #X_train = X_train.drop('address', axis='columns')
        X_train = X_train.drop('famsize', axis='columns')
        X_train = X_train.drop('Pstatus', axis='columns')
        X_train = X_train.drop('Medu', axis='columns')
        X_train = X_train.drop('Fedu', axis='columns')
        X_train = X_train.drop('Mjob', axis='columns')
        X_train = X_train.drop('Fjob', axis='columns')
        X_train = X_train.drop('reason', axis='columns')
        X_train = X_train.drop('guardian', axis='columns')
        X_train = X_train.drop('traveltime', axis='columns')
        #X_train = X_train.drop('studytime', axis='columns')
        #X_train = X_train.drop('failures', axis='columns')
        X_train = X_train.drop('edusupport', axis='columns')
        X_train = X_train.drop('nursery', axis='columns')
        X_train = X_train.drop('higher', axis='columns')
        X_train = X_train.drop('internet', axis='columns')
        X_train = X_train.drop('romantic', axis='columns')
        X_train = X_train.drop('famrel', axis='columns')
        X_train = X_train.drop('freetime', axis='columns')
        X_train = X_train.drop('goout', axis='columns')
        X_train = X_train.drop('Dalc', axis='columns')
        X_train = X_train.drop('Walc', axis='columns')
        X_train = X_train.drop('health', axis='columns')
        #X_train = X_train.drop('absences', axis='columns')
        #X_train = X_train.drop('G3', axis='columns')

        y_train = data_train.G3
        # Modify OneHotEncoder below as we drop features
        column_trans = make_column_transformer((OneHotEncoder(), ['school','sex','address']), remainder=StandardScaler())
        print("Training and testing set preprocessed by One Hot Encoding and Standard Scaling")
        # Using Linear Regression
        print("Using Linear Regression")
        linreg = LinearRegression()
        pipe = make_pipeline(column_trans, linreg)
        # Training performance across 10-fold CV
        starttime = datetime.datetime.now()
        score = cross_val_score(pipe, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
        endtime = datetime.datetime.now()
        print("10-fold Cross Validation time:\t" + str(endtime-starttime))
        score = -(score.mean())
        print("Mean squared error (Training):\t" + str(score))
        pipe.fit(X_train, y_train)
        print("Linear Regression does not necessarily have valuable tuning parameters.")
        print("Tried to tune by eliminating features and keeping only 'school', 'sex', 'age', 'address', 'studytime', 'failures', 'absences'")
        # Test
        #data_test = pd.read_csv('students_test.txt', sep='\t', header=None, names=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','edusupport','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G3'])
        data_test = pd.read_csv('students_test.txt', sep='\t', header=None, names=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','edusupport','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G3'], dtype={'school':'category','sex':'category','address':'category','famsize':'category','Pstatus':'category','Mjob':'category','Fjob':'category','reason':'category','guardian':'category','edusupport':'category','nursery':'category','higher':'category','internet':'category','romantic':'category'})
        X_test = data_test.drop('G3', axis='columns')

        # Drop the same features here:
        #X_test = X_test.drop('school', axis='columns')
        #X_test = X_test.drop('sex', axis='columns')
        #X_test = X_test.drop('age', axis='columns')
        #X_test = X_test.drop('address', axis='columns')
        X_test = X_test.drop('famsize', axis='columns')
        X_test = X_test.drop('Pstatus', axis='columns')
        X_test = X_test.drop('Medu', axis='columns')
        X_test = X_test.drop('Fedu', axis='columns')
        X_test = X_test.drop('Mjob', axis='columns')
        X_test = X_test.drop('Fjob', axis='columns')
        X_test = X_test.drop('reason', axis='columns')
        X_test = X_test.drop('guardian', axis='columns')
        X_test = X_test.drop('traveltime', axis='columns')
        #X_test = X_test.drop('studytime', axis='columns')
        #X_test = X_test.drop('failures', axis='columns')
        X_test = X_test.drop('edusupport', axis='columns')
        X_test = X_test.drop('nursery', axis='columns')
        X_test = X_test.drop('higher', axis='columns')
        X_test = X_test.drop('internet', axis='columns')
        X_test = X_test.drop('romantic', axis='columns')
        X_test = X_test.drop('famrel', axis='columns')
        X_test = X_test.drop('freetime', axis='columns')
        X_test = X_test.drop('goout', axis='columns')
        X_test = X_test.drop('Dalc', axis='columns')
        X_test = X_test.drop('Walc', axis='columns')
        X_test = X_test.drop('health', axis='columns')
        #X_test = X_test.drop('absences', axis='columns')
        #X_test = X_test.drop('G3', axis='columns')

        y_test = data_test.G3
        y_pred = pipe.predict(X_test)
        mse = metrics.mean_squared_error(y_test, y_pred)

        # Evaluate learned model on testing data, and print the results.
        print("Mean squared error (Testing):\t" + str(mse))
        return






    def model_2_run(self):
        print("--------------------\nModel 2:")
        # Train the model 2 with your best hyper parameters (if have) and features on training data.
        #data_train = pd.read_csv('students_train.txt', sep='\t', header=None, names=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','edusupport','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G3'])
        data_train = pd.read_csv('students_train.txt', sep='\t', header=None, names=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','edusupport','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G3'], dtype={'school':'category','sex':'category','address':'category','famsize':'category','Pstatus':'category','Mjob':'category','Fjob':'category','reason':'category','guardian':'category','edusupport':'category','nursery':'category','higher':'category','internet':'category','romantic':'category'})
        X_train = data_train.drop('G3', axis='columns')

        # Drop features of your choice
        #X_train = X_train.drop('school', axis='columns')
        #X_train = X_train.drop('sex', axis='columns')
        #X_train = X_train.drop('age', axis='columns')
        #X_train = X_train.drop('address', axis='columns')
        X_train = X_train.drop('famsize', axis='columns')
        X_train = X_train.drop('Pstatus', axis='columns')
        X_train = X_train.drop('Medu', axis='columns')
        X_train = X_train.drop('Fedu', axis='columns')
        X_train = X_train.drop('Mjob', axis='columns')
        X_train = X_train.drop('Fjob', axis='columns')
        X_train = X_train.drop('reason', axis='columns')
        X_train = X_train.drop('guardian', axis='columns')
        X_train = X_train.drop('traveltime', axis='columns')
        #X_train = X_train.drop('studytime', axis='columns')
        #X_train = X_train.drop('failures', axis='columns')
        X_train = X_train.drop('edusupport', axis='columns')
        X_train = X_train.drop('nursery', axis='columns')
        X_train = X_train.drop('higher', axis='columns')
        X_train = X_train.drop('internet', axis='columns')
        X_train = X_train.drop('romantic', axis='columns')
        X_train = X_train.drop('famrel', axis='columns')
        X_train = X_train.drop('freetime', axis='columns')
        X_train = X_train.drop('goout', axis='columns')
        X_train = X_train.drop('Dalc', axis='columns')
        X_train = X_train.drop('Walc', axis='columns')
        X_train = X_train.drop('health', axis='columns')
        #X_train = X_train.drop('absences', axis='columns')
        #X_train = X_train.drop('G3', axis='columns')

        y_train = data_train.G3
        # Modify OneHotEncoder below as we drop features
        column_trans = make_column_transformer((OneHotEncoder(), ['school','sex','address']), remainder=StandardScaler())
        print("Training and testing set preprocessed by One Hot Encoding and Standard Scaling")
        # Using SVR
        print("Using Support Vector Regression")
        print("Eliminating features and keeping only 'school', 'sex', 'age', 'address', 'studytime', 'failures', 'absences'")
        # Try out different params below:
        """ param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['rbf', 'poly', 'sigmoid']}
        grid = GridSearchCV(SVR(), param_grid, cv=10, scoring='accuracy')
        pipe = make_pipeline(column_trans, grid)
        pipe.fit(X_train,y_train)
        print(grid.best_estimator_) """
        print("Polynomial Kernel + C=1 turn out to be the best")
        # Continue with the best tune
        svreg = SVR(C=1, kernel='poly')
        pipe = make_pipeline(column_trans, svreg)
        # Training performance across 10-fold CV
        starttime = datetime.datetime.now()
        score = cross_val_score(pipe, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
        endtime = datetime.datetime.now()
        print("10-fold Cross Validation time:\t" + str(endtime-starttime))
        score = -(score.mean())
        print("Mean squared error (Training):\t" + str(score))
        pipe.fit(X_train, y_train)
        # Test
        #data_test = pd.read_csv('students_test.txt', sep='\t', header=None, names=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','edusupport','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G3'])
        data_test = pd.read_csv('students_test.txt', sep='\t', header=None, names=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','edusupport','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G3'], dtype={'school':'category','sex':'category','address':'category','famsize':'category','Pstatus':'category','Mjob':'category','Fjob':'category','reason':'category','guardian':'category','edusupport':'category','nursery':'category','higher':'category','internet':'category','romantic':'category'})
        X_test = data_test.drop('G3', axis='columns')

        # Drop the same features here:
        #X_test = X_test.drop('school', axis='columns')
        #X_test = X_test.drop('sex', axis='columns')
        #X_test = X_test.drop('age', axis='columns')
        #X_test = X_test.drop('address', axis='columns')
        X_test = X_test.drop('famsize', axis='columns')
        X_test = X_test.drop('Pstatus', axis='columns')
        X_test = X_test.drop('Medu', axis='columns')
        X_test = X_test.drop('Fedu', axis='columns')
        X_test = X_test.drop('Mjob', axis='columns')
        X_test = X_test.drop('Fjob', axis='columns')
        X_test = X_test.drop('reason', axis='columns')
        X_test = X_test.drop('guardian', axis='columns')
        X_test = X_test.drop('traveltime', axis='columns')
        #X_test = X_test.drop('studytime', axis='columns')
        #X_test = X_test.drop('failures', axis='columns')
        X_test = X_test.drop('edusupport', axis='columns')
        X_test = X_test.drop('nursery', axis='columns')
        X_test = X_test.drop('higher', axis='columns')
        X_test = X_test.drop('internet', axis='columns')
        X_test = X_test.drop('romantic', axis='columns')
        X_test = X_test.drop('famrel', axis='columns')
        X_test = X_test.drop('freetime', axis='columns')
        X_test = X_test.drop('goout', axis='columns')
        X_test = X_test.drop('Dalc', axis='columns')
        X_test = X_test.drop('Walc', axis='columns')
        X_test = X_test.drop('health', axis='columns')
        #X_test = X_test.drop('absences', axis='columns')
        #X_test = X_test.drop('G3', axis='columns')

        y_test = data_test.G3
        y_pred = pipe.predict(X_test)
        mse = metrics.mean_squared_error(y_test, y_pred)

        # Evaluate learned model on testing data, and print the results.
        print("Mean squared error (Testing):\t" + str(mse))
        return
