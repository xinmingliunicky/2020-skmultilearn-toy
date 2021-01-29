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

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from skmultilearn.problem_transform import BinaryRelevance

class Task3:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 3================")
        return


    def model_1_run(self):
        print("Model 1:")
        # Train the model 1 with your best hyper parameters (if have) and features on training data.
        data_train = pd.read_csv('students_train.txt', sep='\t', header=None, names=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','edusupport','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G3'], dtype={'school':'category','sex':'category','address':'category','famsize':'category','Pstatus':'category','Mjob':'category','Fjob':'category','reason':'category','guardian':'category','edusupport':'category','nursery':'category','higher':'category','internet':'category','romantic':'category'})
        X_train = data_train.drop('edusupport', axis='columns')

        # Drop features of your choice
        #X_train = X_train.drop('school', axis='columns')
        X_train = X_train.drop('sex', axis='columns')
        X_train = X_train.drop('age', axis='columns')
        #X_train = X_train.drop('address', axis='columns')
        X_train = X_train.drop('famsize', axis='columns')
        #X_train = X_train.drop('Pstatus', axis='columns')
        X_train = X_train.drop('Medu', axis='columns')
        X_train = X_train.drop('Fedu', axis='columns')
        X_train = X_train.drop('Mjob', axis='columns')
        X_train = X_train.drop('Fjob', axis='columns')
        X_train = X_train.drop('reason', axis='columns')
        X_train = X_train.drop('guardian', axis='columns')
        X_train = X_train.drop('traveltime', axis='columns')
        #X_train = X_train.drop('studytime', axis='columns')
        #X_train = X_train.drop('failures', axis='columns')
        #X_train = X_train.drop('edusupport', axis='columns')
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

        y_train_raw = data_train.edusupport
        # Binary Encode multiple labels
        y_train = pd.Series(y_train_raw).str.get_dummies(' ')
        #print(y_train)

        # Modify OneHotEncoder below as we drop features
        column_trans = make_column_transformer((OneHotEncoder(), ['school','address','Pstatus']), remainder=StandardScaler())
        print("Training and testing set preprocessed by Binary Recoding, One Hot Encoding and Standard Scaling")
        # Using Logistic Regression
        print("Using Logistic Regression")
        # Try out different params below:
        """ param_grid = {'C': [0.1, 1, 10, 100]}
        grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=10, scoring='f1_macro')
        pipe = make_pipeline(column_trans, grid)
        pipe.fit(X_train,y_train)
        print(grid.best_estimator_) """
        print("C=1 turn out to be the best")
        # Continue with the best tune
        # Using a Binary Relevance Classifier
        binary_rel_clf = BinaryRelevance(LogisticRegression(solver='lbfgs', C=0.1, max_iter=10000))
        pipe = make_pipeline(column_trans, binary_rel_clf)
        print("Eliminating features and keeping only 'school', 'address', 'Pstatus', 'studytime', 'failures', 'absences', 'G3'")
        # Training performance across 10-fold CV
        starttime = datetime.datetime.now()
        score = cross_val_score(pipe, X_train, y_train, cv=10, scoring='accuracy')
        endtime = datetime.datetime.now()
        print("10-fold Cross Validation time:\t" + str(endtime-starttime))
        score = score.mean()
        print("Accuracy (Training):\t" + str(score))
        pipe.fit(X_train, y_train)
        # Test
        data_test = pd.read_csv('students_test.txt', sep='\t', header=None, names=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','edusupport','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G3'], dtype={'school':'category','sex':'category','address':'category','famsize':'category','Pstatus':'category','Mjob':'category','Fjob':'category','reason':'category','guardian':'category','edusupport':'category','nursery':'category','higher':'category','internet':'category','romantic':'category'})
        X_test = data_test.drop('edusupport', axis='columns')

        # Drop the same features here:
        #X_test = X_test.drop('school', axis='columns')
        X_test = X_test.drop('sex', axis='columns')
        X_test = X_test.drop('age', axis='columns')
        #X_test = X_test.drop('address', axis='columns')
        X_test = X_test.drop('famsize', axis='columns')
        #X_test = X_test.drop('Pstatus', axis='columns')
        X_test = X_test.drop('Medu', axis='columns')
        X_test = X_test.drop('Fedu', axis='columns')
        X_test = X_test.drop('Mjob', axis='columns')
        X_test = X_test.drop('Fjob', axis='columns')
        X_test = X_test.drop('reason', axis='columns')
        X_test = X_test.drop('guardian', axis='columns')
        X_test = X_test.drop('traveltime', axis='columns')
        #X_test = X_test.drop('studytime', axis='columns')
        #X_test = X_test.drop('failures', axis='columns')
        #X_test = X_test.drop('edusupport', axis='columns')
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

        y_test_raw = data_test.edusupport
        # Binary Encode multiple labels
        y_test = pd.Series(y_test_raw).str.get_dummies(' ')
        #print(y_test)

        y_pred = pipe.predict(X_test)
        #print(y_pred.toarray())
        acc = metrics.accuracy_score(y_test, y_pred)
        ham = metrics.hamming_loss(y_test, y_pred)

        # Evaluate learned model on testing data, and print the results.
        print("Testing results:")
        print("Accuracy\t" + str(acc) + "\tHamming loss\t" + str(ham))
        return





    def model_2_run(self):
        print("--------------------\nModel 2:")
        # Train the model 2 with your best hyper parameters (if have) and features on training data.
        data_train = pd.read_csv('students_train.txt', sep='\t', header=None, names=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','edusupport','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G3'], dtype={'school':'category','sex':'category','address':'category','famsize':'category','Pstatus':'category','Mjob':'category','Fjob':'category','reason':'category','guardian':'category','edusupport':'category','nursery':'category','higher':'category','internet':'category','romantic':'category'})
        X_train = data_train.drop('edusupport', axis='columns')

        # Drop features of your choice
        #X_train = X_train.drop('school', axis='columns')
        X_train = X_train.drop('sex', axis='columns')
        X_train = X_train.drop('age', axis='columns')
        #X_train = X_train.drop('address', axis='columns')
        X_train = X_train.drop('famsize', axis='columns')
        #X_train = X_train.drop('Pstatus', axis='columns')
        X_train = X_train.drop('Medu', axis='columns')
        X_train = X_train.drop('Fedu', axis='columns')
        X_train = X_train.drop('Mjob', axis='columns')
        X_train = X_train.drop('Fjob', axis='columns')
        X_train = X_train.drop('reason', axis='columns')
        X_train = X_train.drop('guardian', axis='columns')
        X_train = X_train.drop('traveltime', axis='columns')
        #X_train = X_train.drop('studytime', axis='columns')
        #X_train = X_train.drop('failures', axis='columns')
        #X_train = X_train.drop('edusupport', axis='columns')
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

        y_train_raw = data_train.edusupport
        # Binary Encode multiple labels
        y_train = pd.Series(y_train_raw).str.get_dummies(' ')
        #print(y_train)

        # Modify OneHotEncoder below as we drop features
        column_trans = make_column_transformer((OneHotEncoder(), ['school','address','Pstatus']), remainder=StandardScaler())
        print("Training and testing set preprocessed by Binary Recoding, One Hot Encoding and Standard Scaling")
        # Using kNN
        print("Using kNN")
        # Try out different params below:
        """ param_grid = {'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
        grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10, scoring='accuracy')
        pipe = make_pipeline(column_trans, grid)
        pipe.fit(X_train,y_train)
        print(grid.best_estimator_) """
        print("k=7 turn out to be the best")
        # Continue with the best tune
        # Using a Binary Relevance Classifier
        binary_rel_clf = BinaryRelevance(KNeighborsClassifier(n_neighbors=7))
        pipe = make_pipeline(column_trans, binary_rel_clf)
        print("Eliminating features and keeping only 'school', 'address', 'Pstatus', 'studytime', 'failures', 'absences', 'G3'")
        # Training performance across 10-fold CV
        starttime = datetime.datetime.now()
        score = cross_val_score(pipe, X_train, y_train, cv=10, scoring='accuracy')
        endtime = datetime.datetime.now()
        print("10-fold Cross Validation time:\t" + str(endtime-starttime))
        score = score.mean()
        print("Accuracy (Training):\t" + str(score))
        pipe.fit(X_train, y_train)

        # Test
        data_test = pd.read_csv('students_test.txt', sep='\t', header=None, names=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','edusupport','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G3'], dtype={'school':'category','sex':'category','address':'category','famsize':'category','Pstatus':'category','Mjob':'category','Fjob':'category','reason':'category','guardian':'category','edusupport':'category','nursery':'category','higher':'category','internet':'category','romantic':'category'})
        X_test = data_test.drop('edusupport', axis='columns')

        # Drop the same features here:
        #X_test = X_test.drop('school', axis='columns')
        X_test = X_test.drop('sex', axis='columns')
        X_test = X_test.drop('age', axis='columns')
        #X_test = X_test.drop('address', axis='columns')
        X_test = X_test.drop('famsize', axis='columns')
        #X_test = X_test.drop('Pstatus', axis='columns')
        X_test = X_test.drop('Medu', axis='columns')
        X_test = X_test.drop('Fedu', axis='columns')
        X_test = X_test.drop('Mjob', axis='columns')
        X_test = X_test.drop('Fjob', axis='columns')
        X_test = X_test.drop('reason', axis='columns')
        X_test = X_test.drop('guardian', axis='columns')
        X_test = X_test.drop('traveltime', axis='columns')
        #X_test = X_test.drop('studytime', axis='columns')
        #X_test = X_test.drop('failures', axis='columns')
        #X_test = X_test.drop('edusupport', axis='columns')
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

        y_test_raw = data_test.edusupport
        # Binary Encode multiple labels
        y_test = pd.Series(y_test_raw).str.get_dummies(' ')
        #print(y_test)

        y_pred = pipe.predict(X_test)
        #print(y_pred.toarray())
        acc = metrics.accuracy_score(y_test, y_pred)
        ham = metrics.hamming_loss(y_test, y_pred)

        # Evaluate learned model on testing data, and print the results.
        print("Testing results:")
        print("Accuracy\t" + str(acc) + "\tHamming loss\t" + str(ham))
        return
