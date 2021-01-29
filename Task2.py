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
from sklearn.svm import SVC

class Task2:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 2================")
        return

    def print_category_results(self, category, precision, recall, f1):
        print("Category\t" + category + "\tF1\t" + str(f1) + "\tPrecision\t" + str(precision) + "\tRecall\t" + str(
            recall))

    def print_macro_results(self, accuracy, precision, recall, f1):
        print("Accuracy\t" + str(accuracy) + "\tMacro_F1\t" + str(f1) + "\tMacro_Precision\t" + str(
            precision) + "\tMacro_Recall\t" + str(recall))


    def model_1_run(self):
        print("Model 1:")
        # Train the model 1 with your best hyper parameters (if have) and features on training data.
        data_train = pd.read_csv('students_train.txt', sep='\t', header=None, names=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','edusupport','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G3'], dtype={'school':'category','sex':'category','address':'category','famsize':'category','Pstatus':'category','Mjob':'category','Fjob':'category','reason':'category','guardian':'category','edusupport':'category','nursery':'category','higher':'category','internet':'category','romantic':'category'})
        X_train = data_train.drop('Mjob', axis='columns')

        # Drop features of your choice
        #X_train = X_train.drop('school', axis='columns')
        X_train = X_train.drop('sex', axis='columns')
        X_train = X_train.drop('age', axis='columns')
        #X_train = X_train.drop('address', axis='columns')
        X_train = X_train.drop('famsize', axis='columns')
        #X_train = X_train.drop('Pstatus', axis='columns')
        #X_train = X_train.drop('Medu', axis='columns')
        #X_train = X_train.drop('Fedu', axis='columns')
        #X_train = X_train.drop('Mjob', axis='columns')
        #X_train = X_train.drop('Fjob', axis='columns')
        X_train = X_train.drop('reason', axis='columns')
        X_train = X_train.drop('guardian', axis='columns')
        X_train = X_train.drop('traveltime', axis='columns')
        X_train = X_train.drop('studytime', axis='columns')
        X_train = X_train.drop('failures', axis='columns')
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
        X_train = X_train.drop('absences', axis='columns')
        X_train = X_train.drop('G3', axis='columns')

        y_train = data_train.Mjob
        # Modify OneHotEncoder below as we drop features
        column_trans = make_column_transformer((OneHotEncoder(), ['school','address','Pstatus','Fjob']), remainder=StandardScaler())
        print("Training and testing set preprocessed by One Hot Encoding and Standard Scaling")
        # Using Logistic Regression
        print("Using Logistic Regression")
        # Try out different params below:
        """ param_grid = {'C': [0.1, 1, 10, 100]}
        grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=10, scoring='f1_macro')
        pipe = make_pipeline(column_trans, grid)
        pipe.fit(X_train,y_train)
        print(grid.best_estimator_) """
        print("C=10 turn out to be the best")
        # Continue with the best tune
        logreg = LogisticRegression(solver='lbfgs', C=10, max_iter=10000)
        pipe = make_pipeline(column_trans, logreg)
        print("Eliminating features and keeping only 'school', 'address', 'Pstatus', 'Medu', 'Fedu', 'Fjob'")
        # Training performance across 10-fold CV
        starttime = datetime.datetime.now()
        score = cross_val_score(pipe, X_train, y_train, cv=10, scoring='f1_macro')
        endtime = datetime.datetime.now()
        print("10-fold Cross Validation time:\t" + str(endtime-starttime))
        score = score.mean()
        print("Macro F-score (Training):\t" + str(score))
        pipe.fit(X_train, y_train)
        # Test
        data_test = pd.read_csv('students_test.txt', sep='\t', header=None, names=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','edusupport','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G3'], dtype={'school':'category','sex':'category','address':'category','famsize':'category','Pstatus':'category','Mjob':'category','Fjob':'category','reason':'category','guardian':'category','edusupport':'category','nursery':'category','higher':'category','internet':'category','romantic':'category'})
        X_test = data_test.drop('Mjob', axis='columns')

        # Drop the same features here:
        #X_test = X_test.drop('school', axis='columns')
        X_test = X_test.drop('sex', axis='columns')
        X_test = X_test.drop('age', axis='columns')
        #X_test = X_test.drop('address', axis='columns')
        X_test = X_test.drop('famsize', axis='columns')
        #X_test = X_test.drop('Pstatus', axis='columns')
        #X_test = X_test.drop('Medu', axis='columns')
        #X_test = X_test.drop('Fedu', axis='columns')
        #X_test = X_test.drop('Mjob', axis='columns')
        #X_test = X_test.drop('Fjob', axis='columns')
        X_test = X_test.drop('reason', axis='columns')
        X_test = X_test.drop('guardian', axis='columns')
        X_test = X_test.drop('traveltime', axis='columns')
        X_test = X_test.drop('studytime', axis='columns')
        X_test = X_test.drop('failures', axis='columns')
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
        X_test = X_test.drop('absences', axis='columns')
        X_test = X_test.drop('G3', axis='columns')

        y_test = data_test.Mjob
        y_pred = pipe.predict(X_test)
        accuracy_macro = metrics.accuracy_score(y_test, y_pred)
        precision_macro = metrics.precision_score(y_test, y_pred, average='macro')
        recall_macro = metrics.recall_score(y_test, y_pred, average='macro')
        f1_macro = metrics.f1_score(y_test, y_pred, average='macro')

        # For each category, remap as binary and run training & testing again

        # teacher VS not teacher
        y_train_teacher = y_train.map({'teacher':'teacher', 'health':'not', 'services':'not', 'at_home':'not', 'other':'not'}).astype('category')
        y_test_teacher = y_test.map({'teacher':'teacher', 'health':'not', 'services':'not', 'at_home':'not', 'other':'not'}).astype('category')
        pipe.fit(X_train, y_train_teacher)
        y_pred_teacher = pipe.predict(X_test)
        precision_teacher = metrics.precision_score(y_test_teacher, y_pred_teacher, average='binary', pos_label='teacher', zero_division=1)
        recall_teacher = metrics.recall_score(y_test_teacher, y_pred_teacher, average='binary', pos_label='teacher', zero_division=1)
        f1_teacher = metrics.f1_score(y_test_teacher, y_pred_teacher, average='binary', pos_label='teacher', zero_division=1)

        # health VS not health
        y_train_health = y_train.map({'teacher':'not', 'health':'health', 'services':'not', 'at_home':'not', 'other':'not'}).astype('category')
        y_test_health = y_test.map({'teacher':'not', 'health':'health', 'services':'not', 'at_home':'not', 'other':'not'}).astype('category')
        pipe.fit(X_train, y_train_health)
        y_pred_health = pipe.predict(X_test)
        precision_health = metrics.precision_score(y_test_health, y_pred_health, average='binary', pos_label='health', zero_division=1)
        recall_health = metrics.recall_score(y_test_health, y_pred_health, average='binary', pos_label='health', zero_division=1)
        f1_health = metrics.f1_score(y_test_health, y_pred_health, average='binary', pos_label='health', zero_division=1)

        # services VS not services
        y_train_services = y_train.map({'teacher':'not', 'health':'not', 'services':'services', 'at_home':'not', 'other':'not'}).astype('category')
        y_test_services = y_test.map({'teacher':'not', 'health':'not', 'services':'services', 'at_home':'not', 'other':'not'}).astype('category')
        pipe.fit(X_train, y_train_services)
        y_pred_services = pipe.predict(X_test)
        precision_services = metrics.precision_score(y_test_services, y_pred_services, average='binary', pos_label='services', zero_division=1)
        recall_services = metrics.recall_score(y_test_services, y_pred_services, average='binary', pos_label='services', zero_division=1)
        f1_services = metrics.f1_score(y_test_services, y_pred_services, average='binary', pos_label='services', zero_division=1)

        # at_home VS not at_home
        y_train_at_home = y_train.map({'teacher':'not', 'health':'not', 'services':'not', 'at_home':'at_home', 'other':'not'}).astype('category')
        y_test_at_home = y_test.map({'teacher':'not', 'health':'not', 'services':'not', 'at_home':'at_home', 'other':'not'}).astype('category')
        pipe.fit(X_train, y_train_at_home)
        y_pred_at_home = pipe.predict(X_test)
        precision_at_home = metrics.precision_score(y_test_at_home, y_pred_at_home, average='binary', pos_label='at_home', zero_division=1)
        recall_at_home = metrics.recall_score(y_test_at_home, y_pred_at_home, average='binary', pos_label='at_home', zero_division=1)
        f1_at_home = metrics.f1_score(y_test_at_home, y_pred_at_home, average='binary', pos_label='at_home', zero_division=1)

        # other VS not other
        y_train_other = y_train.map({'teacher':'not', 'health':'not', 'services':'not', 'at_home':'not', 'other':'other'}).astype('category')
        y_test_other = y_test.map({'teacher':'not', 'health':'not', 'services':'not', 'at_home':'not', 'other':'other'}).astype('category')
        pipe.fit(X_train, y_train_other)
        y_pred_other = pipe.predict(X_test)
        precision_other = metrics.precision_score(y_test_other, y_pred_other, average='binary', pos_label='other', zero_division=1)
        recall_other = metrics.recall_score(y_test_other, y_pred_other, average='binary', pos_label='other', zero_division=1)
        f1_other = metrics.f1_score(y_test_other, y_pred_other, average='binary', pos_label='other', zero_division=1)

        # Evaluate learned model on testing data, and print the results.
        print("Testing results:")
        self.print_macro_results(accuracy_macro, precision_macro, recall_macro, f1_macro)
        print("Note: Some values below are 0.0 or 1.0, referring to undefined. Either TP+FP=0 or TP+FN=0 happened.")
        self.print_category_results("teacher", precision_teacher, recall_teacher, f1_teacher)
        self.print_category_results("health", precision_health, recall_health, f1_health)
        self.print_category_results("services", precision_services, recall_services, f1_services)
        self.print_category_results("at_home", precision_at_home, recall_at_home, f1_at_home)
        self.print_category_results("other", precision_other, recall_other, f1_other)
        return





    def model_2_run(self):
        print("--------------------\nModel 2:")
        # Train the model 2 with your best hyper parameters (if have) and features on training data.
        data_train = pd.read_csv('students_train.txt', sep='\t', header=None, names=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','edusupport','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G3'], dtype={'school':'category','sex':'category','address':'category','famsize':'category','Pstatus':'category','Mjob':'category','Fjob':'category','reason':'category','guardian':'category','edusupport':'category','nursery':'category','higher':'category','internet':'category','romantic':'category'})
        X_train = data_train.drop('Mjob', axis='columns')

        # Drop features of your choice
        #X_train = X_train.drop('school', axis='columns')
        X_train = X_train.drop('sex', axis='columns')
        X_train = X_train.drop('age', axis='columns')
        #X_train = X_train.drop('address', axis='columns')
        X_train = X_train.drop('famsize', axis='columns')
        #X_train = X_train.drop('Pstatus', axis='columns')
        #X_train = X_train.drop('Medu', axis='columns')
        #X_train = X_train.drop('Fedu', axis='columns')
        #X_train = X_train.drop('Mjob', axis='columns')
        #X_train = X_train.drop('Fjob', axis='columns')
        X_train = X_train.drop('reason', axis='columns')
        X_train = X_train.drop('guardian', axis='columns')
        X_train = X_train.drop('traveltime', axis='columns')
        X_train = X_train.drop('studytime', axis='columns')
        X_train = X_train.drop('failures', axis='columns')
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
        X_train = X_train.drop('absences', axis='columns')
        X_train = X_train.drop('G3', axis='columns')

        y_train = data_train.Mjob
        # Modify OneHotEncoder below as we drop features
        column_trans = make_column_transformer((OneHotEncoder(), ['school','address','Pstatus','Fjob']), remainder=StandardScaler())
        print("Training and testing set preprocessed by One Hot Encoding and Standard Scaling")
        # Using SVC
        print("Using Support Vector Classification")
        # Try out different params below:
        """ param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['rbf', 'poly', 'sigmoid']}
        grid = GridSearchCV(SVC(), param_grid, cv=10, scoring='accuracy')
        pipe = make_pipeline(column_trans, grid)
        pipe.fit(X_train,y_train)
        print(grid.best_estimator_) """
        print("Radial Basis Kernel + C=1 turn out to be the best")
        # Continue with the best tune
        svclf = SVC(C=1, kernel='poly')
        pipe = make_pipeline(column_trans, svclf)
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
        X_test = data_test.drop('Mjob', axis='columns')

        # Drop the same features here:
        #X_test = X_test.drop('school', axis='columns')
        X_test = X_test.drop('sex', axis='columns')
        X_test = X_test.drop('age', axis='columns')
        #X_test = X_test.drop('address', axis='columns')
        X_test = X_test.drop('famsize', axis='columns')
        #X_test = X_test.drop('Pstatus', axis='columns')
        #X_test = X_test.drop('Medu', axis='columns')
        #X_test = X_test.drop('Fedu', axis='columns')
        #X_test = X_test.drop('Mjob', axis='columns')
        #X_test = X_test.drop('Fjob', axis='columns')
        X_test = X_test.drop('reason', axis='columns')
        X_test = X_test.drop('guardian', axis='columns')
        X_test = X_test.drop('traveltime', axis='columns')
        X_test = X_test.drop('studytime', axis='columns')
        X_test = X_test.drop('failures', axis='columns')
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
        X_test = X_test.drop('absences', axis='columns')
        X_test = X_test.drop('G3', axis='columns')

        y_test = data_test.Mjob
        y_pred = pipe.predict(X_test)
        accuracy_macro = metrics.accuracy_score(y_test, y_pred)
        precision_macro = metrics.precision_score(y_test, y_pred, average='macro')
        recall_macro = metrics.recall_score(y_test, y_pred, average='macro')
        f1_macro = metrics.f1_score(y_test, y_pred, average='macro')

        # For each category, remap as binary and run training & testing again

        # teacher VS not teacher
        y_train_teacher = y_train.map({'teacher':'teacher', 'health':'not', 'services':'not', 'at_home':'not', 'other':'not'}).astype('category')
        y_test_teacher = y_test.map({'teacher':'teacher', 'health':'not', 'services':'not', 'at_home':'not', 'other':'not'}).astype('category')
        pipe.fit(X_train, y_train_teacher)
        y_pred_teacher = pipe.predict(X_test)
        precision_teacher = metrics.precision_score(y_test_teacher, y_pred_teacher, average='binary', pos_label='teacher', zero_division=1)
        recall_teacher = metrics.recall_score(y_test_teacher, y_pred_teacher, average='binary', pos_label='teacher', zero_division=1)
        f1_teacher = metrics.f1_score(y_test_teacher, y_pred_teacher, average='binary', pos_label='teacher', zero_division=1)

        # health VS not health
        y_train_health = y_train.map({'teacher':'not', 'health':'health', 'services':'not', 'at_home':'not', 'other':'not'}).astype('category')
        y_test_health = y_test.map({'teacher':'not', 'health':'health', 'services':'not', 'at_home':'not', 'other':'not'}).astype('category')
        pipe.fit(X_train, y_train_health)
        y_pred_health = pipe.predict(X_test)
        precision_health = metrics.precision_score(y_test_health, y_pred_health, average='binary', pos_label='health', zero_division=1)
        recall_health = metrics.recall_score(y_test_health, y_pred_health, average='binary', pos_label='health', zero_division=1)
        f1_health = metrics.f1_score(y_test_health, y_pred_health, average='binary', pos_label='health', zero_division=1)

        # services VS not services
        y_train_services = y_train.map({'teacher':'not', 'health':'not', 'services':'services', 'at_home':'not', 'other':'not'}).astype('category')
        y_test_services = y_test.map({'teacher':'not', 'health':'not', 'services':'services', 'at_home':'not', 'other':'not'}).astype('category')
        pipe.fit(X_train, y_train_services)
        y_pred_services = pipe.predict(X_test)
        precision_services = metrics.precision_score(y_test_services, y_pred_services, average='binary', pos_label='services', zero_division=1)
        recall_services = metrics.recall_score(y_test_services, y_pred_services, average='binary', pos_label='services', zero_division=1)
        f1_services = metrics.f1_score(y_test_services, y_pred_services, average='binary', pos_label='services', zero_division=1)

        # at_home VS not at_home
        y_train_at_home = y_train.map({'teacher':'not', 'health':'not', 'services':'not', 'at_home':'at_home', 'other':'not'}).astype('category')
        y_test_at_home = y_test.map({'teacher':'not', 'health':'not', 'services':'not', 'at_home':'at_home', 'other':'not'}).astype('category')
        pipe.fit(X_train, y_train_at_home)
        y_pred_at_home = pipe.predict(X_test)
        precision_at_home = metrics.precision_score(y_test_at_home, y_pred_at_home, average='binary', pos_label='at_home', zero_division=1)
        recall_at_home = metrics.recall_score(y_test_at_home, y_pred_at_home, average='binary', pos_label='at_home', zero_division=1)
        f1_at_home = metrics.f1_score(y_test_at_home, y_pred_at_home, average='binary', pos_label='at_home', zero_division=1)

        # other VS not other
        y_train_other = y_train.map({'teacher':'not', 'health':'not', 'services':'not', 'at_home':'not', 'other':'other'}).astype('category')
        y_test_other = y_test.map({'teacher':'not', 'health':'not', 'services':'not', 'at_home':'not', 'other':'other'}).astype('category')
        pipe.fit(X_train, y_train_other)
        y_pred_other = pipe.predict(X_test)
        precision_other = metrics.precision_score(y_test_other, y_pred_other, average='binary', pos_label='other', zero_division=1)
        recall_other = metrics.recall_score(y_test_other, y_pred_other, average='binary', pos_label='other', zero_division=1)
        f1_other = metrics.f1_score(y_test_other, y_pred_other, average='binary', pos_label='other', zero_division=1)

        # Evaluate learned model on testing data, and print the results.
        print("Testing results:")
        self.print_macro_results(accuracy_macro, precision_macro, recall_macro, f1_macro)
        print("Note: Some values below are 0.0 or 1.0, referring to undefined. Either TP+FP=0 or TP+FN=0 happened.")
        self.print_category_results("teacher", precision_teacher, recall_teacher, f1_teacher)
        self.print_category_results("health", precision_health, recall_health, f1_health)
        self.print_category_results("services", precision_services, recall_services, f1_services)
        self.print_category_results("at_home", precision_at_home, recall_at_home, f1_at_home)
        self.print_category_results("other", precision_other, recall_other, f1_other)
        return
