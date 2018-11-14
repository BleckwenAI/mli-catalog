import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from IPython.display import display


class Bikes:
    def __init__(self):
        self.PATH = "datasets/bike_sharing.csv"
        self.data = pd.read_csv(self.PATH)
        self.numerical = ['year', 'holiday', 'working_day', 'temperature', 
                          'felt_temperature', 'humidity', 'wind_speed', 'days_since_2011', 'count']
        self.categorical = ['month', 'hour', 'season', 'week_day', 'weather_situation']
        for c in self.numerical:
            self.data[c] = self.data[c].astype(np.float64)
        
        self.data_dummies = pd.get_dummies(self.data, 
                                           columns=self.categorical, 
                                           prefix_sep="=",
                                           drop_first=False)

        self.X = self.data.drop(['count'], axis = 1)
        self.X_d = self.data_dummies.drop(['count'], axis = 1)
        self.y = self.data['count'].values
        
    # Functions for loading datasets with pandas and splitting them into train and test
    def load(self,dummies=True):
        X = self.X
        if dummies:
            X = self.X_d
        X_train, X_test, y_train, y_test = self.train_test_split(X, self.y)
        return X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
    
    # A little preview of what the dataset looks like
    def preview(self):
        print(f"This dataset contains {self.data.shape[0]} instances with {self.data.shape[1]} attributes")
        display(self.data.sample(5))
    
    def feature_names(self,dummies=True):
        if dummies:
            return list(self.X_d.columns)
        return list(self.X.columns)
    
    def train_test_split(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def is_numerical(self, column_name):
        return column_name in set(self.numerical)
    
    
class Loans:
    def __init__(self):
        self.PATH = "datasets/loans.csv"
        self.data = pd.read_csv(self.PATH)
        self.categorical = ['home_ownership', 'verification_status', 'purpose']
        self.numerical = ['loan_amnt', 'int_rate', 'installment', 'grade', 
                          'emp_length', 'annual_inc', 'dti', 'delinq_2yrs',
                          'inq_last_6mths', 'open_acc', 'revol_bal', 'revol_util',
                          'open_acc_6m', 'inq_fi', 'mths_since_crl_was_opened']
        for c in self.categorical:
            self.data[c] = self.data[c].astype('category')
        
        self.data_dummies = pd.get_dummies(self.data, 
                                           columns=self.categorical, 
                                           prefix_sep="=",
                                           drop_first=False)

        self.X = self.data.drop(['bad_loan_status'], axis=1)
        self.y = self.data['bad_loan_status'].values
        self.X_d = self.data_dummies.drop(['bad_loan_status'], axis=1)
    
    def load(self, dummies=True):
        X = self.X
        if dummies:
            X = self.X_d
        X_train, X_test, y_train, y_test = self.train_test_split(X, self.y)
        return X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
    
    def preview(self):
        print(f"This dataset contains {self.data.shape[0]} instances with {self.data.shape[1]} attributes")
        display(self.data.sample(5)) 
    
    def feature_names(self,dummies=True):
        if dummies:
            return list(self.X_d.columns)
        return list(self.X.columns)
    
    def train_test_split(self, X, y):
        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=1024)
    
    def is_numerical(self, column_name):
        return column_name in set(self.numerical)
