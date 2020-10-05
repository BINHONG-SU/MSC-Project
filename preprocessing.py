import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
#############################################################################
class Preprocessing:

    def __init__(self):
        return None

    def FeatureEncoding(self, data):
        gender_dic = {'F':1,'M':0}
        data['gender'] = data['gender'].map(gender_dic)

        smoking_dic = {'N':1,'Y':2,'D':3}
        data['Smoking_status'] = data['Smoking_status'].map(smoking_dic)
        alcohol_dic = {'N':1,'Y':2,'D':3}
        data['Alcohol_status'] = data['Alcohol_status'].map(alcohol_dic)
        return data

    def MissingData(self, data):
        Base = data.drop(['dementia','Smoking_status', 'Alcohol_status', 'BMI_group', 'SBP_group', 'LDL_group', 'HbA1c_group'],axis=1)

        Base['Smoking_status'] = data['Smoking_status']
        Train_Data = Base.dropna()
        Train_X = np.array(Train_Data.drop(['Smoking_status'], axis=1))
        Train_Y = Train_Data['Smoking_status'].values
        tuned_parameters = {'n_neighbors': [50, 100, 150, 200, 300, 400]}
        model = KNeighborsClassifier()
        grid = GridSearchCV(model, tuned_parameters, cv=3)
        grid = grid.fit(Train_X, Train_Y)
        print('Smoking_status_KNN', grid.best_estimator_)
        Base = Base.fillna('NaN')
        for index, row in Base.iterrows():
            if row['Smoking_status'] == 'NaN':
                Predict_X = np.array(row.drop(['Smoking_status']))
                Predict_Y = grid.predict(Predict_X.reshape(1, -1))
                Base['Smoking_status'][index] = Predict_Y[0]

        Base['BMI_group'] = data['BMI_group']
        Train_Data = Base.dropna()
        Train_X = np.array(Train_Data.drop(['BMI_group'], axis=1))
        Train_Y = Train_Data['BMI_group'].values
        tuned_parameters = {'n_neighbors': [50, 100, 150, 200, 300, 400]}
        model = KNeighborsClassifier()
        grid = GridSearchCV(model, tuned_parameters, cv=3)
        grid = grid.fit(Train_X, Train_Y)
        print('BMI_group', grid.best_estimator_)
        Base = Base.fillna('NaN')
        for index, row in Base.iterrows():
            if row['BMI_group'] == 'NaN':
                Predict_X = np.array(row.drop(['BMI_group']))
                Predict_Y = grid.predict(Predict_X.reshape(1, -1))
                Base['BMI_group'][index] = Predict_Y[0]

        Base['Alcohol_status'] = data['Alcohol_status']
        Train_Data = Base.dropna()
        Train_X = np.array(Train_Data.drop(['Alcohol_status'], axis=1))
        Train_Y = Train_Data['Alcohol_status'].values
        tuned_parameters = {'n_neighbors': [50, 100, 150, 200, 300, 400]}
        model = KNeighborsClassifier()
        grid = GridSearchCV(model, tuned_parameters, cv=3)
        grid = grid.fit(Train_X, Train_Y)
        print('Alcohol_status', grid.best_estimator_)
        Base = Base.fillna('NaN')
        for index, row in Base.iterrows():
            if row['Alcohol_status'] == 'NaN':
                Predict_X = np.array(row.drop(['Alcohol_status']))
                Predict_Y = grid.predict(Predict_X.reshape(1, -1))
                Base['Alcohol_status'][index] = Predict_Y[0]

        Base['dementia'] = data['dementia']
        return Base

    def FeatureSelection_MIFS(self, X, Y,num_features):
        headers = X.columns
        F = X
        MI = []
        for index, col in F.iteritems():
            a = normalized_mutual_info_score(col, Y)
            MI.append(a)
        MI = np.array(MI)
        features_df = pd.DataFrame(MI, index=headers)
        features_df = features_df.sort_values(by=0, ascending=False)
        feature_1 = features_df[0:1]
        S = F[list(feature_1.index)]
        F = F.drop(labels=list(feature_1.index), axis=1)

        for i in range(X.shape[1] - 1):
            MI = []
            headers = F.columns
            for index, col in F.iteritems():
                a = normalized_mutual_info_score(col, Y)
                b = 0
                for index2, col2 in S.iteritems():
                    b = b + normalized_mutual_info_score(col2, col)
                a = a - 0.5 * b
                MI.append(a)
            MI = np.array(MI)
            features_df = pd.DataFrame(MI, index=headers)
            features_df = features_df.sort_values(by=0, ascending=False)
            feature_1 = features_df[0:1]
            S = pd.concat([S, F[list(feature_1.index)]], axis=1)
            F = F.drop(labels=list(feature_1.index), axis=1)
        Order = S.columns.values.tolist()
        #for i in Order:
            #print(i)
        X = X.drop(labels = Order[num_features:32],axis=1)
        if 'Smoking_status' in X.columns:
            smoking_dic = {1: 'non_smoker', 2: 'current_smoker', 3: 'ex_smoker'}
            X['Smoking_status'] = X['Smoking_status'].map(smoking_dic)
            dummy_smoking = pd.get_dummies(X['Smoking_status'])
            X = pd.concat([X, dummy_smoking], axis=1)
            X = X.drop(['Smoking_status'], axis=1)

        if 'Alcohol_status' in X.columns:
            alcohol_dic = {1: 'non_drinker', 2: 'current_drinker', 3: 'ex_drinker'}
            X['Alcohol_status'] = X['Alcohol_status'].map(alcohol_dic)
            dummy_alcohol = pd.get_dummies(X['Alcohol_status'])
            X = pd.concat([X, dummy_alcohol], axis=1)
            X = X.drop(['Alcohol_status'], axis=1)
        print('Feature Selection_MIFS: %d Features' %num_features)
        return X

    def FeatureSelection_MI(self, X, Y):
        headers = X.columns
        MI = []
        for index, col in X.iteritems():
            a = normalized_mutual_info_score(col, Y)
            MI.append(a)
        MI = np.array(MI)
        features_df = pd.DataFrame(MI, index=headers)
        features_df = features_df.sort_values(by=0, ascending=False)
        print(features_df)
        return X

    def FeatureSelection_RF(self, X, Y,num_features):
        tuned_parameters = {'n_estimators': [10, 20, 50, 80, 100, 200,400]}
        model = RandomForestClassifier()
        grid = GridSearchCV(model, tuned_parameters, cv=3)
        grid = grid.fit(X, Y)
        feature_importance = grid.best_estimator_.feature_importances_
        features = X.columns
        features_df = pd.DataFrame(feature_importance, index=features)
        features_df = features_df.sort_values(by=0, ascending=False)
        #print(features_df)
        X = X.drop(labels=list(features_df[num_features:32].index), axis=1)
        if 'Smoking_status' in X.columns:
            smoking_dic = {1: 'non_smoker', 2: 'current_smoker', 3: 'ex_smoker'}
            X['Smoking_status'] = X['Smoking_status'].map(smoking_dic)
            dummy_smoking = pd.get_dummies(X['Smoking_status'])
            X = pd.concat([X, dummy_smoking], axis=1)
            X = X.drop(['Smoking_status'], axis=1)

        if 'Alcohol_status' in X.columns:
            alcohol_dic = {1: 'non_drinker', 2: 'current_drinker', 3: 'ex_drinker'}
            X['Alcohol_status'] = X['Alcohol_status'].map(alcohol_dic)
            dummy_alcohol = pd.get_dummies(X['Alcohol_status'])
            X = pd.concat([X, dummy_alcohol], axis=1)
            X = X.drop(['Alcohol_status'], axis=1)
        print('Feature Selection_RF: %d Features' %num_features)
        return X
#################################################################################