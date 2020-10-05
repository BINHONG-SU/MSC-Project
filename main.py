import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from preprocessing import Preprocessing
Preprocessing = Preprocessing()
from models import models
models = models()
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, auc, roc_curve, normalized_mutual_info_score


data = pd.read_csv('./data.csv',index_col=0)
data.drop(['patkey', 'index_date', 'MATCHID'], axis=1)
data['age_at_index'] = data['age_at_index']-5
data = Preprocessing.FeatureEncoding(data)
data = Preprocessing.MissingData(data)
data.to_csv('data_complete.csv')
data = pd.read_csv('./data_complete.csv',index_col=0)
#==========================================================================================
#After using the KNN to deal with missing data, count and plot the histogram of features
'''
print(data.loc[:,'Smoking_status'].value_counts())
print(data.loc[:,'BMI_group'].value_counts())
print(data.loc[:,'Alcohol_status'].value_counts())
    
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%s' % int(height))

name_list = ['non-smoker','current somker','ex-smoker']
num_list = [45220,42332,12386]
num_list1 = [47307,42597,12402]
x =list(range(len(num_list)))
total_width, n = 0.8, 2
width = total_width / n
a = plt.bar(x, num_list, width=width, label='Orignal',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
b = plt.bar(x, num_list1, width=width, label='After Completion',tick_label = name_list,fc = 'r')
autolabel(a)
autolabel(b)
plt.ylabel('The number of people')
plt.legend()
plt.show()

name_list = ['BMI<18.5','18.5≤BMI<25','25≤BMI<30','30≤BMI<40','BMI≥40']
num_list = [2000,33065,36530,20031,1700]
num_list1 = [2000,38845,39647,20114,1700]
x =list(range(len(num_list)))
total_width, n = 0.9, 2
width = total_width / n
a = plt.bar(x, num_list, width=width, label='Orignal',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
b = plt.bar(x, num_list1, width=width, label='After Completion',tick_label = name_list,fc = 'r')
autolabel(a)
autolabel(b)
plt.ylabel('The number of people')
plt.legend()
plt.show()

name_list = ['non-drinker','current drinker','ex-drinker']
num_list = [14660,64852,11678]
num_list1 = [14660,75968,11678]
x =list(range(len(num_list)))
total_width, n = 0.8, 2
width = total_width / n
a = plt.bar(x, num_list, width=width, label='Orignal',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
b = plt.bar(x, num_list1, width=width, label='After Completion',tick_label = name_list,fc = 'r')
autolabel(a)
autolabel(b)
plt.ylabel('The number of people')
plt.legend()
plt.show()
'''
#============================================================================================
age5559_male = data[(data['age_at_index']>=55)&(data['age_at_index']<=59)&(data['gender']==0)]
age5559_female = data[(data['age_at_index']>=55)&(data['age_at_index']<=59)&(data['gender']==1)]
age6064_male = data[(data['age_at_index']>=60)&(data['age_at_index']<=64)&(data['gender']==0)]
age6064_female = data[(data['age_at_index']>=60)&(data['age_at_index']<=64)&(data['gender']==1)]
age6569_male = data[(data['age_at_index']>=65)&(data['age_at_index']<=69)&(data['gender']==0)]
age6569_female = data[(data['age_at_index']>=65)&(data['age_at_index']<=69)&(data['gender']==1)]
age7074_male = data[(data['age_at_index']>=70)&(data['age_at_index']<=74)&(data['gender']==0)]
age7074_female = data[(data['age_at_index']>=70)&(data['age_at_index']<=74)&(data['gender']==1)]
age7579_male = data[(data['age_at_index']>=75)&(data['age_at_index']<=79)&(data['gender']==0)]
age7579_female = data[(data['age_at_index']>=75)&(data['age_at_index']<=79)&(data['gender']==1)]
age8084_male = data[(data['age_at_index']>=80)&(data['age_at_index']<=84)&(data['gender']==0)]
age8084_female = data[(data['age_at_index']>=80)&(data['age_at_index']<=84)&(data['gender']==1)]
age8589_male = data[(data['age_at_index']>=85)&(data['age_at_index']<=89)&(data['gender']==0)]
age8589_female = data[(data['age_at_index']>=85)&(data['age_at_index']<=89)&(data['gender']==1)]
age9098_male = data[(data['age_at_index']>=90)&(data['age_at_index']<=98)&(data['gender']==0)]
age9098_female = data[(data['age_at_index']>=90)&(data['age_at_index']<=98)&(data['gender']==1)]
groups = [age5559_male,age5559_female,age6064_male,age6064_female,age6569_male,age6569_female,
          age7074_male,age7074_female,age7579_male,age8084_male,age8084_female,age8589_male,
          age8589_female,age9098_male,age9098_female]
groups_name = ['age5559_male','age5559_female','age6064_male','age6064_female',
                'age6569_male','age6569_female','age7074_male','age7074_female',
                'age7579_male','age8084_male','age8084_female','age8589_male',
                'age8589_female','age9098_male','age9098_female']
#===================================================================================================================
#Develop different traditional machine learning models for each group based on different number of retained features
count = 0
for group in groups:
    X = group.drop(['dementia', 'age_at_index', 'gender'], axis=1)
    Y = group['dementia']
    ACC_LR_list = []
    ACC_SVM_list = []
    ACC_RF_list = []
    ACC_KNN_list = []
    ACC_NB_list = []
    for num_features in range(32, 0, -1):
        print('Group:', groups_name[count])
        X_Select = Preprocessing.FeatureSelection_MIFS(X, Y, num_features)
        X_train, X_test, Y_train, Y_test = train_test_split(X_Select, Y, test_size=0.2, random_state=0)
        ACC_LR = models.LR(X_train, Y_train, X_test, Y_test)
        ACC_SVM = models.SVM(X_train, Y_train, X_test, Y_test)
        ACC_RF = models.RF(X_train, Y_train, X_test, Y_test)
        ACC_KNN = models.KNN(X_train, Y_train, X_test, Y_test)
        ACC_NB = models.NB(X_train, Y_train, X_test, Y_test)
        ACC_LR_list.append(ACC_LR)
        ACC_SVM_list.append(ACC_SVM)
        ACC_RF_list.append(ACC_RF)
        ACC_KNN_list.append(ACC_KNN)
        ACC_NB_list.append(ACC_NB)
        print()
    x = np.arange(32,0,-1)
    plt.figure(count+1)
    plt.plot(x, np.array(ACC_LR_list), color='r', label='Classifier: LR')
    plt.plot(x, np.array(ACC_SVM_list), color='y', label='Classifier: SVM')
    plt.plot(x, np.array(ACC_RF_list), color='g', label='Classifier: RF')
    plt.plot(x, np.array(ACC_KNN_list), color='b', label='Classifier: KNN')
    plt.plot(x, np.array(ACC_NB_list), color='k', label='Classifier: NB')
    plt.xlabel('Number of features')
    plt.ylabel('Prediction Accuracy')
    plt.legend(loc='lower right')
    plt.title('Group: %s, Feature Selection: MIFS'%groups_name[count])
    count += 1
plt.show()
#===================================================================================================================
#Get results of predictive models based on male samples aged from 55 to 59 (study case)
'''
data = age5559_male
X = data.drop(['dementia', 'age_at_index', 'gender'], axis=1)
Y = data['dementia']
X_Select = Preprocessing.FeatureSelection_MIFS(X, Y, 31)
X_train, X_test, Y_train, Y_test = train_test_split(X_Select, Y, test_size=0.2, random_state=0)
models.LR(X_train, Y_train, X_test, Y_test)
models.SVM(X_train, Y_train, X_test, Y_test)
models.RF(X_train, Y_train, X_test, Y_test)
models.KNN(X_train, Y_train, X_test, Y_test)
models.NB(X_train, Y_train, X_test, Y_test)
'''
#============================================================================================
#Validate the effectiveness of KNN (used for dealing with missing values) #Here, continue to drop features: 'BMI_group','Smoking_status','Alcohol_status'
'''
data = age5559_male
X = data.drop(['dementia', 'age_at_index', 'gender'], axis=1)
Y = data['dementia']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print('with KNN')
models.LR(X_train, Y_train, X_test, Y_test)
models.SVM(X_train, Y_train, X_test, Y_test)
models.RF(X_train, Y_train, X_test, Y_test)
models.KNN(X_train, Y_train, X_test, Y_test)
models.NB(X_train, Y_train, X_test, Y_test)
models.NN_age5559_male(X_train, Y_train, X_test, Y_test)

X = data.drop(['dementia', 'age_at_index', 'gender','BMI_group','Smoking_status','Alcohol_status'], axis=1)
Y = data['dementia']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print('without KNN')
models.LR(X_train, Y_train, X_test, Y_test)
models.SVM(X_train, Y_train, X_test, Y_test)
models.RF(X_train, Y_train, X_test, Y_test)
models.KNN(X_train, Y_train, X_test, Y_test)
models.NB(X_train, Y_train, X_test, Y_test)
models.NN_age5559_male(X_train, Y_train, X_test, Y_test) #It needs to modify the number of input_dim in this function
'''
#============================================================================================
#Develop neural networks for each group
'''
data = age5559_male
X = data.drop(['dementia', 'age_at_index', 'gender'], axis=1)
Y = data['dementia']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
models.NN_age5559_male(X_train, Y_train, X_test, Y_test)
'''

'''
data = age5559_female
X = data.drop(['dementia', 'age_at_index', 'gender'], axis=1)
Y = data['dementia']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
models.NN_age5559_female(X_train, Y_train, X_test, Y_test)
'''

'''
data = age6064_male
X = data.drop(['dementia', 'age_at_index', 'gender'], axis=1)
Y = data['dementia']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
models.NN_age6064_male(X_train, Y_train, X_test, Y_test)
'''

'''
data = age6064_female
X = data.drop(['dementia', 'age_at_index', 'gender'], axis=1)
Y = data['dementia']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
models.NN_age6064_female(X_train, Y_train, X_test, Y_test)
'''

'''
data = age6569_male
X = data.drop(['dementia', 'age_at_index', 'gender'], axis=1)
Y = data['dementia']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
models.NN_age6569_male(X_train, Y_train, X_test, Y_test)
'''

'''
data = age6569_female
X = data.drop(['dementia', 'age_at_index', 'gender'], axis=1)
Y = data['dementia']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
models.NN_age6569_female(X_train, Y_train, X_test, Y_test)
'''

'''
data = age7074_male
X = data.drop(['dementia', 'age_at_index', 'gender'], axis=1)
Y = data['dementia']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
models.NN_age7074_male(X_train, Y_train, X_test, Y_test)
'''

'''
data = age7074_female
X = data.drop(['dementia', 'age_at_index', 'gender'], axis=1)
Y = data['dementia']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
models.NN_age7074_female(X_train, Y_train, X_test, Y_test)
'''

'''
data = age7579_male
X = data.drop(['dementia', 'age_at_index', 'gender'], axis=1)
Y = data['dementia']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
models.NN_age7579_male(X_train, Y_train, X_test, Y_test)
'''

'''
data = age7579_female
X = data.drop(['dementia', 'age_at_index', 'gender'], axis=1)
Y = data['dementia']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
models.NN_age7579_female(X_train, Y_train, X_test, Y_test)
'''

'''
data = age8084_male
X = data.drop(['dementia', 'age_at_index', 'gender'], axis=1)
Y = data['dementia']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
models.NN_age8084_male(X_train, Y_train, X_test, Y_test)
'''

'''
data = age8084_female
X = data.drop(['dementia', 'age_at_index', 'gender'], axis=1)
Y = data['dementia']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
models.NN_age8084_female(X_train, Y_train, X_test, Y_test)
'''

'''
data = age8589_male
X = data.drop(['dementia', 'age_at_index', 'gender'], axis=1)
Y = data['dementia']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
models.NN_age8589_male(X_train, Y_train, X_test, Y_test)
'''

'''
data = age8589_female
X = data.drop(['dementia', 'age_at_index', 'gender'], axis=1)
Y = data['dementia']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
models.NN_age8589_female(X_train, Y_train, X_test, Y_test)
'''

'''
data = age9098_male
X = data.drop(['dementia', 'age_at_index', 'gender'], axis=1)
Y = data['dementia']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
models.NN_age9098_male(X_train, Y_train, X_test, Y_test)
'''

'''
data = age9098_female
X = data.drop(['dementia', 'age_at_index', 'gender'], axis=1)
Y = data['dementia']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
models.NN_age9098_female(X_train, Y_train, X_test, Y_test)
'''