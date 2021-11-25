import numpy as np
import pandas as pd
import sklearn.datasets
breast_cancer= sklearn.datasets.load_breast_cancer()
print(breast_cancer)
X=breast_cancer.data
Y=breast_cancer.target
print(X.shape, Y.shape)
data=pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
data['class'] = breast_cancer.target
data.head()
data.describe()
print(data['class'].value_counts())
print(breast_cancer.target_names)
data.groupby('class').mean()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1,stratify=Y,random_state=1)
print(Y.mean(), Y_train.mean(), Y_test.mean())
print(X_train.mean(), X_test.mean(), X.mean())
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)
from sklearn.metrics import accuracy_score
prediction_on_training_data = classifier.predict(X_train)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy = ', accuracy_on_training_data)
prediction_on_test_data = classifier.predict(X_test)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy = ', accuracy_on_test_data)
input_data=(13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)
input_data_as_numpy_array = np.asarray(input_data)
print(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=classifier.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
    print('malignant')
else:
    print('benign')

