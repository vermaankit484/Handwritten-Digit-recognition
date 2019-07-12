import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as ts
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pathlib
import pickle
from sklearn.metrics import confusion_matrix

df = pd.read_csv("mnist_train.csv", sep = ',')

#Provides the shape of the dataset
print(df.shape)

"""#to check distribution of labels
print(df['label'].value_counts())
sns.countplot(df['label'])
plt.show()"""

x_train = df.iloc[:, 1:] #Contains all the pixel values
y_train = df.iloc[:, 0] #Contains the labels correspoding to each digit


#Used to standardize feautres
sc = StandardScaler()
x_train = sc.fit_transform(x_train)

#print(x_train)

#We won't split our data into test data and train data as we have separate data for training and testing


#Pickle library is used to save trained model
path = pathlib.Path('finalized_model.sav')
if path.exists() == True:
    filename = 'finalized_model.sav'
    g_s = pickle.load(open(filename, 'rb'))
else:
    #Creating classifier
    clf = svm.SVC()
    parameters = [{'C' : [0.001, 0.1, 1, 10, 1000], 'kernel' : ['poly'], 'gamma' : [0.1, 0.5, 1, 5, 10]}]
    g_s = GridSearchCV(estimator = clf, param_grid = parameters, scoring = 'accuracy', cv = 5)
    g_s = g_s.fit(x_train, y_train)
    filename = 'finalized_model.sav'
    pickle.dump(g_s, open(filename, 'wb'))
print(g_s.best_score_*100)

c = g_s.best_params_['C']
gmma = g_s.best_params_['gamma']

"""
 for i in (np.random.randint(0,99,2)):
 two_d = (np.reshape(x_train.values[i], (28, 28)) * 255).astype(np.uint8)
 plt.title('predicted label: {0}'. format(y_train[i]))
 plt.imshow(two_d, interpolation='nearest', cmap='gray')
 plt.show()"""

path2 = pathlib.Path('model2.sav')
if path2 == True:
    fn = 'model2.sav'
    clf2 = pickle.load(open(fn, 'rb'))
else:
    clf2 = svm.SVC(kernel = 'poly', C = c, gamma = gmma)
    clf2.fit(x_train, y_train)
    fn = 'model2.sav'
    pickle.dump(clf2,open(fn, 'wb'))

#Dataset to test our model
df2 = pd.read_csv("mnist_test.csv", sep = ',')
train = df2.iloc[:, 1:]
test = df2.iloc[:, 0]
train = sc.fit_transform(train)
pred = clf2.predict(train)
print(metrics.accuracy_score(pred, test))

#Confusion Matrix is such that C(i, j) is equal to the number of observations in group j which were supposed to be in group i
print("Confusion Matrix\n", confusion_matrix(pred, test))
