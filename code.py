# Importing libraries

import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
import seaborn as sns

import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
df = pd.read_csv(url, names=names)

df.keys()
Output: Index(['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class'],
      dtype='object')
      
df['class']
Output: 0      2
1      2
2      2
3      2
4      2
      ..
694    2
695    2
696    4
697    4
698    4
Name: class, Length: 699, dtype: int64
                     
# Preprocess the data

df.replace('?',-99999, inplace=True)
print(df.axes)

df.drop(['id'], 1, inplace=True)    

Output: [RangeIndex(start=0, stop=699, step=1), Index(['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class'],
      dtype='object')]

# Let explore the dataset and do a few visualizations
print(df.loc[10])

# Print the shape of the dataset
print(df.shape)

Output: clump_thickness           1
uniform_cell_size         1
uniform_cell_shape        1
marginal_adhesion         1
single_epithelial_size    1
bare_nuclei               1
bland_chromatin           3
normal_nucleoli           1
mitoses                   1
class                     2
Name: 10, dtype: object
(699, 10)

# Describe the dataset

print(df.describe())

Output:     clump_thickness  uniform_cell_size  uniform_cell_shape  \
count       699.000000         699.000000          699.000000   
mean          4.417740           3.134478            3.207439   
std           2.815741           3.051459            2.971913   
min           1.000000           1.000000            1.000000   
25%           2.000000           1.000000            1.000000   
50%           4.000000           1.000000            1.000000   
75%           6.000000           5.000000            5.000000   
max          10.000000          10.000000           10.000000   

       marginal_adhesion  single_epithelial_size  bland_chromatin  \
count         699.000000              699.000000       699.000000   
mean            2.806867                3.216023         3.437768   
std             2.855379                2.214300         2.438364   
min             1.000000                1.000000         1.000000   
25%             1.000000                2.000000         2.000000   
50%             1.000000                2.000000         3.000000   
75%             4.000000                4.000000         5.000000   
max            10.000000               10.000000        10.000000   

       normal_nucleoli     mitoses       class  
count       699.000000  699.000000  699.000000  
mean          2.866953    1.589413    2.689557  
std           3.053634    1.715078    0.951273  
min           1.000000    1.000000    2.000000  
25%           1.000000    1.000000    2.000000  
50%           1.000000    1.000000    2.000000  
75%           4.000000    1.000000    4.000000  
max          10.000000   10.000000    4.000000 

df.info()

Output: <class 'pandas.core.frame.DataFrame'>
RangeIndex: 699 entries, 0 to 698
Data columns (total 10 columns):
 #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   clump_thickness         699 non-null    int64 
 1   uniform_cell_size       699 non-null    int64 
 2   uniform_cell_shape      699 non-null    int64 
 3   marginal_adhesion       699 non-null    int64 
 4   single_epithelial_size  699 non-null    int64 
 5   bare_nuclei             699 non-null    object
 6   bland_chromatin         699 non-null    int64 
 7   normal_nucleoli         699 non-null    int64 
 8   mitoses                 699 non-null    int64 
 9   class                   699 non-null    int64 
dtypes: int64(9), object(1)
memory usage: 54.7+ KB

# Plot histograms for each variable
df.hist(figsize = (10, 9))
plt.show()

Output: Added in a different file

# Create scatter plot matrix
scatter_matrix(df, figsize = (18,18))
plt.show()

Output: Added in a different file
       
sns.countplot(df['class'])

Result: 
       
df.describe()

Output:	clump_thickness	uniform_cell_size	uniform_cell_shape	marginal_adhesion	single_epithelial_size	bland_chromatin	normal_nucleoli	mitoses	class
count	        699.000000	         699.000000	        699.000000	        699.000000	          699.000000              	   699.000000	         699.000000	      699.000000	699.000000
mean	        4.417740	         3.134478	        3.207439	        2.806867	          3.216023	                 3.437768	         2.866953	      1.589413	2.689557
std	        2.815741	         3.051459	        2.971913	        2.855379	          2.214300	                 2.438364	         3.053634	      1.715078	0.951273
min	        1.000000	         1.000000	        1.000000	        1.000000	          1.000000	                 1.000000	         1.000000	      1.000000	2.000000
25%	        2.000000	         1.000000         	 1.000000	        1.000000	          2.000000	                 2.000000	         1.000000	      1.000000	2.000000
50%	        4.000000	         1.000000	        1.000000	        1.000000	          2.000000	                 3.000000	         1.000000	      1.000000	2.000000
75%	        6.000000	         5.000000	        5.000000	        4.000000	          4.000000	                 5.000000	         4.000000	      1.000000	4.000000
max	        10.000000	         10.000000	        10.000000	        10.000000	          10.000000	                 10.000000	         10.000000	      10.000000	4.000000
       
# Splitting the dataset into independent (X) and dependent (Y) data sets

X = df.iloc[:,2:31].values
Y = df.iloc[:,1].values       
                     
# Splitting the dataset into 75% training and 25% testing

X = df.iloc[:,2:31].values
Y = df.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train

Output: array([[ 0.26790687,  0.39501803,  2.1667716 , ..., -0.29552173,
        -0.33390732,  1.39420953],
       [-0.39584348, -0.63307525, -0.55816036, ..., -0.61546674,
        -0.33390732, -0.7172523 ],
       [ 2.2591579 ,  2.45120457,  1.71261627, ...,  1.62414835,
         3.28610862,  1.39420953],
       ...,
       [-0.72771865, -0.63307525, -0.55816036, ..., -0.61546674,
        -0.33390732, -0.7172523 ],
       [-0.72771865, -0.63307525, -0.55816036, ..., -0.61546674,
        -0.33390732, -0.7172523 ],
       [-0.72771865, -0.63307525, -0.55816036, ..., -0.61546674,
        -0.33390732, -0.7172523 ]])
       
# Define models to train

def models(X_train,Y_train):
    
    #Using KNeighborsClassifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    knn.fit(X_train, Y_train)
    
    #Using SVC linear
    from sklearn.svm import SVC
    svc_lin = SVC(kernel = 'linear', random_state = 0)
    svc_lin.fit(X_train, Y_train)
    
    #Using DecisionTreeClassifier 
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, Y_train)


    
 
    print('[0] K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
    print('[1] Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
    print('[2] Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
    
    return knn,svc_lin,tree

# Getting all the models
model = models(X_train, Y_train)

Output: [0] K Nearest Neighbor Training Accuracy: 0.7480916030534351
[1] Support Vector Machine (Linear Classifier) Training Accuracy: 0.75
[2] Decision Tree Classifier Training Accuracy: 0.9675572519083969
       
#Show other ways to get the classification accuracy & other metrics 

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model)):
    print('Model',i)
    print( classification_report(Y_test, model[i].predict(X_test)))
    print( accuracy_score(Y_test, model[i].predict(X_test)))
    print()
       
Output: Model 0
              precision    recall  f1-score   support

           1       0.77      0.98      0.86        88
           2       0.00      0.00      0.00        15
           3       0.07      0.07      0.07        14
           4       0.20      0.17      0.18        12
           5       0.09      0.10      0.10        10
           6       0.00      0.00      0.00         7
           7       0.00      0.00      0.00         6
           8       0.00      0.00      0.00        10
           9       0.00      0.00      0.00         1
          10       0.45      0.83      0.59        12

    accuracy                           0.57       175
   macro avg       0.16      0.21      0.18       175
weighted avg       0.45      0.57      0.50       175

0.5714285714285714

Model 1
              precision    recall  f1-score   support

           1       0.79      0.97      0.87        88
           2       0.25      0.07      0.11        15
           3       0.14      0.07      0.10        14
           4       0.21      0.33      0.26        12
           5       0.33      0.10      0.15        10
           6       0.33      0.14      0.20         7
           7       0.00      0.00      0.00         6
           8       0.29      0.20      0.24        10
           9       0.00      0.00      0.00         1
          10       0.39      0.75      0.51        12

    accuracy                           0.59       175
   macro avg       0.27      0.26      0.24       175
weighted avg       0.52      0.59      0.54       175

0.5942857142857143

Model 2
              precision    recall  f1-score   support

           1       0.84      0.94      0.89        88
           2       0.14      0.07      0.09        15
           3       0.09      0.07      0.08        14
           4       0.14      0.17      0.15        12
           5       0.00      0.00      0.00        10
           6       0.33      0.29      0.31         7
           7       0.20      0.17      0.18         6
           8       0.33      0.50      0.40        10
           9       0.14      1.00      0.25         1
          10       0.80      0.67      0.73        12

    accuracy                           0.59       175
   macro avg       0.30      0.39      0.31       175
weighted avg       0.55      0.59      0.56       175

0.5942857142857143       

       
