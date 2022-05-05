# Random Forest Regressor
This algorithm is a combination of each tree from the decision tree which is then combined into a single model.

Random Forest is an algorithm for classification. Then, how does it work? Random Forest works by building several decision trees and combining them to get more stable and accurate predictions. The 'forest' built by Random Forest is a collection of decision trees which are usually trained by the bagging method. The general idea of ​​the bagging method is a combination of learning models to improve overall results

The Random Forest algorithm increases the randomness of the model while growing the tree. Instead of looking for the most important feature when splitting a node, Random Forest looks for the best feature among a random subset of features. As a result, this method produces a wide variety and generally results in better models.

![Logo](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/rfc_vs_dt1.png)

# Dataset

in this project, we will predict house prices in Boston. The dataset consists of 333 rows and 14 columns.
and we have prepared a description of each column below:
![image](https://user-images.githubusercontent.com/86812576/166703666-67695e69-2f06-4563-95fe-7f63e32b4a5b.png)

crim    : city ​​crime rate per capita

zn      : percentage of total housing > 25,000 sqr.ft

indus   : percentage of non-retail business zones in the city

chas    : 1 = crossed by the river Charles, 0 = not crossed by the Charles river

nox     : NOx pollution concentration (parts per 10 million)

rm      : number of rooms (on average)

age     : percentage of houses built before 1940

dis     : average distance to 5 work districts in Boston (weigthed average)

rad     : accessibility index to radial highway

tax     : tax(per $10000)

ptratio : student-teacher ratio (by town)

black   : percentage of black people

lstat   : percentage of lower class population

medv    : median house price (Target)




# Import Package
import common package:

import **numpy as np**

import **pandas as pd**


from **sklearn.model_selection** import **train_test_split**

from **sklearn.pipeline** import **Pipeline**

from **sklearn.compose** import **ColumnTransformer**


from **jcopml.utils** import **save_model, load_model**

from **jcopml.pipeline** import **num_pipe, cat_pipe**

from **jcopml.plot** import **plot_missing_value**

from **jcopml.feature_importance** import **mean_score_decrease**

import Algorithm's Package:

from **sklearn.ensemble** import **RandomForestRegressor**

from **sklearn.model_selection** import **GridSearchCV**

from **jcopml.tuning** import **grid_search_params as gsp**

# Import Data

which i have explained before, the dataset has a column index called ID

# Mini Exploratory Data Analysis
I always work on data science projects with simple think so that I can benchmark. Using a simple model to benchmark. And most of the time it's more efficient and sometimes find a good one. but at the beginning I did mini Exploratory Data Analysis. **because i focus more on the algorithm**

We check whether there is any missing data or not. We can see that our data is clean. No features are removed, we will use them all and go straight to dataset splitting

# Dataset Splitting
split the data into X, and y

X = all columns except the target column.

y = 'medv' as target

test_size = 0.2 (which means 80% for train, and 20% for test)

# Training
In the Training step there are 3 main things that I specify.

First, the preprocessor: here the columns will be grouped into numeric and categoric.

included in the numeric column are: 'crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
                             'ptratio', 'black', 'lstat'.

and in the categoric column are: 'chas'. The interesting thing about random forest is that there is no obligation to use an encoder. for example if we have 1, 2, 3, 4, 5 then the decision will automatically encode. but if you use an encoder then it will help random forest so that the decision is not too heavy

second, pipeline: contains the preprocessor as 'prep' which I defined earlier, and the algorithm as 'algo' which in this case I use Random Forest Regressor.

and third, tuning with Grid Search: in this case I use the tuning recommendations (gsp.rf_params) that often occur in many cases. but does not rule out hyperparameter tuning if the model results are not good. with cross validation = 3.

**GridSearch Parameters Recommendation** :

**{'algo__n_estimators': [100, 150, 200],**

 **'algo__max_depth': [20, 50, 80],**
 
 **'algo__max_features': [0.3, 0.6, 0.8],**
 
 **'algo__min_samples_leaf': [1, 5, 10]}**
 

# Results dan Feature Importance
we can see that the score is 0.91 and this is very good, but in the Random Forest Algorithm there is a tendency to overfit because we know the Decision Tree is very easy to overfit even though we have averaged it there is still a chance for it.

![RF reg](https://user-images.githubusercontent.com/86812576/166943533-5df75fe8-1fcf-4044-80ab-294061e6165b.png)

**'algo__min_samples_leaf': 1** this will make the model overfit.

We can remodel with several different parameter tuning. Can we improve?

Scaling helps SVM and KNN but not for Random Forest, because if it is called, the position will be the same and have no effect. So scaling has no effect on tree base algorithms. But can do Feature Importance

### Feature Importance
**Mean Loss Decrease**
We'll look at the average error/loss reduction contributed by each feature. For the classification case, the loss used is Gini-impurity, so it is often called the mean impurity decrease. for the case of regression, the loss used is the Mean Square Error, it may be called the mean MSE decrese. The loss in question is the criterion. The greater the loss that a feature has succeeded in reducing, the more important that feature is.

![image](https://user-images.githubusercontent.com/86812576/166947310-24844cdc-5631-4d29-9b6d-bb3660ebabeb.png)


