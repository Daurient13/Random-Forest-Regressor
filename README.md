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
