https://github.com/2004velvizhi/ibm.git
 Import pandas as pd
Import numpy as np
From sklearn.model_selection import train_test_split
From sklearn.linear_model import LLinearRegression
Data_pd.read_csv(https://raw.githubusercontent.com/amankharwal/Website-
data/master/advertising.csv)
Print(data.head())
 TV           Radio   Newspaper  Sales
0 230.1     37.8        69.2             22.1
1 44.5.      39.3.       45.1             10.4
3 151.5     41.3        58.5              16.5
4 180.8.    10.8       58.4                  17.9
4 180.8.    10.8       58.4                  17.9
Our goal is to predict monthly sales, so we will first consolidate all stores and days into total monthly sales. 
Dataset = pd.read_csv('/input/demand_forecasting_kernals_only/sample_submission.csv’)
Df= dataset.copy()	
Df.head()
Id     sales
0          52
1           52
2           52
3           52
4            52
Def load_data(file_name):
"""Returns a pandas dataframe from a csv file.""""
Return pd.read_csv(file_name)
Sales_data = load_data('../input/demand-forecasting-kernels-only/train.csv')
Df_s.sales_data.copy(
Df_s.info()
<class 'pandas.core.frame.DataFrame">
RangeIndex: 913000 entries, 0 to 912999
Data columns (total 4 columns):   
#   Column  Non-Null Count   Dtype  
---  ------  --------------   -----  
date    913000 non-null  object 
store   913000 non-null  int64  
item    913000 non-null  int64  
sales   913000 non-null  int64  
Dtypes: int64(3), object(1) 
Memory usage: 27.9+ MB 
Df_s.tail()
Date 	                         store 	item  sales
912995 2017-12-27   10       50      63 
912996 2017-12-28   10.      50.      59
 912997 2017-12-29   10       50       74 
912998 2017-12-30   10       50        62 
912999 2017-12-31    10      50         82 
# To view basic statistical details about dataset: 
 Df_s[‘sales’].describe() 
Count    913000.000000 
Mean         52.250287 
Std          28.801144 
Min           0.000000 
25%          30.000000 
50%          47.000000 
75%          70.000000 
Max         231.000000 
Name: sales, dtype: float64 
   Sales seem to be unbalanced! 
Df_s[‘sales’].plot() 
import numpy as np
from sklearn.linear_model import LinearRegression
# Sample dataset
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Input features
y = np.array([2, 4, 5, 4, 5])  # Target values
# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)
# Make predictions
predictions = model.predict(X)
# Print the model parameters
print("Coefficients:", model.coef_)  # Slope of the line
print("Intercept:", model.intercept_)  # Intercept of the line
```
This code creates a linear regression model, fits it to the dataset, and makes predictions. 
Random forest:
Training a model like a Random Forest involves using a dataset to build an ensemble of decision trees. Here's a simplified example in Python using the scikit-learn library:
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Generate a sample dataset (you can replace this with your own data)
X, y = make_regression(n_samples=100, n_features=1, noise=0.2, random_state=42)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = rf_model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
Decision tree:
Training a Decision Tree model involves using a dataset to create a tree-like structure of decisions to make predictions. Here's a simplified example in Python using the scikit-learn library:
```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Generate a sample dataset (you can replace this with your own data)
X, y = make_regression(n_samples=100, n_features=1, noise=0.2, random_state=42)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train a Decision Tree Regressor
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = tree_model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```In this example, we generate a synthetic dataset, split it into training and testing sets, create a Decision Tree Regressor, and make predictions. 
Performing Simple Linear Regression
Equation of linear regression
y=c+m1x1+m2x2+...+mnxny=c+m1x1+m2x2+...+mnxn
yy is the response
cc is the intercept
m1m1 is the coefficient for the first feature
mnmn is the coefficient for the nth feature
In our case:
y=c+m1×TVy=c+m1×TV
The mm values are called the model coefficients or model parameters.
Generic Steps in model building using statsmodels
We first assign the feature variable, TV, in this case, to the variable X and the response variable, Sales, to the variable y.
X =advertising['TV']
y =advertising['Sales']
Train-Test Split
You now need to split our variable into training and testing sets. You'll perform this by importing train_test_split from the sklearn.model_selection library. It is usually a good practice to keep 70% of the data in your train dataset and the rest 30% in your test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size =0.7, test_size =0.3, random_state =0;
# Let's now take a look at the train dataset
X_train.head()
74     213.4
3      151.5
185    205.0
26     142.9
90     134.3
Name: TV, dtype: float64
y_train.head()
74     17.0
3      16.5
185    22.6
26     15.0
90     14.0
Name: Sales, dtype: float64
Building a Linear Model
You first need to import the statsmodel.api library using which you'll perform the linear regression.
import statsmodels.api assm
By default, the statsmodels library fits a line on the dataset which passes through the origin. But in order to have an intercept, you need to manually use the add_constant attribute of statsmodels. And once you've added the constant to your X_train dataset, you can go ahead and fit a regression line using the OLS (Ordinary Least Squares) attribute of statsmodels as shown below
# Add a constant to get an intercept
X_train_sm = sm.add_constant(X_train)
# Fit the resgression line using 'OLS'
lr=sm.OLS(y_train, X_train_sm).fit()
# Print the parameters, i.e. the intercept and the slope of the regression line fitted
lr.params
const    6.948683
TV       0.054546
dtype: float64
In [19]:
# Performing a summary operation lists out all the different parameters of the regression line fitted
print(lr.summary())
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  Sales   R-squared:                       0.816
Model:                            OLS   Adj. R-squared:                  0.814
Method:                 Least Squares   F-statistic:                     611.2
Date:                Thu, 07 Mar 2019   Prob (F-statistic):           1.52e-52
Time:                        06:21:53   Log-Likelihood:                -321.12
No. Observations:                 140   AIC:                             646.2
Df Residuals:                     138   BIC:                             652.1
Df Model:                           1                                         
Covariance Type:            nonrobust
coefstd err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          6.9487      0.385     18.068      0.000       6.188       7.709
TV             0.0545      0.002     24.722      0.000       0.050       0.059
==============================================================================
Omnibus:                        0.027   Durbin-Watson:                   2.196
Prob(Omnibus):                  0.987   Jarque-Bera (JB):                0.150
Skew:                          -0.006   Prob(JB):                        0.928
Kurtosis:                       2.840   Cond. No.                         328
 EDA Libraries: 
 Import pandas as pd
Import numpy as np 
 Import matplotlib.colors as col 
From mpl_toolkits.mplot3d import Axes3D 
Import matplotlib.pyplot as plt
Import seaborn as sns
%matplotlib inline 
 Import datetime 
From pathlib import Path
Import random
8.EVALUATION:
Definition: Evaluation is the process of assessing or appraising something based on certain criteria or standards to determine its value, effectiveness, quality, or significance
Certainly! Here's a basic example of how to perform model evaluation using Python and the scikit-learn library. We'll use a simple classification model for illustration:
```python
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Assuming you have a dataset with features X and labels y
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train a model (Logistic Regression in this case)
model = LogisticRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
# Print the evaluation metrics
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
print("Confusion Matrix:\n", conf_matrix)
Model Evaluation
Residual analysis
To validate assumptions of the model, and hence the reliability for inference
Distribution of the error terms:We need to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like,
y_train_pred=lr.predict(X_train_sm)
res= (y_train -y_train_pred)
fig=plt.figure()
sns.distplot(res, bins =15)
fig.suptitle('Error Terms', fontsize=15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize=15)         # X-label
plt.show()


