#!/usr/bin/env python
# coding: utf-8

# In[4]:


# data cleaning

import pandas as pd

# Load your dataset
df = pd.read_csv("WeatherDataset.csv")
df.head()


# In[5]:


# Check for missing values
print(df.isnull().sum())


# In[6]:


#HANDLING DATE AND TIME COLUM FROM CATEGORICAL TO NUMERICAL
df['Date/Time'] = pd.to_datetime(df['Date/Time'])
df['Year'] = df['Date/Time'].dt.year
df['Month'] = df['Date/Time'].dt.month
df['Day'] = df['Date/Time'].dt.day
df['Hour'] = df['Date/Time'].dt.hour


# In[7]:


#HANDLING WEATHER COLUM FROM CATEGORICAL TO NUMERICAL USING 1-HOT ENCODING
df_encoded = pd.get_dummies(df, columns=['Weather'], drop_first=True)


# In[8]:


#SCALING MY DATA USING STANDARD SCALER EXCEPT DATE AND WEATHER COLUMNS
from sklearn.preprocessing import StandardScaler

# scaling data except for date and weather columns
numerical_cols = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# In[9]:


#CHECKING FOR OUTLIERS IN DATASE COLUMS USING IQR (Interquartile Range)
import pandas as pd
numerical_cols = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']

# Print the number of outliers for each column
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
    print(f"Number of outliers in '{col}':", len(outliers))


# #  removing outliers from your dataset but for now am not deleting them
# df_no_outliers = df[~((df['Temp_C'] < (Q1 - 1.5 * IQR)) | (df['Temp_C'] > (Q3 + 1.5 * IQR)))]


# In[10]:


#CHECKING FOR OUTLIERS IN DATASE COLUMS USING Box Plots
import seaborn as sns
import matplotlib.pyplot as plt

# Set up the matplotlib figure
f, axes = plt.subplots(len(numerical_cols), 1, figsize=(8, 10))

# Plot a box plot for each numerical column
for i, col in enumerate(numerical_cols):
    sns.boxplot(x=df[col], ax=axes[i])
    axes[i].set_title(f'Box plot of {col}')

plt.tight_layout()
plt.show()


# In[11]:


#IMPUTING OUTLIERS WITH MEDIAN FOR THE SPECIFIC COLUMN
numerical_cols = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    median = df[col].median()

    # Identify outliers
    outliers_condition = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))

    # Impute outliers with the median of the column
    df.loc[outliers_condition, col] = median

    # Optionally, print out the number of outliers replaced
    print(f"Number of outliers replaced in '{col}':", outliers_condition.sum())


# In[12]:


#checking the updated summary statistics for the numerical columns
print(df.describe())


# In[13]:


#checking if outliers have been reduced
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
    print(f"Remaining outliers in '{col}':", len(outliers))


# In[14]:


#Dealing with assumptions of multiple regression analysis

# 1. LINEARITY
import seaborn as sns
import matplotlib.pyplot as plt

#'Temp_C' is the dependent variable and others are independent variables
independent_vars = ['Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']

for var in independent_vars:
    sns.scatterplot(x=df[var], y=df['Temp_C'])
    plt.xlabel(var)
    plt.ylabel('Temp_C')
    plt.title(f'Scatter plot between Temp_C and {var}')
    plt.show()


# In[15]:


#2. Correlation matrix
corr_matrix = df[independent_vars + ['Temp_C']].corr()
print(corr_matrix)


# In[16]:


'''Independence The residuals (errors) of the model should be independent of each other.
This is often assessed using the Durbin-Watson test, where a value close to 2.0 suggests independence.

Note: The actual calculation of residuals and subsequent testing for independence typically come after you've 
fitted a model. It's a bit of a catch-22, but for now, ensure there's no autocorrelation in your independent variables,
especially if you're dealing with time series data.'''


# In[17]:


#Developing a Multiple Regression Model
import statsmodels.api as sm

#'Temp_C' is the dependent variable
X = df[independent_vars] # Independent variables
y = df['Temp_C']

# Adding a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Summary of the model
print(model.summary())


# In[ ]:


'''3.Homoscedasticity
The variance of error terms should be constant across all levels of the independent variables.
This can be visually inspected using a scatter plot of residuals after model fitting.'''


# In[18]:


# Calculate residuals
residuals = model.resid

# Plot residuals
plt.scatter(model.fittedvalues, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()


# In[ ]:


'''Normality of Residuals
The residuals should be normally distributed. This can be checked with a Q-Q plot or a
Shapiro-Wilk test after fitting the model.'''


# In[19]:


import scipy.stats as stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Normal Q-Q plot")
plt.show()

# Shapiro-Wilk test
from scipy import stats
shapiro_test = stats.shapiro(residuals)
print(f"Shapiro-Wilk test: Statistic={shapiro_test[0]}, p-value={shapiro_test[1]}")


# In[20]:


#Splitting the Dataset into Training and Testing Sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


#Creating and Training the Multiple Regression Model
from sklearn.linear_model import LinearRegression

# Create a linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)


# In[23]:


'''Examine Model Performance, Interpretability, Accuracy, and Complexity

Model Performance and Interpretability:'''
# Coefficients
print("Coefficients:", model.coef_)

# Intercept
print("Intercept:", model.intercept_)

# R-squared value
print("R-squared:", model.score(X_test, y_test))


# In[24]:


'''Model Accuracy: MSE AND R-SQUARED BEFORE FEATURE SELECTION'''
from sklearn.metrics import mean_squared_error, r2_score
# Make predictions on the testing set
y_pred_before_fs = model.predict(X_test)

# Evaluate model performance before feature selection
mse_before_fs = mean_squared_error(y_test, y_pred_before_fs)
r2_before_fs = r2_score(y_test, y_pred_before_fs)

print("Before feature selection:")
print("Mean Squared Error:", mse_before_fs)
print("R-squared:", r2_before_fs)


# In[25]:


'''After Feature Selection: MSE AND R-SQUARED VALUES'''
from sklearn.linear_model import LassoCV

# Fit Lasso regression model for feature selection
lasso_model = LassoCV(cv=5)
lasso_model.fit(X_train, y_train)

# Select features with non-zero coefficients
selected_features = X_train.columns[lasso_model.coef_ != 0]

# Retrain the model with selected features
model_after_fs = LinearRegression()
model_after_fs.fit(X_train[selected_features], y_train)

# Make predictions on the testing set
y_pred_after_fs = model_after_fs.predict(X_test[selected_features])

# Evaluate model performance after feature selection
mse_after_fs = mean_squared_error(y_test, y_pred_after_fs)
r2_after_fs = r2_score(y_test, y_pred_after_fs)

print("\nAfter feature selection:")
print("Mean Squared Error:", mse_after_fs)
print("R-squared:", r2_after_fs)


# In[26]:


#MODEL DEPLOYEMENT Installation of flask
#Saving model in model.pkl file
import joblib
# Save the trained model to a file
joblib.dump(model, 'model.pkl')


# In[ ]:




