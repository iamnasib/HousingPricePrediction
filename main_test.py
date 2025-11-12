import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from scipy.stats import randint

# Loading the data
housing=pd.read_csv('housing.csv')

housing['income_cat']= pd.cut(housing['median_income'],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

split=StratifiedShuffleSplit(n_splits=2,test_size=0.2,random_state=42)

for train_index,test_index in split.split(housing,housing['income_cat']):
    strat_train_set=housing.loc[train_index].drop('income_cat',axis=1) #drop the income_cat as we don't need that because spliting is      already done
    strat_test_set=housing.loc[test_index].drop('income_cat',axis=1)

#Now we will work on the copy of Train Set

housing=strat_train_set.copy()

# Separate Feature and Labels
housing_labels=housing['median_house_value'].copy()
housing=housing.drop('median_house_value',axis=1)

# Separate Numerical and categorical columns
num_attribs=housing.select_dtypes(include=[np.number]).columns.tolist() # OR housing.drop('ocean_proximity',axis=1).columns.tolist()
cat_attribs = housing.select_dtypes(exclude=[np.number]).columns.tolist()

# Create pipelines
# numerical columns
num_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])

# Categorical Pipeline
cat_pipeline= Pipeline([
    ('encoder',OneHotEncoder(handle_unknown='ignore')) # If some new category comes while predicting it will ignore that(Set as 0)
])

#Full PipeLine
full_pipeline=ColumnTransformer([
    ('numerical',num_pipeline,num_attribs),
    ('categorical',cat_pipeline,cat_attribs)
])

housing_prepared=full_pipeline.fit_transform(housing)

# Train the shortlisted models and check their error to finalize one
linear_reg=LinearRegression()
linear_reg.fit(housing_prepared,housing_labels)

decision_reg=DecisionTreeRegressor()
decision_reg.fit(housing_prepared,housing_labels)

random_forest_reg=RandomForestRegressor()
# random_forest_reg.fit(housing_prepared,housing_labels)

#Predict using training Data
# linear_preds=linear_reg.predict(housing_prepared)
# decision_preds=decision_reg.predict(housing_prepared)
# random_forest_preds=random_forest_reg.predict(housing_prepared)

#Calculate RMSE 
#IMPORTANT:Training RMSE only shows how well the model fits the training data. It does not tell us how well it will perform on unseen data
# linear_err=root_mean_squared_error(housing_labels,linear_preds)
# decision_err=root_mean_squared_error(housing_labels,decision_preds)
# random_forest_err=root_mean_squared_error(housing_labels,random_forest_preds)

# print(f"The RSME of LinearRegression is: {linear_err}")
# print(f"The RSME of DecisionTreeRegressor is: {decision_err}")
# print(f"The RSME of RandomForestRegressor is: {random_forest_err}")

# Cross validation to get a better estimate of the errors using K-Fold Cross Validation

# linear_rmses=pd.Series(-cross_val_score(linear_reg,housing_prepared,housing_labels,scoring='neg_root_mean_squared_error',cv=10))
# decision_rmses=pd.Series(-cross_val_score(decision_reg,housing_prepared,housing_labels,scoring='neg_root_mean_squared_error',cv=10))
# random_forest_rmses=pd.Series(-cross_val_score(random_forest_reg,housing_prepared,housing_labels,scoring='neg_root_mean_squared_error',cv=10))
# print(f"The mean of RSMEs (10 Folds)  of LinearRegression is: {linear_rmses}")
# print(f"The mean of RSMEs (10 Folds) of DecisionTreeRegressor is: {decision_rmses}")
# print(f"The mean of RSMEs (10 Folds) of RandomForestRegressor is: {random_forest_rmses}")

rnd_hyper_params={
        'n_estimators':randint(50,300),
        'max_features': ['sqrt', 'log2'],
        'max_depth': randint(5,50),
        'min_samples_split': randint(2,10),
        'min_samples_leaf': randint(1,5)
    }

# tuned_model=RandomizedSearchCV(
#         random_forest_reg,
#         rnd_hyper_params,
#         n_iter=50,
#         cv=5,
#         scoring='neg_root_mean_squared_error',
#         n_jobs=-1,
#         random_state=42
#         )

# tuned_model.fit(housing_prepared,housing_labels)

grid_params={
    'max_depth': [30,36,40],
    'min_samples_split': [4,5,6],
    'n_estimators': [140, 145, 150]
}

tuned_model=GridSearchCV(
    random_forest_reg,
    grid_params,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
tuned_model.fit(housing_prepared,housing_labels)

print(-tuned_model.best_score_,tuned_model.best_params_)

# From the above results we can see that RandomForestRegressor is performing the best so we will use that for final evaluation on Test Set
# END OF TESTING PHASE#
# MAIN code will be in main.py file#