import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from scipy.stats import randint

MODEL_FILE= 'model.pkl'
PIPELINE_FILE= 'pipeline.pkl'

def build_pipeline(num_attribs,cat_attribs):
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

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    print("Training the model as no pre-trained model found...")
    # Loading the data

    housing=pd.read_csv('housing.csv')
    housing['income_cat']= pd.cut(housing['median_income'],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

    for train_index, test_index in split.split(housing,housing['income_cat']):
        housing.loc[test_index].drop('income_cat',axis=1).copy().to_csv('test_set.csv',index=False)
        housing=housing.loc[train_index].drop('income_cat',axis=1)
        
    housing_labels=housing['median_house_value'].copy()
    housing_features=housing.drop('median_house_value',axis=1)
    
    num_attribs=housing_features.select_dtypes(include=[np.number]).columns.tolist()
    cat_attribs=housing_features.select_dtypes(exclude=[np.number]).columns.tolist()

    pipeline = build_pipeline(num_attribs,cat_attribs)
    transformed_data = pipeline.fit_transform(housing_features)

    #Create the model
    model= RandomForestRegressor(random_state=42)

    # Train  Fine tune using Grid search CV
    # HyperParameter Tuning using Grid Search CV
    grid_params={
    'max_depth': [30,36,40],
    'min_samples_split': [4,5,6],
    'n_estimators': [140, 145, 150]
    } # These parameter found using the random Search CV in main_test.py file

    tuned_model=GridSearchCV(
        model,
        grid_params,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    tuned_model.fit(transformed_data,housing_labels)

    #SAve the model and pipeline to disk
    joblib.dump(tuned_model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("Model training completed and saved to disk.")

else:
    print("Loading the pre-trained model...")
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)

    input_data=pd.read_csv('input.csv')

    transformed_input=pipeline.transform(input_data)

    input_data['predicted_house_value']=model.predict(transformed_input)

    input_data.to_csv('predictions.csv',index=False)
    print("Predictions saved to predictions.csv")