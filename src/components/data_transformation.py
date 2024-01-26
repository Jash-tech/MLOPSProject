import pandas as pd 
import numpy as np 
import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from pathlib import Path 

@dataclass

class DataTransformationConfig:
    preprocessor_obj_filepath=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation(self):
        try:

            logging.info('Data Transformation initiated')

            # Define which columns should be ordinal-encoded and which should be scaled
            numerical_columns=['carat', 'depth', 'table', 'x', 'y', 'z']
            categorical_columns=['cut', 'color', 'clarity']

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            # Numerical Pipeline
            num_pipeline=Pipeline (
                      steps=[
                      ("imputer",SimpleImputer(strategy='median')),
                      ("scaler",StandardScaler())
                       ]
                       )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(

                steps=[
                  ("imputer",SimpleImputer(strategy="most_frequent")),
                  ("ordinalencoder",OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories]))

                  ]
                  )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_columns),
            ('cat_pipeline',cat_pipeline,categorical_columns)
            ])
            
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)



    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data complete")

            preprocessing_obj=self.get_data_transformation()

            target_column_name = 'price'
            drop_columns = [target_column_name,'id']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1) #xtrain
            target_feature_train_df=train_df[target_column_name] #ytrain
            
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1) #xtest
            target_feature_test_df=test_df[target_column_name]   #ytest

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_filepath,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")

            return (
                train_arr,
                test_arr
            )




        except Exception as e:
            raise CustomException(e,sys)

