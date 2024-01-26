import pandas as pd 
import numpy as np 
import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.utils import save_object
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path 

@dataclass

class DataIngestionConfig:

    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")

        try:


            data=pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vREhwEnHejwV61BVmYEdlpDsL4gt-mhs1ShilXVJhqEltCHyman08UT6LLI0kC9_3DUR2cQuoamFgqP/pub?gid=0&single=true&output=csv")
            logging.info("Data Reading")
            os.makedirs(os.path.dirname(os.path.join(self.data_ingestion_config.raw_data_path)),exist_ok=True)

            data.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            logging.info("Saved the raw dataset in artifact folder")

            train_data,test_data=train_test_split(data,test_size=0.25)
            logging.info("Created Train test Split")

            train_data.to_csv(self.data_ingestion_config.train_data_path)
            test_data.to_csv(self.data_ingestion_config.test_data_path)
            logging.info("Saved Train test data in artifacts")

            logging.info("Data Ingestion completed")

            return (self.data_ingestion_config.train_data_path,self.data_ingestion_config.test_data_path)


        except Exception as e:
            raise CustomException(e, sys)










