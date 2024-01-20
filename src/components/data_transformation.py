import pandas as pd 
import numpy as np 
import sys
import os
from src.exception import CustomException
from src.logger import logging

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
    pass

class DataTransformation:
    def __init__():
        pass

    def initiate_data_ingestion():
        pass

