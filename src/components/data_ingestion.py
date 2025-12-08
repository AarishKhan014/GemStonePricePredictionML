# src/data_ingestion.py
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process")
        try:
            # Read dataset
            csv_path = os.path.join('notebooks', 'data', 'Gemstone.csv')
            if not os.path.exists(csv_path):
                logging.info("Gemstone File Not Found.")
                raise CustomException(f"File Now Found {csv_path}", sys)
            
            df = pd.read_csv(csv_path)
            logging.info(f"Dataset read successfully with shape {df.shape}")

            # Create artifacts folder if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            # Split train and test
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info(f"Train data saved at {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved at {self.ingestion_config.test_data_path}")

            logging.info("Data ingestion completed successfully")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error("Exception occurred during data ingestion")
            raise CustomException(e, sys)
        
## for testing
# if __name__ == '__main__':
#     obj=DataIngestion()
#     train_data_path, test_data_path = obj.initiate_data_ingestion()
#     data_transformation = DataTransformation()
#     train_arr, test_arr, pkle = data_transformation.initiate_data_transformation(train_data_path, test_data_path) 