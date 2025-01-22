import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionconfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion process starts')

        try:
            logging.info('Read Dataset as pandas Dataframe')

            # Read the dataset
            df = pd.read_csv(os.path.join('notebooks/data', 'gemstone.csv'))

            # Make directory to save raw data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the raw data to csv file
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info('Raw data is created')

            # Train-test split
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            # Save train and test to respective csv files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of Data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at Data Ingestion Stage')
            raise CustomException(e, sys)

    