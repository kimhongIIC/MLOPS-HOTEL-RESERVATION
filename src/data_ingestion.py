import os 
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.custom_exception import CustomException
from src.logger import get_logger
from config.paths_config import *
from utils.common_functions import read_yaml 

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]
        
        os.makedirs(RAW_DIR, exist_ok=True)
        
        logger.info(f"DataIngestion class initialized {self.bucket_name} and file is {self.file_name}")
    
    def download_data(self):
        try: 
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"Data downloaded from bucket {self.bucket_name} to {RAW_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error downloading data")
            raise CustomException("Failed to download CSV file",e)

    def split_data(self):
        try: 
            logger.info("Starting data splitting process")
            data = pd.read_csv(RAW_FILE_PATH)
            
            train_data, test_data = train_test_split(data, test_size=1-self.train_test_ratio, random_state=42)
            
            train_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)
            
            logger.info(f"Train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to {TEST_FILE_PATH}")
        
        except Exception as e:
            logger.error(f"Error while splitting data")
            raise CustomException("Failed to split data into training and test sets",e)
    
    def run(self):
        try:
            logger.info("Strating data ingestion process")
            self.download_data()
            self.split_data()
            
            logger.info("Data ingestion process completed successfully")
        except Exception as ce:
            logger.error(f"CustomException: {str(ce)}")
        finally:
            logger.info("Data ingestion process finished")
            

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
