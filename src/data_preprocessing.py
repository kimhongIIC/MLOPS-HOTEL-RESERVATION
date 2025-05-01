import os 
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from joblib import parallel_backend

logger = get_logger(__name__)

class DataPreprocessor:
    
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config_path = config_path
        
        # Load configuration
        self.config = read_yaml(self.config_path)
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def preprocess_data(self, df):
        try:
            logger.info('Starting data preprocessing...')
            
            df.drop(columns=["Unnamed: 0","Booking_ID"], inplace=True)
            df.drop_duplicates(inplace=True)
            
            cat_cols = self.config['data_preprocessing']["categorical_columns"]
            num_cols = self.config['data_preprocessing']["numerical_columns"]
            
            logger.info('Encoding categorical columns...')
            label_encoder = LabelEncoder()
            mappings = {}
            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = {label:code for label, code in zip(label_encoder.classes_,label_encoder.transform(label_encoder.classes_))}
                logger.info(f'Encoded {col} with mapping: {mappings[col]}')     
            logger.info('Label mapping are:')
            for col, mapping in mappings.items():
                logger.info(f'{col}: {mapping}')
                
            logger.info('Handling skewness...')
            skew_threshold = self.config['data_preprocessing']["skew_threshold"]
            skewness = df[num_cols].apply(lambda x:x.skew())
            for column in skewness[skewness>skew_threshold].index:
                df[column] = np.log1p(df[column])
            return df
                        
        except Exception as e:
            logger.error(f'Error during data preprocessing: {e}')
            raise CustomException('Error during preprocess data', e)
    
    def balance_data(self, df):
        try:
            logger.info('Handling Imbalanced data')
            X = df.drop(columns=["booking_status"])
            y = df["booking_status"]
            
            with parallel_backend('threading'):
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
            
            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df["booking_status"] = y_resampled
            logger.info('Data balancing completed.')
            return balanced_df
        except Exception as e:
            logger.error(f'Error during balancing preprocessing: {e}')
            raise CustomException('Error during balancing data', e)
        
    def select_features(self, df):
        try:
            logger.info('Feature selection using Random Forest...')
            X = df.drop(columns=["booking_status"])
            y = df["booking_status"]
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            
            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance
            })
            top_features_importance_df=feature_importance_df.sort_values(by='importance', ascending=False)
            num_features_to_select = self.config['data_preprocessing']["no_of_features"]
            top_importance_feature = top_features_importance_df['feature'].head(num_features_to_select).values
            top_importance_feature_df = df[top_importance_feature.tolist() + ["booking_status"]]
            logger.info(f'Selected top {num_features_to_select} features: {top_importance_feature}')
            return top_importance_feature_df        
            
        except Exception as e:
            logger.error(f'Error during feature selection: {e}')
            raise CustomException('Error during feature selection', e)
    
    def save_data(self, df,file_path):
        try:
            logger.info('Saving processed data...')
            df.to_csv(file_path, index=False)
            logger.info(f'Data saved to {file_path}')
        except Exception as e:
            logger.error(f'Error saving data: {e}')
            raise CustomException('Error saving data', e)
        
    def process(self):
        try:
            logger.info('Loading training data...')
            train_df = load_data(self.train_path)
            logger.info('Loading test data...')
            test_df = load_data(self.test_path)
            
            # Preprocess training data
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            
            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]
            
            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)
            
            logger.info('Data preprocessing completed successfully.')
            
        except Exception as e:
            logger.error(f'Error in processing data: {e}')
            raise CustomException('Error in processing data', e)

if __name__ == "__main__":
    processor = DataPreprocessor(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_PATH
    )
    processor.process()
