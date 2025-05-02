# import os
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import RandomizedSearchCV
# import lightgbm as lgb
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix
# from src.logger import get_logger
# from src.custom_exception import CustomException
# from config.paths_config import *
# from config.model_params import *
# from utils.common_functions import read_yaml, load_data
# from scipy.stats import randint, uniform

# import mlflow
# import mlflow.sklearn

# logger = get_logger(__name__)

# class ModelTraining: 
#     def __init__(self, train_path, test_path, model_output_path):
#         self.train_path = train_path
#         self.test_path = test_path
#         self.model_output_path = model_output_path
        
#         self.params_dist = LIGHTGBM_PARAMS
#         self.random_search_params = RANDOM_SEARCH_PARAMS
        
#     def load_and_split_data(self):
#         try:
#             logger.info(f'Loading data from {self.train_path}')
#             train_df = load_data(self.train_path)
            
#             logger.info(f'Loading data from {self.test_path}')
#             test_df = load_data(self.test_path)
            
#             X_train = train_df.drop(columns=['booking_status'])
#             y_train = train_df['booking_status']
            
#             X_test = test_df.drop(columns=['booking_status'])
#             y_test = test_df['booking_status']
            
#             logger.info('Splitting data into train and test sets...')
            
#             return X_train, y_train, X_test, y_test
        
#         except Exception as e:
#             logger.error(f'Error loading and splitting data: {e}')
#             raise CustomException('Error loading and splitting data', e)
    
#     def train_lgbm(self, X_train, y_train):
#         try:
#             logger.info('Initializing LightGBM model...')
            
#             lgbm = lgb.LGBMClassifier(random_state=self.random_search_params['random_state'])
            
#             logger.info('Performing Randomized Search for hyperparameter tuning...')
#             random_search = RandomizedSearchCV(
#                 estimator=lgbm,
#                 param_distributions=self.params_dist,
#                 n_iter=self.random_search_params['n_iter'],
#                 cv=self.random_search_params['cv'],
#                 n_jobs=self.random_search_params['n_jobs'],
#                 verbose=self.random_search_params['verbose'],
#                 random_state=self.random_search_params['random_state'],
#                 scoring=self.random_search_params['scoring']
#             )
            
#             random_search.fit(X_train, y_train, early_stopping_rounds=10)
#             logger.info('Randomized Search completed.')
            
#             best_params = random_search.best_params_
#             best_lgbm_model = random_search.best_estimator_
            
#             logger.info(f'Best parameters found: {best_params}')
            
#             return best_lgbm_model
        
#         except Exception as e:
#             logger.error(f'Error during model training: {e}')
#             raise CustomException('Error during model training', e)
    
#     def evaluate_model(self, model, X_test, y_test):
#         try:
#             logger.info('Evaluating our model')
#             y_pred = model.predict(X_test)
            
#             accuracy = accuracy_score(y_test, y_pred)
#             f1 = f1_score(y_test, y_pred)
#             recall = recall_score(y_test, y_pred)
#             precision = precision_score(y_test, y_pred)
            
#             logger.info(f'Accuracy: {accuracy}')
#             logger.info(f'F1 Score: {f1}')
#             logger.info(f'Recall: {recall}')
#             logger.info(f'Precision: {precision}')
            
#             return {
#                 'accuracy': accuracy,
#                 'f1_score': f1,
#                 'recall': recall,
#                 'precision': precision
#             }
            
#         except Exception as e:
#             logger.error(f'Error during model evaluation: {e}')
#             raise CustomException('Error during model evaluation', e)
    
#     def save_model(self, model):
#         try:
#             os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
#             logger.info(f'Saving model to {self.model_output_path}')
#             joblib.dump(model, self.model_output_path)
#             logger.info('Model saved successfully.')
#         except Exception as e:
#             logger.error(f'Error saving model: {e}')
#             raise CustomException('Error saving model', e)
    
#     def run(self):
#         try:
#             with mlflow.start_run():
                
#                 logger.info('Starting model training pipeline...')
#                 logger.info('Starting our MLFLOW experimentation')
#                 logger.info('Logging the training and testing dataset to MLFLOW')
                
#                 mlflow.log_artifact(self.train_path, artifact_path='datasets')
#                 mlflow.log_artifact(self.test_path, artifact_path='datasets')
                
#                 X_train, y_train, X_test, y_test = self.load_and_split_data()
                
#                 model = self.train_lgbm(X_train, y_train)
                
#                 metrics = self.evaluate_model(model, X_test, y_test)
                
#                 self.save_model(model)
                
#                 logger.info('Logging model into MLFLOW')
#                 mlflow.log_artifact(self.model_output_path)
                
#                 logger.info('Logging model params into MLFLOW')
#                 mlflow.log_params(model.get_params())
                
#                 logger.info('Logging model metrics into MLFLOW')
#                 mlflow.log_metrics(metrics)
            
#                 logger.info('Model training process completed successfully.')
        
#         except Exception as e:
#             logger.error(f'Error in model training process: {e}')
#             raise CustomException('Error in model training process', e)
        
# if __name__ == "__main__":
#     model_trainer = ModelTraining(
#         train_path=PROCESSED_TRAIN_DATA_PATH,
#         test_path=PROCESSED_TEST_DATA_PATH,
#         model_output_path=MODEL_OUTPUT_PATH
#     )
#     model_trainer.run()
    
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import load_data

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining: 
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        
        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
        
    def load_and_split_data(self):
        try:
            logger.info(f'Loading data from {self.train_path}')
            train_df = load_data(self.train_path)
            
            logger.info(f'Loading data from {self.test_path}')
            test_df = load_data(self.test_path)
            
            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']
            
            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']
            
            logger.info('Splitting data into train and test sets...')
            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            logger.error(f'Error loading and splitting data: {e}')
            raise CustomException('Error loading and splitting data', e)
    
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info('Initializing LightGBM model...')
            # carve out a small validation set for early stopping
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train,
                test_size=0.1,
                random_state=self.random_search_params['random_state']
            )
            
            lgbm = lgb.LGBMClassifier(
                random_state=self.random_search_params['random_state'],
                verbosity=1
            )
            
            logger.info('Performing Randomized Search for hyperparameter tuning...')
            random_search = RandomizedSearchCV(
                estimator=lgbm,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                n_jobs=self.random_search_params['n_jobs'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                scoring=self.random_search_params['scoring']
            )
            
            # pass early-stopping args into fit()
            random_search.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
            logger.info('Randomized Search completed.')
            
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            logger.info(f'Best parameters found: {best_params}')
            
            return best_lgbm_model
        
        except Exception as e:
            logger.error(f'Error during model training: {e}')
            raise CustomException('Error during model training', e)
    
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info('Evaluating our model')
            y_pred = model.predict(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred)
            }
            for name, val in metrics.items():
                logger.info(f'{name}: {val}')
            
            return metrics
            
        except Exception as e:
            logger.error(f'Error during model evaluation: {e}')
            raise CustomException('Error during model evaluation', e)
    
    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            logger.info(f'Saving model to {self.model_output_path}')
            joblib.dump(model, self.model_output_path)
            logger.info('Model saved successfully.')
        except Exception as e:
            logger.error(f'Error saving model: {e}')
            raise CustomException('Error saving model', e)
    
    def run(self):
        try:
            with mlflow.start_run():
                logger.info('Starting model training pipeline...')
                
                mlflow.log_artifact(self.train_path, artifact_path='datasets')
                mlflow.log_artifact(self.test_path, artifact_path='datasets')
                
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                model = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(model, X_test, y_test)
                self.save_model(model)
                
                logger.info('Logging model into MLFLOW')
                mlflow.log_artifact(self.model_output_path)
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                
                logger.info('Model training process completed successfully.')
        
        except Exception as e:
            logger.error(f'Error in model training process: {e}')
            raise CustomException('Error in model training process', e)
        
if __name__ == "__main__":
    trainer = ModelTraining(
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH
    )
    trainer.run()
