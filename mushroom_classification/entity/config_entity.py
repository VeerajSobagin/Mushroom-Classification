import os,sys
from mushroom_classification.exception import mushroomException
from mushroom_classification.logger import logging
from datetime import datetime

'''
there are main 6 steps in this project 
    1-DataIngestion
    2-DataValidation
    3-DataTransformation
    4-ModelTrainer
    5-ModelEvaluation
    6-ModelPusher

this file will act as input file for each and every step
'''

FILE_NAME = "mushrooms.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
TARGET_ENCODER_OBJECT_FILE_NAME = "target_encoder.pkl"
MODEL_FILE_NAME = "model.pkl"

class TrainingPipelineConfig:

    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception  as e:
            raise mushroomException(e,sys)     


'''
input for data_ingetion.py file it will fetch input from hear
'''
class DataIngestionConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.database_name="mushroom"
            self.collection_name="mushroom-classification"
            # creating main direcory for storing data from database and test, train data
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir , "data_ingestion")
            # storing directory for data imported from database "data_ingestion/feature_store/"
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir,"feature_store",FILE_NAME)
            # directory for storing train and test data "data_ingestion/dataset/"
            self.train_file_path = os.path.join(self.data_ingestion_dir,"dataset",TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir,"dataset",TEST_FILE_NAME)
            self.test_size = 0.2
        except Exception  as e:
            raise mushroomException(e,sys)     

    def to_dict(self,)->dict:
        try:
            return self.__dict__
        except Exception  as e:
            raise mushroomException(e,sys)     


class DataTransformationConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir , "data_transformation")
        self.transform_object_path = os.path.join(self.data_transformation_dir,"transformer",TRANSFORMER_OBJECT_FILE_NAME)
        self.transformed_train_path =  os.path.join(self.data_transformation_dir,"transformed",TRAIN_FILE_NAME.replace("csv","npz"))
        self.transformed_test_path =os.path.join(self.data_transformation_dir,"transformed",TEST_FILE_NAME.replace("csv","npz"))
        self.target_encoder_path = os.path.join(self.data_transformation_dir,"target_encoder",TARGET_ENCODER_OBJECT_FILE_NAME)

class ModelTrainerConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir , "model_trainer")
        self.model_path = os.path.join(self.model_trainer_dir,"model",MODEL_FILE_NAME)
        self.expected_score = 0.7
        self.overfitting_threshold = 0.1