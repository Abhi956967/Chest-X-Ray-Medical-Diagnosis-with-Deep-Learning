import sys
from Xray.components.data_ingestion import DataIngestion
from Xray.components.data_transformation import DataTransformation
# from Xray.components.model_training import ModelTrainer
# from Xray.components.model_evaluation import ModelEvaluation
# from Xray.components.model_pusher import ModelPusher
from Xray.exception import XRayException
from Xray.logger import logging

from Xray.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact
    # ModelTrainerArtifact,
    # ModelEvaluationArtifact,
    # ModelPusherArtifact
    )

from Xray.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig
    # ModelTrainerConfig,
    # ModelEvaluationConfig,
    # ModelPusherConfig
)


class TrainPipeline:
    
    # Constructor method that initializes the pipeline with the data ingestion configuration
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        
    # Method to start the data ingestion process and return the artifact
    def start_data_ingestion(self) -> DataIngestionArtifact:
        # Logging the entry into the method
        logging.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            # Logging that data fetching from S3 is starting
            logging.info("Getting the data from s3 bucket")

            # Creating an instance of the DataIngestion class with the configuration
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config,
            )

            # Starting the ingestion process and storing the result in an artifact object
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            # Logging successful data retrieval
            logging.info("Got the train_set and test_set from s3")

            # Logging the exit of the method
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )

            # Returning the artifact containing paths or metadata for ingested data
            return data_ingestion_artifact

        # Handling any exceptions that occur and raising a custom exception
        except Exception as e:
            raise XRayException(e, sys)
        
        
# Entry point of the script
# if __name__ == "__main__":
    
#     # Creating an instance of the TrainPipeline class
#     train_pipeline = TrainPipeline()
    
#     # Starting the data ingestion process
#     train_pipeline.start_data_ingestion()
# # End of the TrainPipeline class

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        
        logging.info("Entered the start_data_transformation method of TrainPipeline class")

        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
            )

            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )

            logging.info(
                "Exited the start_data_transformation method of TrainPipeline class"
            )

            return data_transformation_artifact

        except Exception as e:
            raise XRayException(e, sys)
        
    
    
    
    def run_pipeline(self) -> None:
        logging.info("Entered the run_pipeline method of TrainPipeline class")

        try:
            # Starting data ingestion
            data_ingestion_artifact = DataIngestionArtifact = self.start_data_ingestion()
            data_transformation_artifact: DataTransformationArtifact = (
                self.start_data_transformation(
                    data_ingestion_artifact=data_ingestion_artifact
                )
            )

            # Starting data transformation
            
            logging.info("Exited the run_pipeline method of TrainPipeline class")
            
        except Exception as e:
            raise XRayException(e, sys)
           
            