# Importing the sys module to interact with the Python runtime environment
import sys 

# Importing the os module to interact with the operating system
import os

# Importing the DataIngestion class from the components module inside the Xray package
from Xray.components.data_ingestion import DataIngestion

# Importing the DataIngestionArtifact class which will hold the output of the data ingestion process
from Xray.entity.artifact_entity import DataIngestionArtifact

# Importing the DataIngestionConfig class which contains configurations for data ingestion
from Xray.entity.config_entity import DataIngestionConfig

# Importing a custom exception class to handle errors specific to the Xray project
from Xray.exception import XRayException

# Importing the logging utility for writing logs
from Xray.logger import logging 

# Defining the TrainPipeline class which orchestrates the training pipeline process
class TrainPipeline:
    
    # Constructor method that initializes the pipeline with the data ingestion configuration
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        
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
if __name__ == "__main__":
    
    # Creating an instance of the TrainPipeline class
    train_pipeline = TrainPipeline()
    
    # Starting the data ingestion process
    train_pipeline.start_data_ingestion()
# End of the TrainPipeline class
           
            