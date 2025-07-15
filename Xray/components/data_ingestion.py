# Importing sys module to get system-specific exception information
import sys 

# Importing S3Operation class to handle interactions with AWS S3
from Xray.cloud_storage.s3_operation import S3Operation

# Importing constants used for configuration paths and bucket details
from Xray.constant.training_pipeline import *

# Importing the data class used to store the result (artifact) of the data ingestion process
from Xray.entity.artifact_entity import DataIngestionArtifact

# Importing the configuration class that contains paths and settings for ingestion
from Xray.entity.config_entity import DataIngestionConfig 

# Importing the custom exception class for consistent error handling
from Xray.exception import XRayException 

# Importing the custom logging module to log messages to a file or console
from Xray.logger import logging 



# Defining the DataIngestion class which is responsible for retrieving data from S3
class DataIngestion:
    
    # Constructor to initialize with the config and S3 operation helper
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        # Assigning the passed configuration object
        self.data_ingestion_config = data_ingestion_config
        
        # Creating an instance of S3Operation to perform AWS S3 tasks
        self.s3 = S3Operation()

        
        # Method to download data from an S3 bucket into a local folder
    def get_data_from_s3(self) -> None:
        try:
            # Logging entry into the method
            logging.info("Entered the get_data_from_s3 method of Data ingestion class")

            # Syncing a folder from the specified S3 bucket to the local directory
            self.s3.sync_folder_from_s3(
                folder=self.data_ingestion_config.data_path,                  # Local target folder
                bucket_name=self.data_ingestion_config.bucket_name,          # Source bucket
                bucket_folder_name=self.data_ingestion_config.s3_data_folder # Folder inside the S3 bucket
            )

            # Logging successful exit from the method
            logging.info("Exited the get_data_from_s3 method of Data ingestion class")
        
        # Handling any exception that occurs during the S3 sync
        except Exception as e:
            # Raising a custom exception with system info
            raise XRayException(e, sys)

        
        # Method to initiate the entire data ingestion process and return an artifact
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        
        # Logging entry into the method
        logging.info("Entered the initiate_data_ingestion method of Data ingestion class")

        try:
            # Calling the method to download data from S3
            self.get_data_from_s3()

            # Creating an artifact object to store the paths to training and testing data
            data_ingestion_artifact: DataIngestionArtifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_data_path,
                test_file_path=self.data_ingestion_config.test_data_path,
            )

            # Logging successful exit from the method
            logging.info("Exited the initiate_data_ingestion method of Data ingestion class")

            # Returning the artifact object
            return data_ingestion_artifact
            
        # Handling exceptions during the data ingestion process
        except Exception as e:
            # Raising a custom exception
            raise XRayException(e, sys)

    
