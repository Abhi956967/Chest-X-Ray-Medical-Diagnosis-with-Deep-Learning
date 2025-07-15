# Importing the os module to work with file and directory paths
import os      

# Importing dataclass from dataclasses module to create simple data classes
from dataclasses import dataclass

# Importing the torch.device class (although not used in this code snippet)
from torch import device

# Importing constants from the training pipeline config (like paths, bucket name, etc.)
from Xray.constant.training_pipeline import *




# Using @dataclass for cleaner data container class (though custom __init__ is defined, so it's not fully utilized)
@dataclass
class DataIngestionConfig:
    
    # Constructor to initialize configuration values for data ingestion
    def __init__(self):
        # Setting the S3 folder name from the constant
        self.s3_data_folder: str = S3_DATA_FOLDER
        
        # Setting the S3 bucket name from the constant
        self.bucket_name: str = BUCKET_NAME
        
        # Creating a root artifact directory path using a base path and a timestamp (for versioning)
        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
        
        # Defining the local directory where data from S3 will be downloaded
        # Structure: <artifact_dir>/data_ingestion/<s3_data_folder>
        self.data_path: str = os.path.join(
            self.artifact_dir, "data_ingestion", self.s3_data_folder
        )
        
        # Defining the path where training data will be stored after ingestion
        self.train_data_path: str = os.path.join(self.data_path, "train")

        # Defining the path where test data will be stored after ingestion
        # NOTE: This line returns a tuple because of the trailing comma — needs correction.
        self.test_data_path: str = os.path.join(self.data_path, "test")

        
