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


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.color_jitter_transforms: dict = {
            "brightness": BRIGHTNESS,
            "contrast": CONTRAST,
            "saturation": SATURATION,
            "hue": HUE,
        }

        self.RESIZE: int = RESIZE

        self.CENTERCROP: int = CENTERCROP

        self.RANDOMROTATION: int = RANDOMROTATION

        self.normalize_transforms: dict = {
            "mean": NORMALIZE_LIST_1,
            "std": NORMALIZE_LIST_2,
        }

        self.data_loader_params: dict = {
            "batch_size": BATCH_SIZE,
            "shuffle": SHUFFLE,
            "pin_memory": PIN_MEMORY,
        }

        self.artifact_dir: str = os.path.join(
            ARTIFACT_DIR, TIMESTAMP, "data_transformation"
        )

        self.train_transforms_file: str = os.path.join(
            self.artifact_dir, TRAIN_TRANSFORMS_FILE
        )

        self.test_transforms_file: str = os.path.join(
            self.artifact_dir, TEST_TRANSFORMS_FILE
        )
        
        
        
        
@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.artifact_dir: int = os.path.join(ARTIFACT_DIR, TIMESTAMP, "model_training")

        self.trained_bentoml_model_name: str = "xray_model"

        self.trained_model_path: int = os.path.join(
            self.artifact_dir, TRAINED_MODEL_NAME
        )

        self.train_transforms_key: str = TRAIN_TRANSFORMS_KEY

        self.epochs: int = EPOCH

        self.optimizer_params: dict = {"lr": 0.01, "momentum": 0.8}

        self.scheduler_params: dict = {"step_size": STEP_SIZE, "gamma": GAMMA}

        self.device: device = DEVICE
        
