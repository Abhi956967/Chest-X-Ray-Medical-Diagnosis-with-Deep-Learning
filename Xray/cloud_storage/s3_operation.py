# Importing required modules
import os   # Module to interact with the operating system (used here to run AWS CLI commands)
import sys  # Provides access to system-specific parameters and functions, useful for exception handling

# Importing custom exception class from the project
from Xray.exception import XRayException  # Custom exception to handle and format errors consistently across the project

# Define a class for handling AWS S3 operations like syncing files
class S3Operation:

    # Method to upload (sync) a local folder to an S3 bucket
    def sync_folder_to_s3(self, folder: str, bucket_name: str, bucket_folder_name: str) -> None:
        """
        Upload local folder to S3 bucket at specified folder path.
        folder: Local directory to be uploaded
        bucket_name: Name of the S3 bucket
        bucket_folder_name: Target folder in the S3 bucket where files will be uploaded
        """
        try:
            # Construct the AWS CLI command to sync local folder to S3 bucket
            command: str = (
                f"aws s3 sync {folder} s3://{bucket_name}/{bucket_folder_name}/"
            )

            # Run the constructed command in the terminal
            os.system(command)

        except Exception as e:
            # If an error occurs, raise a custom exception with traceback info
            raise XRayException(e, sys)

    # Method to download (sync) a folder from S3 bucket to local directory
    def sync_folder_from_s3(self, folder: str, bucket_name: str, bucket_folder_name: str) -> None:
        """
        Download folder from S3 bucket to local folder.
        folder: Local directory where data from S3 will be downloaded
        bucket_name: Name of the S3 bucket
        bucket_folder_name: Folder in S3 bucket to be synced to local
        """
        try:
            # Construct the AWS CLI command to sync from S3 bucket to local folder
            command: str = (
                f"aws s3 sync s3://{bucket_name}/{bucket_folder_name}/ {folder}/"
            )

            # Run the command in the terminal
            os.system(command)

        except Exception as e:
            # Raise a custom exception in case of failure
            raise XRayException(e, sys)
# End of the S3Operation class



         