import sys
import os
from xray.exception import XRayException
from xray.pipeline.train_pipeline import TrainPipeline


def start_training():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()
        data_root = "artifacts/07_18_2025_21_44_02/data_ingestion/data"

        # List the contents to verify structure
        print(f"Contents of '{data_root}':", os.listdir(data_root))

        # Deep check
        for dirpath, dirnames, filenames in os.walk(data_root):
            print(f"Found dir: {dirpath}")
            for f in filenames:
                print(f"  - {f}")

    except Exception as e:
        raise XRayException(e, sys)

# C:\Users\ABHISHEK MAURYA\Chest-X-Ray-Medical- DL\artifacts\07_18_2025_21_44_02
if __name__ == "__main__":
    start_training()