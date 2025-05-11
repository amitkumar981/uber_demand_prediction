import numpy as np
import pandas as pd
import mlflow
from mlflow import MlflowClient
import json
from pathlib import Path

import logging

import mlflow
mlflow.set_tracking_uri('https://dagshub.com/amitkumar981/uber_demand_prediction.mlflow')

import dagshub
dagshub.init(repo_owner='amitkumar981', repo_name='uber_demand_prediction', mlflow=True)

# configure logging
logger = logging.getLogger("register_model")
logger.setLevel(logging.DEBUG)

# configure console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# configure formatter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# add handler to logger
logger.addHandler(console_handler)

def load_model_info(save_info_path:str):
    try:
        logger.info(f"load model info form {save_info_path}")
        with open(save_info_path,'r') as f:
            model_info=json.load(f)
        logger.info(f"model info loaded successfully")
        return model_info
    except Exception as e:
        logger.info(f"error in loading model info from {save_info_path} as {e}")


def main():
    current_dir=Path(__file__).resolve()
    root_dir=current_dir.parent.parent.parent

    save_info_path=root_dir/'run_information.json'

    model_info=load_model_info(save_info_path)

    model_name=model_info['model_name']
    logger.info(f"fatch model name :{model_name} successfully")
    run_id=model_info['run_id']
    logger.info(f"fatch run_id :{run_id} successfully")

     #model  to register path
    model_registry_path = f"runs:/{run_id}/{model_name}"

    model_version=mlflow.register_model(model_uri=model_registry_path,name=model_name)

    #get the model version
    registered_model_version=model_version.version
    registered_model_name=model_version.name
    logger.info(f"The latest model version in model registry is {registered_model_version}")

    #update the model version
    client=MlflowClient()
    client.transition_model_version_stage(name=registered_model_name,version=registered_model_version,
                                          stage='staging',archive_existing_versions=False)
    
    logger.info(f"update model stage successfully")

    
                                          



if __name__=="__main__":
 main()

 