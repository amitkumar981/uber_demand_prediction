import json
import mlflow
import os
from mlflow import MlflowClient
import dagshub
from pathlib import Path
mlflow.set_tracking_uri('https://dagshub.com/amitkumar981/uber_demand_prediction.mlflow')

dagshub.init(repo_owner='amitkumar981', repo_name='uber_demand_prediction', mlflow=True)

def model_load_information(file_path):
    with open(file_path,'rb') as f:
        run_info=json.load(f)
    return run_info


current_dir=Path(__file__).resolve()
print(current_dir)

 
root_path = current_dir.parent.parent
file_path =root_path/'run_information.json'
model_name=model_load_information(file_path=file_path)['model_name']



stage='staging'

client=MlflowClient()
latest_versions=client.get_latest_versions(name=model_name,stages=[stage])
model_latest_version=latest_versions[0].version

#promote model
promote_stage='production'

client.transition_model_version_stage(name=model_name,version=model_latest_version,
                                      stage=promote_stage,archive_existing_versions=True)