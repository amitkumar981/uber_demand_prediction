import pytest
import mlflow
import dagshub
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn import set_config


set_config(transform_output="pandas")

# Init DagsHub
mlflow.set_tracking_uri('https://dagshub.com/amitkumar981/uber_demand_prediction.mlflow')

import dagshub
dagshub.init(repo_owner='amitkumar981', repo_name='uber_demand_prediction', mlflow=True)


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


# set model name
model_path = load_model_information("run_information.json")["model_name"]

# load the latest model from model registry


current_dir=Path(__file__).resolve()

root_path = current_dir.parent.parent

file_path =root_path/'run_information.json'
model_name=load_model_information(file_path=file_path)['model_name']

stage='Staging'

# load the model
model_path = f"models:/{model_name}/{stage}"
model = mlflow.sklearn.load_model(model_path)


# data_path
train_data_path = root_path / "data/processed/training_df.csv"
test_data_path = root_path / "data/processed/testing_df.csv"

# path for the encoder
encoder_path = root_path /"src/model/preprocessor.joblib"
encoder = joblib.load(encoder_path)

# build the model pipeline
model_pipe = Pipeline(steps=[
    ("encoder",encoder),
    ("regressor",model)
])

# test function
@pytest.mark.parametrize(argnames="data_path,threshold",
                         argvalues=[(train_data_path,0.2),
                                    (test_data_path,0.2)])
def test_performance(data_path, threshold):
    # load the data from path
    data = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime")
    # make X and y
    X = data.drop(columns=["ride_count"])
    y = data["ride_count"]
    # do predictions
    y_pred = model_pipe.predict(X)
    # calculate the loss
    loss = mean_absolute_percentage_error(y, y_pred)
    # check the performance
    assert loss <= threshold,  f"The model does not pass the performance threshold of {threshold}"