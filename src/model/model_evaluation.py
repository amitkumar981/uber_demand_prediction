#import llbraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
import yaml
import logging
from pathlib import Path
import joblib
import json

import mlflow
mlflow.set_tracking_uri('https://dagshub.com/amitkumar981/uber_demand_prediction.mlflow')

import dagshub
dagshub.init(repo_owner='amitkumar981', repo_name='uber_demand_prediction', mlflow=True)


# configure logging
logger = logging.getLogger("model evaluation")
logger.setLevel(logging.DEBUG)

# configure console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# configure formatter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# add handler to logger
logger.addHandler(console_handler)

def load_data(data_type:str,data_path) ->pd.DataFrame:
    try:
        logger.info(f"loading {data_type} from: : {data_path}")
        df=pd.read_csv(data_path,parse_dates=['tpep_pickup_datetime']).set_index('tpep_pickup_datetime')
        logger.info(f"loaded data successfully from : {data_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"error in {data_type} loading from {data_path} : {e}")
        return None

def load_artifact(obj_name: str, object_path: str):
    try:
        logger.info(f"Loading {obj_name} from: {object_path}")
        model_obj = joblib.load(object_path)
        logger.info(f"{obj_name} loaded successfully from: {object_path}")
        return model_obj
    except Exception as e:
        logger.error(f"Error loading {obj_name} from {object_path}: {e}")
        return None

def model_evauation(train_data,test_data,model,scaler):
    #extrect x and y
    x_train=train_data.drop(columns=['ride_count'])
    y_train=train_data['ride_count']
    x_train_encoded=scaler.transform(x_train)
    y_train_pred=model.predict(x_train_encoded)

    x_test=test_data.drop(columns=['ride_count'])
    y_test=test_data['ride_count']
    x_test_encoded=scaler.transform(x_test)
    y_pred_test=model.predict(x_test_encoded)

    training_error=mean_absolute_percentage_error(y_train,y_train_pred)
    logger.info(f"training error calculated : {training_error}")
    test_error=mean_absolute_percentage_error(y_test,y_pred_test)
    logger.info(f"testing error calculated : {test_error}")


def save_model_info(save_json_path,run_id, artifact_path, model_name):
    info_dict = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_name": model_name
    }
    with open(save_json_path,"w") as f:
        json.dump(info_dict,f,indent=4)

def main():
    #set experiment
    mlflow.set_experiment('dvc-pipeline-run')

    with mlflow.start_run() as run:
         # set tags
        mlflow.set_tag("model","uber_demand_prediction")

        current_dir=Path(__file__).resolve()
        root_dir=current_dir.parent.parent.parent

        train_data_path=root_dir/"data/processed/training_df.csv"
        test_data_path=root_dir/"data/processed/testing_df.csv"
        

        train_df=load_data(data_type='training',data_path=train_data_path)
        test_df=load_data(data_type='testing',data_path=test_data_path)
       


        object_names = ['preprocessor', 'model']

        preprocessor_path = root_dir / f"src/model/{object_names[0]}.joblib"
        model_path = root_dir / f"src/model/{object_names[1]}.joblib"

        
        preprocessor=load_artifact('preprocessor',preprocessor_path)
        model=load_artifact('model',model_path)

        x_train=train_df.drop(columns=['ride_count'])
        y_train=train_df['ride_count']
        x_train_encoded=preprocessor.transform(x_train)
        y_train_pred=model.predict(x_train_encoded)

        x_test=test_df.drop(columns=['ride_count'])
        y_test=test_df['ride_count']
        x_test_encoded=preprocessor.transform(x_test)
        y_pred_test=model.predict(x_test_encoded)

        training_error=mean_absolute_percentage_error(y_train,y_train_pred)
        mlflow.log_metric('train_mape',training_error)
        logger.info(f"log training_error complete{training_error}")
        test_error=mean_absolute_percentage_error(y_test,y_pred_test)
        mlflow.log_metric('test_mape',test_error)
        logger.info(f"log testing_error complete{test_error}")

        # calculate cross val scores
        cv_scores = cross_val_score(model,
                                    x_train_encoded,
                                    y_train,
                                    cv=5,
                                    scoring="neg_mean_absolute_percentage_error",
                                    n_jobs=-1)
                
        # mean cross val score
        mean_cv_score = -(cv_scores.mean())
        mlflow.log_metric('mean_cv_score',-(mean_cv_score))
        logger.info(f"logging cv_score complete :{mean_cv_score}")

        mlflow.log_artifact(root_dir / f"src/model/{object_names[0]}.joblib")


        # mlflow dataset input datatype
        TARGET='ride_count'
        train_data_input = mlflow.data.from_pandas(train_df,targets=TARGET)
        test_data_input = mlflow.data.from_pandas(test_df,targets=TARGET)

        # log input
        mlflow.log_input(dataset=train_data_input,context="training")
        mlflow.log_input(dataset=test_data_input,context="validation")
        logger.info(f"dataset logging complete")
        
        # model signature
        sample_input = x_train.sample(20, random_state=42)
        sample_transformed = preprocessor.transform(sample_input)
        model_signature = mlflow.models.infer_signature(model_input=sample_transformed,
                                                    model_output=model.predict(sample_transformed))
        
         # log the final model
        mlflow.sklearn.log_model(model,"uber_demand_prediction",signature=model_signature)
        logger.info('logging model compplete')

        # save the model info
        save_json_path = root_dir/'run_information.json'
        # get the current run artifact uri
        artifact_uri = mlflow.get_artifact_uri()
         # get the run id 
        run_id = run.info.run_id
        model_name = "uber_demand_prediction"

        save_model_info(save_json_path=save_json_path,
                        run_id=run_id,
                        artifact_path=artifact_uri,
                        model_name=model_name)
        logger.info(f"model infomation saved")


if __name__=="__main__":
    main()




        
    
    

    
    
    

