#import llbraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import yaml
import logging
from pathlib import Path
import joblib


# configure logging
logger = logging.getLogger('model_training')
logger.setLevel(logging.DEBUG)

# configure console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# configure formatter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# add handler to logger
logger.addHandler(console_handler)

def load_data(data_path:str) ->pd.DataFrame:
    try:
        logger.info(f"loading data from : {data_path}")
        df=pd.read_csv(data_path,parse_dates=['tpep_pickup_datetime']).set_index('tpep_pickup_datetime')
        logger.info(f"loaded data successfully from : {data_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"error in data loading from {data_path} : {e}")
        return None

def split_x_and_y(data:pd.DataFrame):
    try:
        logger.info(f"spliting x_train,y_train")
        x_train=data.drop(columns=['ride_count'])
        logger.info(f"split x_train complete with shape : {x_train.shape}")
        y_train=data['ride_count']
        logger.info(f"split y_train complete with shape {y_train.shape}")
        return x_train,y_train
    except Exception as e:
        logger.error(f"error in spliting at {e}")
        return None

def train_preprocessor(data):
    try:
        logger.info(f"build preprocessor...")
        cat_cols=['region','day_of_week']
        preprocessor=ColumnTransformer(transformers=[
            ('OHE',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False),cat_cols)
        ],remainder='passthrough',verbose_feature_names_out=True)

        preprocessor.fit(data)
        logger.info(f" training preprocessor complete")
        return preprocessor
    except Exception as e:
        logger.error(f"error in bulding processor")
        return None

def train_model(x,y):
    logger.info(f"start model train")
    lr=LinearRegression()
    lr.fit(x,y)
    logger.info(f"model training complete")
    return lr

# Save model using joblib
def save_model(obj, path: str, name: str):
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        full_path = Path(path) / f"{name}.joblib"
        joblib.dump(obj, full_path)
        logger.info(f"{name} saved at {full_path} successfully")
    except Exception as e:
        logger.error(f"error in saving {name}: {e}")
        

def main():
    current_dir=Path(__file__).resolve()
    root_dir=current_dir.parent.parent.parent
    
    data_path=root_dir/"data/processed/training_df.csv"

    df=load_data(data_path)

    x_train,y_train=split_x_and_y(df)

    preprocessor=train_preprocessor(x_train)

    x_train_encoded=preprocessor.transform(x_train)
    logger.info(f"transformed x_train complete with shape{x_train_encoded.shape}")

    model=train_model(x=x_train_encoded,y=y_train)

    save_model(obj=preprocessor, path=root_dir / 'src/model', name='preprocessor')
    save_model(obj=model, path=root_dir / 'src/model', name='model')


if __name__=="__main__":
    main()


