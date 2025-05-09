# importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
import yaml
import logging
import joblib

# configure logging
logger = logging.getLogger('feature extraction')
logger.setLevel(logging.DEBUG)

# configure console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# configure formatter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# add handler to logger
logger.addHandler(console_handler)

# Load data
def load_data(data_path: str) -> pd.io.parsers.TextFileReader:
    try:
        logger.info(f"loading data at: {data_path}")
        df_reader = pd.read_csv(data_path, chunksize=100000, usecols=["pickup_latitude", "pickup_longitude"])
        logger.info(f"data loaded successfully from: {data_path}")
        return df_reader
    except Exception as e:
        logger.error(f"error in loading data from {data_path}: {e}")
        return None

# Load parameters from YAML
def load_params(param_path: str):
    try:
        logger.info("initializing loading params")
        with open(param_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.debug(f"KMeans config: {params['feature_extraction']['KMeans']}")
        return params
    except Exception as e:
        logger.error(f"error in loading params: {e}")
        return None

# Train the scaler using partial fit
def fit_scaler(data_path):
    df_reader = load_data(data_path)
    if df_reader is None:
        logger.error("Data loading failed, aborting scaler training")
        return None
    try:
        logger.info("start scaler training")
        scaler = StandardScaler()
        for chunk in df_reader:
            scaler.partial_fit(chunk)
        logger.info("scaler training completed")
        return scaler
    except Exception as e:
        logger.error(f"error in scaler training: {e}")
        return None

# Train the clustering model using partial fit
def train_model(data_path, scaler, kmeans_params):
    df_reader = load_data(data_path)
    if df_reader is None:
        logger.error("Data loading failed, aborting model training")
        return None
    try:
        kmeans = MiniBatchKMeans(**kmeans_params)
        logger.info(f"start model training with {kmeans}")
        for chunk in df_reader:
            scaled_chunk = scaler.transform(chunk)
            kmeans.partial_fit(scaled_chunk)
        logger.info("model training completed")
        return kmeans
    except Exception as e:
        logger.error(f"error in model training: {e}")
        return None

# Save model using joblib
def save_model(obj, path: str, name: str):
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        full_path = Path(path) / f"{name}.joblib"
        joblib.dump(obj, full_path)
        logger.info(f"{name} saved at {full_path} successfully")
    except Exception as e:
        logger.error(f"error in saving {name}: {e}")

# Make predictions on full dataset
def make_prediction(model, scaler, data_path):
    try:
        logger.info(f"loading data from {data_path} for prediction")
        df_final = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
        location_subset = df_final.loc[:,["pickup_longitude","pickup_latitude"]]
        logger.info(f"loaded data successfully from {data_path}")
        scaled_data = scaler.transform(location_subset)
        logger.info("scaling completed")
        logger.info("start making prediction ...")
        df_final['region'] = model.predict(scaled_data)
        predicted_df = df_final.set_index('tpep_pickup_datetime')
        logger.info(f"added 'region' as new column in dataset {predicted_df.shape}")
        return predicted_df
    except Exception as e:
        logger.error(f"error in making prediction: {e}")
        return None

# Resample data and compute exponential moving average
def resample_data(data, alpha):
    try:
        logger.info(f"start resampling data with data shape {data.shape}")
        data = data.drop(columns=["pickup_latitude", "pickup_longitude"])
        logger.info(f"data shape after dropping columns: {data.shape}")
        data = data.groupby('region').resample('15min').size().reset_index(name='ride_count')
        logger.info(f"resampled_data shape after resampling {data.shape}")
        zero_count_before = (data['ride_count'] == 0).sum()
        logger.info(f"Ride count with zero before epsilon: {zero_count_before}")
        epsilon = 10
        data['ride_count'] = data['ride_count'].replace(0, epsilon)
        zero_count_after = (data['ride_count'] == 0).sum()
        logger.info(f"Ride count with zero after epsilon: {zero_count_after}")
        data['avg_ride'] = data['ride_count'].ewm(alpha=alpha).mean().round()
        logger.info(f"added avg_ride column successfully")
        logger.info(f"resampling completed with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"error in resampling data: {e}")
        return None

def save_data(data, save_path):
    try:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"saving data...")
        data.to_csv(save_path, index=False)
        logger.info(f"data saved at: {save_path}")
    except Exception as e:
        logger.error(f"error in saving data: {e}")

def main():
    try:
        try:
            current_dir = Path(__file__).resolve()
        except NameError:
            current_dir = Path.cwd()
        root_dir = current_dir.parent.parent.parent

        data_path = root_dir / "data/interim/df_without_outlier.csv"
        param_path = root_dir / "params.yaml"

        params = load_params(param_path)
        if params is None:
            logger.error("Parameters loading failed, aborting pipeline")
            return
        kmeans_params = params['feature_extraction']['KMeans']
        alpha = params['feature_extraction']['resample_data']['alpha']

        scaler = fit_scaler(data_path)
        if scaler is None:
            logger.error("Scaler training failed, aborting pipeline")
            return

        kmeans = train_model(data_path, scaler, kmeans_params)
        if kmeans is None:
            logger.error("Model training failed, aborting pipeline")
            return

        save_model(obj=scaler, path=root_dir / 'src/model', name='scaler')
        save_model(obj=kmeans, path=root_dir / 'src/model', name='kmeans')

        predicted_df = make_prediction(model=kmeans, scaler=scaler, data_path=data_path)
        if predicted_df is None:
            logger.error("Prediction failed, aborting pipeline")
            return

        resampled_df = resample_data(predicted_df, alpha=alpha)
        if resampled_df is None:
            logger.error("Resampling failed, aborting pipeline")
            return

        save_path = root_dir / "data/processed/resampled_df.csv"
        save_data(resampled_df, save_path)

    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")

if __name__ == "__main__":
    main()
