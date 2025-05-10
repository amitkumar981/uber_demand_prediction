import numpy as np
import pandas as pd
import logging
from pathlib import Path

# configure logging
logger = logging.getLogger('feature processing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

def load_data(data_path: str) -> pd.DataFrame:
    try:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully from {data_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        return None

def preprocessing_data(data: pd.DataFrame):
    try:
        data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
        logger.info("Converted 'tpep_pickup_datetime' to datetime")

        data['day_of_week'] = data['tpep_pickup_datetime'].dt.day_of_week
        logger.info("Extracted day_of_week from datetime")

        data['month'] = data['tpep_pickup_datetime'].dt.month
        logger.info("Extracted month from datetime")

        data.set_index('tpep_pickup_datetime', inplace=True)
        logger.info("Set 'tpep_pickup_datetime' as index")

        region_group = data.groupby('region')
        for p in range(1, 6):
            data[f"lag_{p}"] = region_group['ride_count'].shift(p)
        logger.info(f"Added lag features: {list(data.columns)}")

        missing_values = data.isnull().sum().sum()
        logger.info(f"Total missing values before drop: {missing_values}")

        cleaned_df = data.dropna()
        logger.info(f"Missing values after drop: {cleaned_df.isnull().sum().sum()}")
        logger.info(f"Data preprocessing complete with shape {cleaned_df.shape}")

        return cleaned_df
    except Exception as e:
        logger.error(f"Error in preprocessing data: {e}")
        return None

def split_data(data: pd.DataFrame):
    try:
        logger.info("Splitting data...")
        training_df = data.loc[data['month'].isin([1, 2])]
        testing_df = data.loc[data['month'] == 3]

        logger.info(f"Training set shape before drop: {training_df.shape}")
        logger.info(f"Testing set shape before drop: {testing_df.shape}")

        training_df = training_df.drop('month', axis=1)
        testing_df = testing_df.drop('month', axis=1)

        logger.info(f"Training set shape after drop: {training_df.shape}")
        logger.info(f"Testing set shape after drop: {testing_df.shape}")

        return training_df, testing_df
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        return None, None

def save_data(data: pd.DataFrame, save_path: Path):
    try:
        data.to_csv(save_path, index=True)
        logger.info(f"Data saved at {save_path}")
    except Exception as e:
        logger.error(f"Error saving data to {save_path}: {e}")

def main():
    current_dir = Path(__file__).resolve()
    root_dir = current_dir.parent.parent.parent
    data_path = root_dir / "data/processed/resampled_df.csv"

    df = load_data(data_path)
    if df is None:
        logger.error("Loading data failed. Exiting.")
        return

    cleaned_df = preprocessing_data(df)
    if cleaned_df is None:
        logger.error("Preprocessing failed. Exiting.")
        return

    training_df, testing_df = split_data(cleaned_df)
    if training_df is None or testing_df is None:
        logger.error("Splitting failed. Exiting.")
        return

    save_data(training_df, root_dir / 'data' / 'processed' / 'training_df.csv')
    save_data(testing_df, root_dir / 'data' / 'processed' / 'testing_df.csv')

if __name__ == "__main__":
    main()

    

        
    





