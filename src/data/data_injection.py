#importing libraries
import numpy as np
import dask.dataframe as dd
import logging
from pathlib import Path
import path

#configure logging

#configure logging
logger=logging.getLogger('data_injection')
logger.setLevel(logging.DEBUG)

#configure console handler
file_handler=logging.StreamHandler()
file_handler.setLevel(logging.DEBUG)

#add handler to logger
logger.addHandler(file_handler)

#configure formatter
formatter=logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

max_fare_amount_value=52
max_trip_distance=18.56
min_fare_amount_value=0.50
min_trip_distance=0.25
min_latitude = 40.60
max_latitude = 40.85
min_longitude = -74.05
max_longitude = -73.70

def load_data(data_paths: list[str]):
    try:
        logger.info(f"Loading data from {len(data_paths)} files")
        dataframes = []
        for path in data_paths:
            logger.info(f"Reading: {path}")
            cols = [
                'passenger_count', 'trip_distance', 'pickup_longitude','pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                'fare_amount', 'total_amount','tpep_pickup_datetime']
            
            df = dd.read_csv(path,usecols=cols,assume_missing=True,parse_dates=['tpep_pickup_datetime'])
            dataframes.append(df)
        combined_df = dd.concat(dataframes, axis=0)
        logger.info("Merged all data successfully")
        return combined_df

    except Exception as e:
        logger.error(f"Error in loading at {path}: {e}")
        return None

def remove_outlier(data: dd.DataFrame) -> dd.DataFrame:
    try:
        logger.info("Starting outlier removal...")

        # Apply all filters and assign the result
        filtered_data = data[
            (data['fare_amount'].between(min_fare_amount_value, max_fare_amount_value, inclusive='both')) &
            (data['trip_distance'].between(min_trip_distance, max_trip_distance, inclusive='both')) &
            (data['pickup_latitude'].between(min_latitude, max_latitude, inclusive='both')) &
            (data['pickup_longitude'].between(min_longitude, max_longitude, inclusive='both')) &
            (data['dropoff_latitude'].between(min_latitude, max_latitude, inclusive='both')) &
            (data['dropoff_longitude'].between(min_longitude, max_longitude, inclusive='both'))
        ]
        logger.info(f"complete outlier removel successfully")

        logger.info(f"droping columns....")

        cols_to_drop = [
            'passenger_count', 'trip_distance', 'dropoff_longitude',
            'dropoff_latitude', 'fare_amount', 'total_amount'
        ]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        logger.info(f"drop columns successfully")

        logger.info(f"intializing computing data...")
        filtered_data=filtered_data.compute()
        logger.info(f"compute data successfully with sape{filtered_data.shape}")

        return filtered_data

    except Exception as e:
        logger.error(f"Error in removing outliers: {e}")
        return data  
    
def save_data(data,save_path):
    try:
        logger.info(f"saving data at{save_path}")
        data.to_csv(save_path,index=True)
        logger.info(f"saved data successfully at {save_path}")
    except Exception as e:
        logger.error(f"error in saving data :{e}")

def main():
    current_dir = Path(__file__).resolve()
    root_dir = current_dir.parent.parent.parent
    print(root_dir)
    data_names=['yellow_tripdata_2016-01.csv',
                 'yellow_tripdata_2016-02.csv',
                 'yellow_tripdata_2016-03.csv']
    
    data_paths=[root_dir/"data"/"raw"/name for name in data_names]

    combined_df=load_data(data_paths)

    final_df=remove_outlier(combined_df)

    save_path=root_dir/'data'/'interim'/'df_without_outlier.csv'
    save_data(final_df,save_path)
    print(final_df.columns)

if __name__=="__main__":
     main()

     
            
        



    



