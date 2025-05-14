import joblib
import dagshub
import mlflow
import pandas as pd
import streamlit as st
from pathlib import Path
import datetime as dt
from sklearn.pipeline import Pipeline
from sklearn import set_config
from time import sleep
import operator  # For itemgetter

set_config(transform_output="pandas")

mlflow.set_tracking_uri('https://dagshub.com/amitkumar981/uber_demand_prediction.mlflow')
dagshub.init(repo_owner='amitkumar981', repo_name='uber_demand_prediction', mlflow=True)

# get model name
registered_model_name = 'uber_demand_prediction'
stage = "Production"
model_path = f"models:/{registered_model_name}/{stage}"

# load the latest model from model registry
model = mlflow.sklearn.load_model(model_path)

root_dir = Path(__file__).parent

# Data and model paths
plot_data_path = root_dir / "data/external/plot_data.csv"
data_path = root_dir / "data/processed/testing_df.csv"
kmeans_path = root_dir / "src/model/kmeans.joblib"
scaler_path = root_dir / "src/model/scaler.joblib"
encoder_path = root_dir / "src/model/preprocessor.joblib"
#model_path = root_dir / "src/model/model.joblib"

# Load model objects
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)
#model = joblib.load(model_path)
kmeans = joblib.load(kmeans_path)

# Load datasets
df_plot = pd.read_csv(plot_data_path)
df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime")

# App UI
st.title("Uber Demand in New York City 🚕🌆")
st.sidebar.title("Options")

map_type = st.sidebar.radio(label="Select the type of Map",
                            options=["Complete NYC Map", "Only for Neighborhood Regions"],
                            index=1)

st.subheader("Date")
date = st.date_input("Select the date", value=None,
                     min_value=dt.date(2016, 3, 1),
                     max_value=dt.date(2016, 3, 31))
st.write("**Date:**", date)

st.subheader("Time")
time = st.time_input("Select the time", value=None)
st.write("**Current Time:**", time)

if date and time:
    delta = dt.timedelta(minutes=15)
    next_interval = dt.datetime(year=date.year,
                                 month=date.month,
                                 day=date.day,
                                 hour=time.hour,
                                 minute=time.minute) + delta
    st.write("Demand for Time: ", next_interval.time())

    index = pd.Timestamp(f"{date} {next_interval.time()}")
    st.write("**Date & Time:**", index)

    st.subheader("Location")
    sample_loc = df_plot.sample(1).reset_index(drop=True)
    lat = sample_loc["pickup_latitude"].item()
    long = sample_loc["pickup_longitude"].item()
    region = sample_loc["region"].item()
    st.write("**Your Current Location**")
    st.write(f"Lat: {lat}")
    st.write(f"Long: {long}")

    with st.spinner("Fetching your Current Region"):
        sleep(3)

    st.write("Region ID: ", region)

    scaled_cord = scaler.transform(sample_loc.iloc[:, 0:2])

    st.subheader("MAP")

    colors = [
        "#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#ADFF2F", "#32CD32", "#008000", "#006400",
        "#00FF00", "#7CFC00", "#00FA9A", "#00FFFF", "#40E0D0", "#4682B4", "#1E90FF", "#0000FF",
        "#0000CD", "#8A2BE2", "#9932CC", "#BA55D3", "#FF00FF", "#FF1493", "#C71585", "#FF4500",
        "#FF6347", "#FFA07A", "#FFDAB9", "#FFE4B5", "#F5DEB3", "#EEE8AA", "#800000", "#B22222",
        "#DC143C", "#FA8072", "#E9967A", "#F08080", "#CD5C5C", "#8B0000", "#A52A2A", "#B8860B",
        "#DAA520", "#BDB76B", "#556B2F", "#6B8E23", "#228B22", "#2E8B57", "#3CB371", "#20B2AA",
        "#5F9EA0", "#7B68EE"
    ]

    region_colors = {region: colors[i] for i, region in enumerate(df_plot["region"].unique().tolist())}
    df_plot["color"] = df_plot["region"].map(region_colors)

    pipe = Pipeline([
        ('encoder', encoder),
        ('reg', model)
    ])

    if map_type == "Complete NYC Map":
        progress_bar = st.progress(value=0, text="Operation in progress. Please wait.")
        for percent_complete in range(100):
            sleep(0.05)
            progress_bar.progress(percent_complete + 1, text="Operation in progress. Please wait.")
        progress_bar.empty()

        st.map(data=df_plot, latitude="pickup_latitude",
               longitude="pickup_longitude", size=0.01,
               color="color")

        input_data = df.loc[index, :].sort_values("region")
        target = input_data["ride_count"]
        predictions = pipe.predict(input_data.drop(columns=["ride_count"]))

        st.markdown("### Map Legend")
        for i in range(len(predictions)):
            region_id = input_data.iloc[i]["region"]
            color = colors[int(region_id)]
            demand = predictions[i]
            region_label = f"{region_id} (Current region)" if region == region_id else region_id
            st.markdown(
                f'<div style="display: flex; align-items: center;">'
                f'<div style="background-color:{color}; width: 20px; height: 10px; margin-right: 10px;"></div>'
                f'Region ID: {region_label} <br>'
                f"Demand: {int(demand)} <br><br>", unsafe_allow_html=True
            )

    elif map_type == "Only for Neighborhood Regions":
        distances = kmeans.transform(scaled_cord).to_numpy().ravel().tolist()
        distances = list(enumerate(distances))
        sorted_distances = sorted(distances, key=operator.itemgetter(1))[:9]
        indexes = sorted([ind[0] for ind in sorted_distances])

        df_plot_filtered = df_plot[df_plot["region"].isin(indexes)]

        progress_bar = st.progress(value=0, text="Operation in progress. Please wait.")
        for percent_complete in range(100):
            sleep(0.05)
            progress_bar.progress(percent_complete + 1, text="Operation in progress. Please wait.")
        progress_bar.empty()

        st.map(data=df_plot_filtered, latitude="pickup_latitude",
               longitude="pickup_longitude", size=0.01,
               color="color")

        input_data = df.loc[index, :]
        input_data = input_data.loc[input_data["region"].isin(indexes)].sort_values("region")
        target = input_data["ride_count"]
        predictions = pipe.predict(input_data.drop(columns=["ride_count"]))

        st.markdown("### Map Legend")
        for i in range(len(predictions)):
            region_idx = input_data.iloc[i]["region"]
            color = colors[int(region_idx)]
            demand = predictions[i]
            region_label = f"{region_idx} (Current region)" if region == region_idx else region_idx
            st.markdown(
                f'<div style="display: flex; align-items: center;">'
                f'<div style="background-color:{color}; width: 20px; height: 10px; margin-right: 10px;"></div>'
                f'Region ID: {region_label} <br>'
                f"Demand: {int(demand)} <br><br>", unsafe_allow_html=True
            )


