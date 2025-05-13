# Use official Python image as the base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Optional: Configure Git to use the token for DagsHub
RUN git config --global credential.helper store

# Copy requirements and install dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy project files
COPY ./data/external/plot_data.csv ./data/external/plot_data.csv
COPY ./data/processed/testing_df.csv ./data/processed/testing_df.csv
COPY ./src/model/ ./src/model/
COPY ./app.py .

# Expose Streamlit port
EXPOSE 8000

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]

