from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import requests
import mlflow
from mlflow import MlflowClient
import os

# Define the MLflow client
mlflow.set_tracking_uri("http://ml-service:5102")
client = MlflowClient()

# Set the default model name
model_name = "Titanic_Disaster_Model"


csv_file_path = "/opt/airflow/data/train.csv"

def train_model():
    with open(csv_file_path, 'rb') as f:
        files = {'file': ('train.csv', f)}
        response = requests.post("http://ml-service:8080/model/train", files=files)
        print(response.status_code, response.text)

    # Check if the request was successful
    if response.status_code != 200:
        data = response.json()
        raise Exception(f"Training failed: {data.get('error', 'Unknown error')}")
    
    # Parse response data
    data = response.json()
    new_accuracy = data["test_accuracy"]

    # Retrieve the latest staging model's test accuracy
    latest_version_info = client.get_latest_versions(model_name, stages=["Staging"])
    if latest_version_info:
        latest_version = latest_version_info[0]
        staging_accuracy = client.get_metric_history(latest_version.run_id, "test_accuracy")[-1].value

        # If new model is better or equal, transition it to Staging
        if new_accuracy >= staging_accuracy:
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage="Archived"
            )
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage="Staging"
            )
        else:
            return False  # Indicates that the new model's accuracy is lower
    print("Model training and evaluation completed successfully.")
    return True

# Define the Airflow DAG
with DAG(
    dag_id="daily_model_training",
    start_date=datetime(2025, 10, 26),
    schedule="0 18 * * *",  # Runs daily at 2 AM
    catchup=False,
) as dag:

    # Define a task that calls the train_model function
    train_and_compare_task = PythonOperator(
        task_id="train_and_compare_model",
        python_callable=train_model,
        retries=3,
        retry_delay=timedelta(minutes=5),
    )
    
    notification_task = BashOperator(
        task_id="show_notification",
        bash_command='echo "The new model\'s accuracy is equal to or lower than the old model\'s accuracy."',
        dag=dag
    )

    # Set task dependencies
    train_and_compare_task >> notification_task

    # Run notification only if train_and_compare_task returns False
    notification_task.trigger_rule = 'all_done'

