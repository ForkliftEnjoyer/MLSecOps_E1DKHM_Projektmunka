import pickle
import pandas as pd
import numpy as np
from flask import jsonify
from constants import NOMINAL_COLUMNS, DISCRETE_COLUMNS, CONTINOUS_COLUMNS, TEXT_COLUMNS, BINARY_COLUMNS
from scipy import stats
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import mlflow
from mlflow.artifacts import download_artifacts
import json

class MLModel:
    def __init__(self, client):
        """
        Initialize the MLModel with the given MLflow client and 
        load the staging model if available.

        Parameters:
            client (MlflowClient): The MLflow client used to 
            interact with the MLflow registry.

        Attributes:
            model (object): The loaded model, or None if no model 
                is loaded.
            fill_values_nominal (dict): Dictionary of fill values 
                for nominal columns.
            fill_values_discrete (dict): Dictionary of fill values 
                for discrete columns.
            fill_values_continuous (dict): Dictionary of fill values 
                for continuous columns.
            min_max_scaler_dict (dict): Dictionary of MinMaxScaler objects 
                for continuous columns.
            onehot_encoders (dict): Dictionary of OneHotEncoder objects 
                for nominal columns.
        """
        self.client = client
        self.model = None
        self.fill_values_nominal = None
        self.fill_values_discrete = None
        self.fill_values_continuous = None
        self.min_max_scaler_dict = None
        self.onehot_encoders = None
        self.load_staging_model()

    def load_staging_model(self):
        """
        Load the latest model tagged with 'Staging' stage from MLflow 
        if available.
        
        If a model with the 'Staging' tag exists, it loads the model 
        and associated artifacts. Otherwise, prints a warning.

        Returns:
            None
        """
        try:
            latest_staging_model = None
            for model in self.client.search_registered_models():
                for latest_version in model.latest_versions:
                    if latest_version.current_stage == "Staging":
                        latest_staging_model = latest_version
                        break
                if latest_staging_model:
                    break
            
            if latest_staging_model:
                model_uri = latest_staging_model.source
                self.model = mlflow.sklearn.load_model(model_uri)
                print("Staging model loaded successfully.")
                
                # Load associated artifacts
                artifact_uri = latest_staging_model.source.rpartition('/')[0]
                self.load_artifacts(artifact_uri)
            else:
                print("No staging model found.")
                
        except Exception as e:
            print(f"Error loading model or artifacts: {e}")

    def load_artifacts(self, artifact_uri):
        """
        Load necessary artifacts (e.g., scalers, encoders) from the given 
        artifact URI.

        Parameters:
            artifact_uri (str): The URI of the artifact directory containing 
            necessary files.

        Returns:
            None
        """
        try:
            # Load nominal fill values
            nominal_path = download_artifacts(artifact_uri=f"""
                    {artifact_uri}/fill_values_nominal.json""")
            with open(nominal_path, 'r') as f:
                self.fill_values_nominal = json.load(f)

            # Load discrete fill values
            discrete_path = download_artifacts(artifact_uri=f"""
                    {artifact_uri}/fill_values_discrete.json""")
            with open(discrete_path, 'r') as f:
                self.fill_values_discrete = json.load(f)

            # Load continuous fill values
            continuous_path = download_artifacts(artifact_uri=f"""
                    {artifact_uri}/fill_values_continuous.json""")
            with open(continuous_path, 'r') as f:
                self.fill_values_continuous = json.load(f)

            # Load MinMaxScaler dictionary
            scaler_path = download_artifacts(artifact_uri=f"""
                    {artifact_uri}/min_max_scaler_dict.pkl""")
            with open(scaler_path, 'rb') as f:
                self.min_max_scaler_dict = pickle.load(f)

            # Load OneHotEncoders
            encoders_path = download_artifacts(artifact_uri=f"""
                    {artifact_uri}/onehot_encoders.pkl""")
            with open(encoders_path, 'rb') as f:
                self.onehot_encoders = pickle.load(f)

            print("Artifacts loaded successfully.")

        except Exception as e:
            print(f"Error loading artifacts: {e}")

    def predict(self, inference_row):
        """
        Make a prediction using the preloaded staging model.

        Parameters:
            inference_row (list): A list of values representing a 
            single row of data for prediction.

        Returns:
            int: Predicted class label.
        """
        if self.model is None:
            return {'error': 'No staging model is loaded'}, 400

        processed_data = self.preprocessing_pipeline_inference(inference_row)
        prediction = self.model.predict(processed_data)
        return int(prediction)
    
    def preprocessing_pipeline(self, df):
        """Preprocess the data to handle missing values,
        create new features, encode categorical features, 
        and normalize the data using min max scaling.
        Returns the preprocessed dataframe.
        
        Keyword arguments:
        df -- DataFrame with the data

        Returns:
        df -- DataFrame with the preprocessed data
        """

        folder = 'artifacts/encoders'
        MLModel.create_new_folder(folder)

        folder = 'artifacts/preprocessed_data'
        MLModel.create_new_folder(folder)

        folder = 'artifacts/models'
        MLModel.create_new_folder(folder)

        folder = 'artifacts/nan_outlier_handler'
        MLModel.create_new_folder(folder)
        
        df = df.replace('?', np.nan)

        df = MLModel.extract_features(df, "Ticket")

        for col in CONTINOUS_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        self.fill_values_nominal = {col: df[col].mode()[0] for col in NOMINAL_COLUMNS}
        self.fill_values_discrete = {col: df[col].median() for col in DISCRETE_COLUMNS}
        self.fill_values_continuous = {col: df[col].mean(skipna=True) for col in CONTINOUS_COLUMNS}


        for col in NOMINAL_COLUMNS:
            df[col].fillna(self.fill_values_nominal[col], inplace=True)

        for col in DISCRETE_COLUMNS:
            df[col].fillna(self.fill_values_discrete[col], inplace=True)
        df[DISCRETE_COLUMNS] = df[DISCRETE_COLUMNS].astype(int).astype(object)

        for col in CONTINOUS_COLUMNS:
            df[col].fillna(self.fill_values_continuous[col], inplace=True)

        df.drop(columns=TEXT_COLUMNS, inplace=True)

        df[BINARY_COLUMNS] = df[BINARY_COLUMNS].astype(int).astype(object)

        outlier_info = {}
        zscore_info = {}
        for col in CONTINOUS_COLUMNS:
            # Calculate Z-score values for the column
            df[col + '_zscore'] = stats.zscore(df[col])

            # Assuming that outliers are indicated by absolute Z-scores greater than 3
            outlier_indices = df[abs(df[col + '_zscore']) > 3].index

            # Replace outliers with the median of the column
            mean_value = df[col].mean()
            outlier_info[col] = {'outlier_replacement': mean_value, 'outlier_indices': list(outlier_indices)}

            df.loc[outlier_indices, col] = mean_value

            # Drop the Z-score column as it's no longer needed
            df.drop(columns=[col + '_zscore'], inplace=True)

        # OneHot Encoding for ML
        onehot_encoders = {}
        new_columns = []

        for col in NOMINAL_COLUMNS:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

            # print("Type of OH encoder: ", type(encoder))
            new_data = encoder.fit_transform(df[col].to_numpy().reshape(-1, 1))

            new_columns.extend(encoder.get_feature_names_out([col]))

            new_df = pd.DataFrame(new_data, columns=encoder.get_feature_names_out([col]))
            df = pd.concat([df, new_df], axis=1)

            onehot_encoders[col] = encoder

        self.onehot_encoders = onehot_encoders

        df.drop(columns=NOMINAL_COLUMNS, inplace=True)

        min_max_scaler_dict = {}
        min_max_scaler = MinMaxScaler()
        for col in df.columns:
            df[col] = min_max_scaler.fit_transform(df[[col]])
            min_max_scaler_dict[col] = min_max_scaler

        self.min_max_scaler_dict = min_max_scaler_dict

        # Log artifacts to MLflow
        mlflow.log_dict(self.fill_values_nominal, 
                        "fill_values_nominal.json")
        mlflow.log_dict(self.fill_values_discrete, 
                        "fill_values_discrete.json")
        mlflow.log_dict(self.fill_values_continuous, 
                        "fill_values_continuous.json")

        # Serialize and log scalers and encoders
        with open("min_max_scaler_dict.pkl", "wb") as f:
            pickle.dump(self.min_max_scaler_dict, f)
        mlflow.log_artifact("min_max_scaler_dict.pkl")

        with open("onehot_encoders.pkl", "wb") as f:
            pickle.dump(self.onehot_encoders, f)
        mlflow.log_artifact("onehot_encoders.pkl")

        return df

    def preprocessing_pipeline_inference(self, sample_data):
        """Preprocess the inference row to match
        the features we created for training data.
        Returns the preprocessed dataframe for inference.
        
        Keyword arguments:
        sample_data -- Pandas series with the inference data

        Returns:
        input_df -- DataFrame with the preprocessed inference data
        """
        
        sample_data = pd.DataFrame(sample_data)

        sample_data = sample_data.replace('?', np.nan)

        sample_data = MLModel.extract_features(sample_data, "Ticket")

        for col in CONTINOUS_COLUMNS:  
            sample_data[col] = pd.to_numeric(sample_data[col], errors='coerce')

        for col in NOMINAL_COLUMNS:
            sample_data[col].fillna(self.fill_values_nominal[col], inplace=True)

        for col in DISCRETE_COLUMNS:
            sample_data[col].fillna(self.fill_values_discrete[col], inplace=True)
            sample_data[col] = pd.to_numeric(sample_data[col], errors='coerce').astype(int).astype(object)

        for col in CONTINOUS_COLUMNS:
            sample_data[col].fillna(self.fill_values_continuous[col], inplace=True)

        sample_data.drop(columns=TEXT_COLUMNS, inplace=True)

        sample_data[BINARY_COLUMNS] = sample_data[BINARY_COLUMNS].astype(int).astype(object)

        for col, encoder in self.onehot_encoders.items():
            new_data = encoder.transform(sample_data[col].to_numpy().reshape(-1, 1))
            new_df = pd.DataFrame(new_data, columns=encoder.get_feature_names_out([col]))
            sample_data = pd.concat([sample_data, new_df], axis=1).drop(columns=[col])

        for col, scaler in self.min_max_scaler_dict.items():
            expected_features = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else [col]

            # Ensure all expected features are present in sample_data
            missing = [f for f in expected_features if f not in sample_data.columns]
            if missing:
                print(f"Skipping scaler for {col}: missing columns {missing}")
                continue

            # Scaling
            scaled = scaler.transform(sample_data[expected_features])
            sample_data[expected_features] = scaled

        if 'Survived' in sample_data.columns:
            sample_data = sample_data.drop(columns=['Survived'])

        return sample_data

    def get_accuracy(self, X_train, X_test, y_train, y_test):
        """
        Calculate and print the accuracy of the model on both the training and test data sets.
        
        Args:
            X_train: Features for the training set.
            X_test: Features for the test set.
            y_train: Actual labels for the training set.
            y_test: Actual labels for the test set.

        Returns:
            A tuple containing the training accuracy and the test accuracy.
        """
        y_train_pred = self.model.predict(X_train)

        y_test_pred = self.model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print("Train Accuracy: ", train_accuracy)
        print("Test Accuracy: ", test_accuracy)

        return train_accuracy, test_accuracy
    
    def get_accuracy_full(self, X, y):
        """
        Calculate and print the overall accuracy of the model using a data set.

        Args:
            X: Features for the data set.
            y: Actual labels for the data set.

        Returns:
            The accuracy of the model on the provided data set.
        """
        y_pred = self.model.predict(X)

        accuracy = accuracy_score(y, y_pred)

        print("Accuracy: ", accuracy)

        return accuracy

    def train_and_save_model(self, df):
        """Train the model and save it to a file. 
        Returns the train and test accuracy.
        
        Keyword arguments:
        df -- DataFrame with the preprocessed data

        Returns:
        train_accuracy -- Accuracy of the model on the training set
        test_accuracy -- Accuracy of the model on the test set
        """
        y = df["Survived"]
        X = df.drop(columns="Survived")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

        xgb = XGBClassifier(max_depth=4, n_estimators=10)
        xgb.fit(X_train, y_train)

        self.model = xgb

        train_accuracy, test_accuracy = self.get_accuracy(X_train, X_test, y_train, y_test)

        return train_accuracy, test_accuracy, xgb

    @staticmethod
    def create_new_folder(folder):
        """Create a new folder if it doesn't exist.
        
        Keyword arguments:
        folder -- Path to the folder

        Returns:
        None
        """
        Path(folder).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_model(model, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    
    @staticmethod    
    def extract_features(df,column_name):
        # Remove leading/trailing spaces and convert to string
        df[f'{column_name}_clean'] = df[f'{column_name}'].astype(str).str.strip()

        # Extract numeric part of the ticket
        df[f'{column_name}'] = df[f'{column_name}_clean'].apply(lambda x: ''.join([c for c in x if c.isdigit()]) if any(c.isdigit() for c in x) else '0')
        df[f'{column_name}'] = df[f'{column_name}'].astype(int).astype(object)

        # Extract ticket length
        df[f'{column_name}_length'] = df[f'{column_name}_clean'].apply(lambda x: len(x))

        # Drop the temporary cleaned column
        df.drop(columns=[f'{column_name}_clean'], inplace=True)

        return df