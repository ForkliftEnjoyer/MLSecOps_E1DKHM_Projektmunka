import pickle
import pandas as pd
from flask import jsonify
from constants import NOMINAL_COLUMNS, DISCRETE_COLUMNS, CONTINOUS_COLUMNS, TEXT_COLUMNS, BINARY_COLUMNS
import numpy as np
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import os

class MLModel:
    def __init__(self):
        # Load ML artifacts during initialization
        self.fill_values_nominal = (MLModel.load_model(
            'artifacts/nan_outlier_handler/fill_values_nominal.pkl') 
                if os.path.exists('artifacts/nan_outlier_handler/fill_values_nominal.pkl') 
                else print('fill_values_nominal.pkl does not exist'))
        self.fill_values_discrete = (MLModel.load_model(
            'artifacts/nan_outlier_handler/fill_values_discrete.pkl') 
                if os.path.exists('artifacts/nan_outlier_handler/fill_values_discrete.pkl') 
                else print('fill_values_discrete.pkl does not exist'))
        self.fill_values_continuous = (MLModel.load_model(
            'artifacts/nan_outlier_handler/fill_values_continuous.pkl') 
                if os.path.exists('artifacts/nan_outlier_handler/fill_values_continuous.pkl') 
                else print('fill_values_continuous.pkl does not exist'))
        self.min_max_scaler_dict = (MLModel.load_model(
            'artifacts/encoders/min_max_scaler_dict.pkl') 
                if os.path.exists('artifacts/encoders/min_max_scaler_dict.pkl') 
                else print('min_max_scaler_dict.pkl does not exist'))
        self.onehot_encoders = (MLModel.load_model(
            'artifacts/encoders/onehot_encoders_dict.pkl') 
                if os.path.exists('artifacts/encoders/onehot_encoders_dict.pkl') 
                else print('onehot_encoders_dict.pkl does not exist'))
        self.model = (MLModel.load_model(
            'artifacts/models/xgb_model.pkl') 
                if os.path.exists('artifacts/models/xgb_model.pkl') 
                else print('xgb_model.pkl does not exist'))

    def predict(self, inference_row):
        """
        Predicts the outcome based on the input data row.

        This method applies the preprocessing pipeline to the input data, performs necessary
        transformations, and uses the preloaded model to make a prediction. The 'Survived' column
        is removed from the data frame as part of the preprocessing steps. If an error occurs
        during the prediction process, it catches the exception and returns a JSON object with
        the error message and a 500 status code.

        Parameters:
        - inference_row: A single row of input data meant for prediction. Expected to be a list or
        a series that matches the format and order expected by the preprocessing pipeline and model.

        Returns:
        - On success: Returns the prediction as an integer.
        - On failure: Returns a JSON response object with an error message and a 500 status code.

        Notes:
        - Ensure that the input data row is in the correct format and contains the expected features
        excluding 'Survived', which is not required and will be removed during preprocessing.
        - The method is wrapped in a try-except block to handle unexpected errors during prediction.
        """
        try:
            infer_array = pd.Series(inference_row, dtype=str)

            df = self.preprocessing_pipeline_inference(infer_array)
            df.drop('Survived', axis=1, inplace=True)

            y_pred = self.model.predict(df)

            return int(y_pred)

        except Exception as e:
            return jsonify({'message': 'Internal Server Error. ',
                        'error': str(e)}), 500

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

        fill_values_nominal = {col: df[col].mode()[0] for col in NOMINAL_COLUMNS}
        fill_values_discrete = {col: df[col].median() for col in DISCRETE_COLUMNS}
        fill_values_continuous = {col: df[col].mean(skipna=True) for col in CONTINOUS_COLUMNS}


        for col in NOMINAL_COLUMNS:
            df[col].fillna(fill_values_nominal[col], inplace=True)

        for col in DISCRETE_COLUMNS:
            df[col].fillna(fill_values_discrete[col], inplace=True)
        df[DISCRETE_COLUMNS] = df[DISCRETE_COLUMNS].astype(int)

        for col in CONTINOUS_COLUMNS:
            df[col].fillna(fill_values_continuous[col], inplace=True)

        df.drop(columns=TEXT_COLUMNS, inplace=True)

        df[BINARY_COLUMNS] = df[BINARY_COLUMNS].astype(int)

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

        df.drop(columns=NOMINAL_COLUMNS, inplace=True)

        min_max_scaler_dict = {}
        min_max_scaler = MinMaxScaler()
        for col in df.columns:
            df[col] = min_max_scaler.fit_transform(df[[col]])
            min_max_scaler_dict[col] = min_max_scaler

        MLModel.save_model(fill_values_nominal, 
                           'artifacts/nan_outlier_handler/fill_values_nominal.pkl')
        MLModel.save_model(fill_values_discrete, 
                           'artifacts/nan_outlier_handler/fill_values_discrete.pkl')
        MLModel.save_model(fill_values_continuous, 
                           'artifacts/nan_outlier_handler/fill_values_continuous.pkl')
        MLModel.save_model(min_max_scaler_dict, 
                           'artifacts/encoders/min_max_scaler_dict.pkl')
        MLModel.save_model(onehot_encoders, 
                           'artifacts/encoders/onehot_encoders_dict.pkl')

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
            sample_data[col] = pd.to_numeric(sample_data[col], errors='coerce').astype(int)

        for col in CONTINOUS_COLUMNS:
            sample_data[col].fillna(self.fill_values_continuous[col], inplace=True)

        sample_data.drop(columns=TEXT_COLUMNS, inplace=True)

        sample_data[BINARY_COLUMNS] = sample_data[BINARY_COLUMNS].astype(int)

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
        df[f'{column_name}'] = df[f'{column_name}'].astype(int)

        # Extract ticket length
        df[f'{column_name}_length'] = df[f'{column_name}_clean'].apply(lambda x: len(x))

        # Drop the temporary cleaned column
        df.drop(columns=[f'{column_name}_clean'], inplace=True)

        return df