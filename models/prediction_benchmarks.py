import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class TimeGrowth:
    def __init__(self, df):
        self.df = df
        self.sorted_df = self._prepare_dataframe()
        self.models = {}  # Dictionary to store models by npi

    def _prepare_dataframe(self):
        # Select the specified columns
        selected_columns = self.df[['npi', 'service_year', 'service_quarter', 'total_claims']]
        # Sort the new DataFrame by 'npi', 'service_year', and 'service_quarter' in ascending order
        sorted_df = selected_columns.sort_values(by=['npi', 'service_year', 'service_quarter'], ascending=True)
        return sorted_df

    def _prepare_train_test(self, group):
        # Use all but the last observation as features
        X_train = group.iloc[:-1][['service_year', 'service_quarter']]
        # Use all but the last observation as the target
        y_train = group.iloc[:-1]['total_claims'].values
        # The last observation is used as the test set
        X_test = group.iloc[-1][['service_year', 'service_quarter']]
        y_test = group.iloc[-1]['total_claims']
        return X_train, y_train, X_test, y_test

    def fit_predict(self):
        y_pred_list = []
        y_test_list = []
        npi_list = []

        # Group the DataFrame by 'npi'
        grouped = self.sorted_df.groupby('npi')

        # Iterate over each group
        for npi, group in grouped:
            # Prepare training and testing data
            X_train, y_train, X_test, y_test = self._prepare_train_test(group)
            
            # Fit the linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predict the value for the test set
            y_pred = model.predict(X_test.to_frame().T)
            y_pred = np.round(y_pred, 0)
            y_pred = np.maximum(y_pred, 0)
            
            # Store the results and the model
            y_pred_list.append(y_pred[0])
            y_test_list.append(y_test)
            npi_list.append(npi)
            self.models[npi] = model

        return y_pred_list, y_test_list

    def create_predictions_dataframe(self):
        y_pred, _ = self.fit_predict()
        npi_list = self.sorted_df['npi'].unique()
        predictions_df = pd.DataFrame({'npi': npi_list, 'y_pred_tg': y_pred})
        return predictions_df

    def predict_for_period(self, year, quarter):
        """
        Predict the total claims for each npi for a specified period.
        
        Parameters:
        - year: int, the service year for prediction.
        - quarter: int, the service quarter for prediction.
        
        Returns:
        - predictions_df: pd.DataFrame, predictions for the specified period with npi and predicted values.
        """
        predictions = []
        npi_list = []

        for npi, model in self.models.items():
            # Create a DataFrame for the new period
            X_new = pd.DataFrame({'service_year': [year], 'service_quarter': [quarter]})
            
            # Predict the value for the specified period
            y_pred = model.predict(X_new)
            y_pred = np.round(y_pred, 0)
            y_pred = np.maximum(y_pred, 0)
            
            predictions.append(y_pred[0])
            npi_list.append(npi)

        predictions_df = pd.DataFrame({'npi': npi_list, 'y_pred_tg': predictions})
        return predictions_df


class ARModel:
    def __init__(self, df):
        self.df = df
        self.sorted_df = self._prepare_dataframe()

    def _prepare_dataframe(self):
        # Select the specified columns
        selected_columns = self.df[['npi', 'service_year', 'service_quarter', 'total_claims']]
        # Sort the new DataFrame by 'npi', 'service_year', and 'service_quarter' in ascending order
        sorted_df = selected_columns.sort_values(by=['npi', 'service_year', 'service_quarter'], ascending=True)
        return sorted_df

    def _create_lagged_features(self, df, lag=1):
        for i in range(1, lag + 1):
            df[f'lag_{i}'] = df['total_claims'].shift(i)
        return df

    def fit_predict(self):
        y_pred_list = []
        y_test_list = []

        # Group the DataFrame by 'npi'
        grouped = self.sorted_df.groupby('npi')

        # Iterate over each group
        for npi, group in grouped:
            # Create lagged features
            group = self._create_lagged_features(group, lag=2)
            
            # Drop rows with NaN values created by shifting
            group = group.dropna()
            
            # Use all but the last observation as training set
            train = group.iloc[:-1]
            # Use the last observation as the test set
            test = group.iloc[-1]
            
            # Prepare training and testing data
            X_train = train[['lag_1', 'lag_2']]
            y_train = train['total_claims']
            X_test = test[['lag_1', 'lag_2']].to_frame().T
            y_test = test['total_claims']
            
            # Fit the linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predict the value for the test set
            y_pred = model.predict(X_test)
            y_pred = np.round(y_pred, 0)
            y_pred = np.maximum(y_pred, 0)
            
            # Store the results
            y_pred_list.append(y_pred[0])
            y_test_list.append(y_test)

        return y_pred_list, y_test_list
