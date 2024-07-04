import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class TimeGrowth:
    def __init__(self, df):
        self.df = df
        self.sorted_df = self._prepare_dataframe()

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
            
            # Store the results
            y_pred_list.append(y_pred[0])
            y_test_list.append(y_test)
            npi_list.append(npi)

        return y_pred_list, y_test_list

    def create_predictions_dataframe(self):
        y_pred, _ = self.fit_predict()
        npi_list = self.sorted_df['npi'].unique()
        predictions_df = pd.DataFrame({'npi': npi_list, 'y_pred_tg': y_pred})
        return predictions_df
