import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from models.prediction_benchmarks import TimeGrowth

class ElasticNetReg:
    def __init__(self, df):
        self.df = df
        self.new_df = None
        self.best_model = None
        self.scaler = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train_scaled = None
        self.y_pred = None
        self.y_test = None
        self.features = None
        self.sorted_coefficients = None
        self.expected_feature_order = None  # To store the order of features

    def prepare_data(self):
        # Prepare the DataFrame
        grouped = self.df.groupby('npi')['service_year'].apply(set).reset_index()

        # Unique NPI for both 2018 and 2020
        both_2018_2020 = grouped[grouped['service_year'] == {2018, 2020}]

        # Filter out NPIs that appear in both 2018 and 2020
        self.new_df = self.df[~self.df['npi'].isin(both_2018_2020['npi'])]

        # Select numerical columns, excluding 'CountyID', 'service_year', and 'service_quarter'
        numerical_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_exclude = ['npi', 'countyID', 'service_year', 'service_quarter', 'date']
        selected_numerical_columns = [col for col in numerical_columns if col not in columns_to_exclude]

        # Create the new DataFrame with the selected numerical columns and 'date'
        self.new_df = self.new_df[selected_numerical_columns + ['date', 'npi']].copy()

        # Order new_df by npi and date in ascending order
        self.new_df = self.new_df.sort_values(by=['npi', 'date'], ascending=True)

        claims_columns = [col for col in selected_numerical_columns if col.endswith("claims")]
        other_columns = [col for col in selected_numerical_columns if col not in claims_columns]

        # Function to create lagged features for a single column
        def create_lagged_features(df, column, lags):
            for lag in lags:
                lagged_col_name = f'{column}_lag_{lag}'
                df[lagged_col_name] = df.groupby('npi')[column].shift(lag)
            return df

        # Create lagged features for "claims" columns up to 4 previous quarters
        for col in claims_columns:
            self.new_df = create_lagged_features(self.new_df, col, lags=[1, 2, 3, 4])

        # Create lagged features for other columns for only 1 previous quarter
        for col in other_columns:
            self.new_df = create_lagged_features(self.new_df, col, lags=[1])

        # Drop rows with NaN values created by shifting
        self.new_df = self.new_df.dropna()

        # Get predictions from TimeGrowth model
        time_growth_model = TimeGrowth(self.df)
        predictions_df = time_growth_model.create_predictions_dataframe()

        # Add TimeGrowth predictions to new_df by merging on 'npi'
        self.new_df = self.new_df.merge(predictions_df, on='npi', how='left')

        # Prepare the features by including lagged columns and y_pred_tg
        lagged_columns = [col for col in self.new_df.columns if '_lag_' in col]
        self.features = self.new_df[lagged_columns + ['y_pred_tg']]
        
        # Save the order of features
        self.expected_feature_order = self.features.columns.tolist()

        # Create train-test split such that the test set is the last observation of each npi
        def create_train_test_split(df):
            train = df.groupby('npi').apply(lambda x: x.iloc[:-1], include_groups=False).reset_index(level=0, drop=True)
            test = df.groupby('npi').apply(lambda x: x.iloc[-1:], include_groups=False).reset_index(level=0, drop=True)
            return train, test

        train_df, test_df = create_train_test_split(self.new_df)

        X_train = train_df[self.features.columns]
        y_train = train_df['total_claims']
        X_test = test_df[self.features.columns]
        self.y_test = test_df['total_claims']

        # Standardize the features
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)

        # Standardize the target
        self.y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    def fit_model(self):
        # Set up the hyperparameter space
        param_distributions = {
            'alpha': np.logspace(-2, 1, 201),  # alpha values between 0.01 and 10
            'l1_ratio': np.linspace(0.01, 1, 101)  # l1_ratio values between 0 and 1
        }

        # Initialize the Elastic Net model
        elastic_net = ElasticNet(random_state=42, max_iter=3500)

        # Set up Randomized Search CV
        random_search = RandomizedSearchCV(elastic_net, param_distributions, n_iter=200, cv=5, random_state=42, n_jobs=-1)

        # Fit Randomized Search CV
        random_search.fit(self.X_train_scaled, self.y_train_scaled)

        # Get the best model
        self.best_model = random_search.best_estimator_

        # Predict on the test set
        y_pred_scaled = self.best_model.predict(self.X_test_scaled)

        # Reverse the scaling on the predictions
        self.y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        self.y_pred = np.round(self.y_pred, 0)
        self.y_pred = np.maximum(self.y_pred, 0)

        # Store the coefficients
        coefficients = pd.DataFrame({'Feature': self.features.columns, 'Coefficient': self.best_model.coef_})

        # Sort the coefficients by their absolute value
        self.sorted_coefficients = coefficients.reindex(coefficients['Coefficient'].abs().sort_values(ascending=False).index)

    def get_best_params(self):
        return self.best_model.get_params()

    def get_ordered_features(self):
        return self.sorted_coefficients

    def get_predictions(self):
        return self.y_pred, self.y_test.values

    def create_predictions_dataframe(self):
        # Use the same indices for the predictions as the test set
        test_indices = self.y_test.index
        predictions_df = pd.DataFrame({'npi': self.new_df.loc[test_indices, 'npi'], 'y_pred_en_reg': self.y_pred})
        return predictions_df

    def get_expected_feature_order(self):
        return self.expected_feature_order

    def predict_on_new_data(self, new_data):
        """
        Predict on new data using the trained model.
        
        Parameters:
        - new_data: pd.DataFrame, new data with the same structure as training data (including lagged features).
        
        Returns:
        - y_pred_new: np.array, predictions for the new data.
        """
        # Ensure new_data is preprocessed in the same way as the training data
        new_data_scaled = self.scaler.transform(new_data)

        # Predict on the new data
        y_pred_scaled = self.best_model.predict(new_data_scaled)

        # Reverse the scaling on the predictions
        y_pred_new = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_pred_new = np.round(y_pred_new, 0)
        y_pred_new = np.maximum(y_pred_new, 0)

        return y_pred_new

class ElasticNetNew:
    def __init__(self, df):
        self.df = df
        self.new_df = None
        self.best_model = None
        self.scaler = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train_scaled = None
        self.y_pred = None
        self.y_test = None
        self.features = None
        self.sorted_coefficients = None
        self.expected_feature_order = None  # To store the order of features

    def prepare_data(self):
        # Prepare the DataFrame
        grouped = self.df.groupby('npi')['service_year'].apply(set).reset_index()

        # Unique NPI for both 2018 and 2020
        both_2018_2020 = grouped[grouped['service_year'] == {2018, 2020}]

        # Filter out NPIs that appear in both 2018 and 2020
        self.new_df = self.df[~self.df['npi'].isin(both_2018_2020['npi'])]

        # Select numerical columns, excluding 'CountyID', 'service_year', and 'service_quarter'
        numerical_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_exclude = ['npi', 'countyID', 'service_year', 'service_quarter', 'date']
        selected_numerical_columns = [col for col in numerical_columns if col not in columns_to_exclude]

        # Create the new DataFrame with the selected numerical columns and 'date'
        self.new_df = self.new_df[selected_numerical_columns + ['date', 'npi']].copy()

        # Order new_df by npi and date in ascending order
        self.new_df = self.new_df.sort_values(by=['npi', 'date'], ascending=True)

        claims_columns = [col for col in selected_numerical_columns if col.endswith("claims")]
        other_columns = [col for col in selected_numerical_columns if col not in claims_columns]

        # Function to create lagged features for a single column
        def create_lagged_features(df, column, lags):
            for lag in lags:
                lagged_col_name = f'{column}_lag_{lag}'
                df[lagged_col_name] = df.groupby('npi')[column].shift(lag)
            return df

        # Create lagged features for "claims" columns up to 4 previous quarters
        for col in claims_columns:
            self.new_df = create_lagged_features(self.new_df, col, lags=[1, 2, 3, 4])

        # Create lagged features for other columns for only 1 previous quarter
        for col in other_columns:
            self.new_df = create_lagged_features(self.new_df, col, lags=[1])

        # Drop rows with NaN values created by shifting
        self.new_df = self.new_df.dropna()

        # Get predictions from TimeGrowth model
        time_growth_model = TimeGrowth(self.df)
        predictions_df = time_growth_model.create_predictions_dataframe()

        # Add TimeGrowth predictions to new_df by merging on 'npi'
        self.new_df = self.new_df.merge(predictions_df, on='npi', how='left')

        # Prepare the features by including lagged columns and y_pred_tg
        lagged_columns = [col for col in self.new_df.columns if '_lag_' in col]
        self.features = self.new_df[lagged_columns + ['y_pred_tg']]
        
        # Save the order of features
        self.expected_feature_order = self.features.columns.tolist()

        # Define the target
        target = self.new_df['total_claims']

        # Use train_test_split for train-test split
        X_train, X_test, y_train, self.y_test = train_test_split(self.features, target, test_size=0.2, random_state=42)

        # Standardize the features
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)

        # Standardize the target
        self.y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    def get_expected_feature_order(self):
        return self.expected_feature_order

    def fit_model(self):
        # Set up the hyperparameter space
        param_distributions = {
            'alpha': np.logspace(-2, 1, 201),  # alpha values between 0.01 and 10
            'l1_ratio': np.linspace(0.01, 1, 101)  # l1_ratio values between 0 and 1
        }

        # Initialize the Elastic Net model
        elastic_net = ElasticNet(random_state=42, max_iter=3500)

        # Set up Randomized Search CV
        random_search = RandomizedSearchCV(elastic_net, param_distributions, n_iter=200, cv=5, random_state=42, n_jobs=-1)

        # Fit Randomized Search CV
        random_search.fit(self.X_train_scaled, self.y_train_scaled)

        # Get the best model
        self.best_model = random_search.best_estimator_

        # Predict on the test set
        y_pred_scaled = self.best_model.predict(self.X_test_scaled)

        # Reverse the scaling on the predictions
        self.y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        self.y_pred = np.round(self.y_pred, 0)
        self.y_pred = np.maximum(self.y_pred, 0)

        # Store the coefficients
        coefficients = pd.DataFrame({'Feature': self.features.columns, 'Coefficient': self.best_model.coef_})

        # Sort the coefficients by their absolute value
        self.sorted_coefficients = coefficients.reindex(coefficients['Coefficient'].abs().sort_values(ascending=False).index)

    def get_best_params(self):
        return self.best_model.get_params()

    def get_ordered_features(self):
        return self.sorted_coefficients

    def get_predictions(self):
        return self.y_pred, self.y_test.values

    def create_predictions_dataframe(self):
        # Use the same indices for the predictions as the test set
        test_indices = self.y_test.index
        predictions_df = pd.DataFrame({'npi': self.new_df.loc[test_indices, 'npi'], 'y_pred_en_new': self.y_pred})
        return predictions_df

    def predict_on_new_data(self, new_data):
        """
        Predict on new data using the trained model.
        
        Parameters:
        - new_data: pd.DataFrame, new data with the same structure as training data (including lagged features).
        
        Returns:
        - y_pred_new: np.array, predictions for the new data.
        """
        # Ensure new_data is preprocessed in the same way as the training data
        new_data_scaled = self.scaler.transform(new_data)

        # Predict on the new data
        y_pred_scaled = self.best_model.predict(new_data_scaled)

        # Reverse the scaling on the predictions
        y_pred_new = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_pred_new = np.round(y_pred_new, 0)
        y_pred_new = np.maximum(y_pred_new, 0)

        return y_pred_new
