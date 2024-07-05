import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import xgboost as xgb
from models.elastic_net import ElasticNetReg, ElasticNetNew

class BoosterReg:
    def __init__(self, df):
        self.df = df
        self.new_df = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train_scaled = None
        self.y_pred = None
        self.y_test = None
        self.best_model = None
        self.scaler = StandardScaler()
        self.scaler_y = StandardScaler()
        self.features = None
        self.sorted_importances = None

    def prepare_data(self):
        # Prepare new_df following the same steps as in ElasticNetReg
        grouped = self.df.groupby('npi')['service_year'].apply(set).reset_index()

        # Unique NPI for both 2018 and 2020
        both_2018_2020 = grouped[grouped['service_year'] == {2018, 2020}]

        # Filter out NPIs that appear in both_2018_2020
        new_df = self.df[~self.df['npi'].isin(both_2018_2020['npi'])]

        # Select numerical columns, excluding 'CountyID', 'service_year', and 'service_quarter'
        numerical_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_exclude = ['npi', 'countyID', 'service_year', 'service_quarter', 'date']
        selected_numerical_columns = [col for col in numerical_columns if col not in columns_to_exclude]

        # Create the new DataFrame with the selected numerical columns and 'date'
        self.new_df = new_df[selected_numerical_columns + ['date', 'npi']].copy()

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

        # Step 1: Get predictions from ElasticNetReg model and add them to the features
        elastic_net_model = ElasticNetReg(self.df)
        elastic_net_model.prepare_data()
        elastic_net_model.fit_model()

        # Add ElasticNetReg predictions to new_df by merging on 'npi'
        predictions_df = elastic_net_model.create_predictions_dataframe()
        self.new_df = self.new_df.merge(predictions_df, on='npi', how='left')

        # Step 2: Prepare the data for XGBoost
        # Identify lagged columns
        lagged_columns = [col for col in self.new_df.columns if '_lag_' in col]

        # Prepare features by including lagged columns and the ElasticNetReg predictions
        self.features = self.new_df[lagged_columns + ['y_pred_en_reg']]

        # Define the target
        target = self.new_df['total_claims']

        # Step 3: Split the data into training and test sets using the same logic as ElasticNetReg
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
        # Step 5: Set up the hyperparameter space for XGBoost
        param_distributions = {
            'eta': np.linspace(0.01, 0.3, 60),  # Learning rate between 0.01 and 0.3
            'max_depth': np.arange(1, 4, 1),  # Maximum depth of the tree
            'colsample_bytree': np.linspace(0.7, 1, 31),  # Fraction of features to use for each tree
            'lambda': np.logspace(-2, 1, 101),  # L2 regularization
            'alpha': np.logspace(-2, 1, 101)  # L1 regularization
        }

        # Initialize the XGBoost regressor
        xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

        # Step 6: Set up Randomized Search CV
        random_search = RandomizedSearchCV(xgb_regressor, param_distributions, n_iter=250, cv=5, random_state=42, n_jobs=1)

        # Step 7: Fit Randomized Search CV
        random_search.fit(self.X_train_scaled, self.y_train_scaled)

        # Get the best model
        self.best_model = random_search.best_estimator_

        # Step 8: Predict on the test set
        y_pred_scaled = self.best_model.predict(self.X_test_scaled)

        # Reverse the scaling on the predictions
        self.y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        self.y_pred = np.round(self.y_pred, 0)
        self.y_pred = np.maximum(self.y_pred, 0)

        # Obtain feature importances from the best model
        importances = self.best_model.feature_importances_

        # Create a DataFrame for the feature importances
        feature_importances = pd.DataFrame({'Feature': self.features.columns, 'Importance': importances})

        # Sort the feature importances by their value
        self.sorted_importances = feature_importances.sort_values(by='Importance', ascending=False)

    def get_predictions(self):
        return self.y_pred, self.y_test.values

    def get_sorted_importances(self):
        return self.sorted_importances

    def create_predictions_dataframe(self):
        y_pred, _ = self.get_predictions()
        npi_list = self.new_df['npi'].unique()
        predictions_df = pd.DataFrame({'npi': npi_list, 'y_pred_booster': y_pred})

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


class BoosterNew:
    def __init__(self, df):
        self.df = df
        self.new_df = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train_scaled = None
        self.y_pred = None
        self.y_test = None
        self.best_model = None
        self.scaler = StandardScaler()
        self.scaler_y = StandardScaler()
        self.features = None
        self.sorted_importances = None

    def prepare_data(self):
        # Prepare new_df following the same steps as in ElasticNetNew
        grouped = self.df.groupby('npi')['service_year'].apply(set).reset_index()

        # Unique NPI for both 2018 and 2020
        both_2018_2020 = grouped[grouped['service_year'] == {2018, 2020}]

        # Filter out NPIs that appear in both_2018_2020
        new_df = self.df[~self.df['npi'].isin(both_2018_2020['npi'])]

        # Select numerical columns, excluding 'CountyID', 'service_year', and 'service_quarter'
        numerical_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_exclude = ['npi', 'countyID', 'service_year', 'service_quarter', 'date']
        selected_numerical_columns = [col for col in numerical_columns if col not in columns_to_exclude]

        # Create the new DataFrame with the selected numerical columns and 'date'
        self.new_df = new_df[selected_numerical_columns + ['date', 'npi']].copy()

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

        # Step 1: Get predictions from ElasticNetNew model and add them to the features
        elastic_net_model = ElasticNetNew(self.df)
        elastic_net_model.prepare_data()
        elastic_net_model.fit_model()

        # Add ElasticNetNew predictions to new_df by merging on 'npi'
        predictions_df = elastic_net_model.create_predictions_dataframe()
        self.new_df = self.new_df.merge(predictions_df, on='npi', how='left')

        # Step 2: Prepare the data for XGBoost
        # Identify lagged columns
        lagged_columns = [col for col in self.new_df.columns if '_lag_' in col]

        # Prepare features by including lagged columns and the ElasticNetNew predictions
        self.features = self.new_df[lagged_columns + ['y_pred_en_new']]

        # Define the target
        target = self.new_df['total_claims']

        # Step 3: Split the data into training and test sets using sklearn's train_test_split
        X_train, X_test, y_train, y_test = train_test_split(self.features, target, test_size=0.2, random_state=42)

        self.y_test = y_test

        # Standardize the features
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)

        # Standardize the target
        self.y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    def fit_model(self):
        # Step 5: Set up the hyperparameter space for XGBoost
        param_distributions = {
            'eta': np.linspace(0.01, 0.3, 60),  # Learning rate between 0.01 and 0.3
            'max_depth': np.arange(1, 4, 1),  # Maximum depth of the tree
            'colsample_bytree': np.linspace(0.7, 1, 31),  # Fraction of features to use for each tree
            'lambda': np.logspace(-2, 1, 101),  # L2 regularization
            'alpha': np.logspace(-2, 1, 101),  # L1 regularization
            'subsample': np.linspace(0.6, 0.9, 31)  # Subsample ratio of the training instance
        }

        # Initialize the XGBoost regressor
        xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

        # Step 6: Set up Randomized Search CV
        random_search = RandomizedSearchCV(xgb_regressor, param_distributions, n_iter=250, cv=5, random_state=42, n_jobs=1)

        # Step 7: Fit Randomized Search CV
        random_search.fit(self.X_train_scaled, self.y_train_scaled)

        # Get the best model
        self.best_model = random_search.best_estimator_

        # Step 8: Predict on the test set
        y_pred_scaled = self.best_model.predict(self.X_test_scaled)

        # Reverse the scaling on the predictions
        self.y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        self.y_pred = np.round(self.y_pred, 0)
        self.y_pred = np.maximum(self.y_pred, 0)

        # Obtain feature importances from the best model
        importances = self.best_model.feature_importances_

        # Create a DataFrame for the feature importances
        feature_importances = pd.DataFrame({'Feature': self.features.columns, 'Importance': importances})

        # Sort the feature importances by their value
        self.sorted_importances = feature_importances.sort_values(by='Importance', ascending=False)

    def get_predictions(self):
        return self.y_pred, self.y_test.values

    def create_predictions_dataframe(self):
        y_pred, _ = self.get_predictions()
        npi_list = self.new_df.loc[self.y_test.index, 'npi'].values  # Match npi with test set indices
        predictions_df = pd.DataFrame({'npi': npi_list, 'y_pred_booster_new': y_pred})
        return predictions_df

    def get_sorted_importances(self):
        return self.sorted_importances

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
