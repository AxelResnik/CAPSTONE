import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        return pd.read_parquet(self.file_path)

class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def filter_last_quarter_2020(self):
        self.data = self.data[(self.data['service_year'] == 2020) & (self.data['service_quarter'] == 4)]

    def create_ratios(self):
        self.data['standard_claims_ratio'] = np.where(self.data['total_claims'] != 0, self.data['standard_services_claims'] / self.data['total_claims'], 0)
        self.data['tele_claims_ratio'] = np.where(self.data['total_claims'] != 0, self.data['tele_services_claims'] / self.data['total_claims'], 0)
        self.data['other_claims_ratio'] = np.where(self.data['total_claims'] != 0, self.data['total_other_payers_claims'] / self.data['total_claims'], 0)
        self.data['commercial_claims_ratio'] = np.where(self.data['total_claims'] != 0, self.data['total_commercial_claims'] / self.data['total_claims'], 0)
        self.data['medicaid_claims_ratio'] = np.where(self.data['total_claims'] != 0, self.data['total_medicaid_claims'] / self.data['total_claims'], 0)
        self.data['medicare_claims_ratio'] = np.where(self.data['total_claims'] != 0, self.data['total_medicare_claims'] / self.data['total_claims'], 0)
        self.data['psychotherapy_claims_ratio'] = np.where(self.data['total_claims'] != 0, self.data['total_psychotherapy_claims'] / self.data['total_claims'], 0)

    def handle_missing_and_infinite_values(self):
        self.data = self.data.fillna(0)
        self.data.replace([np.inf, -np.inf], 0, inplace=True)

    def select_columns(self):
        essential_identifiers = ['npi']
        labeling_features = [
            'urbanity', 'countyname', 'Grouping_1', 'lat', 'long', 'provider_size',
            'commercial_claims_ratio', 'medicaid_claims_ratio', 'medicare_claims_ratio', 'other_claims_ratio',
            'total_claims_year', 'total_psychotherapy_claims_year', 'standard_services_claims_year', 'tele_services_claims_year', 'standard_services_psychotherapy_claims_year', 'tele_services_psychotherapy_claims_year',
            'total_other_payers_claims_year', 'total_commercial_claims_year', 'total_medicaid_claims_year', 'total_medicare_claims_year']
        numerical_features = ['standard_claims_ratio', 'tele_claims_ratio', 'psychotherapy_claims_ratio']
        final_columns = essential_identifiers + labeling_features + numerical_features
        self.data = self.data[final_columns]
        return numerical_features

class DataNormalizer:
    def __init__(self, data, numerical_features):
        self.data = data
        self.numerical_features = numerical_features

    def normalize(self):
        scaler = RobustScaler()
        self.data[self.numerical_features] = scaler.fit_transform(self.data[self.numerical_features])
        return self.data

class Clusterer:
    def __init__(self, data, numerical_features, n_clusters=3):
        self.data = data
        self.numerical_features = numerical_features
        self.n_clusters = n_clusters

    def perform_clustering(self):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.data['cluster'] = kmeans.fit_predict(self.data[self.numerical_features])
        return self.data

class DataSaver:
    def __init__(self, data, file_path):
        self.data = data
        self.file_path = file_path

    def save(self):
        self.data.to_csv(self.file_path, index=False)

def main():
    dataset_path = 'datos_ie.parquet'
    loader = DataLoader(dataset_path)
    data = loader.load_data()

    preprocessor = DataPreprocessor(data)
    preprocessor.filter_last_quarter_2020()
    preprocessor.create_ratios()
    preprocessor.handle_missing_and_infinite_values()
    numerical_features = preprocessor.select_columns()

    normalizer = DataNormalizer(preprocessor.data, numerical_features)
    data = normalizer.normalize()

    clusterer = Clusterer(data, numerical_features)
    clustered_data = clusterer.perform_clustering()

    saver = DataSaver(clustered_data, 'clustered_dataset.csv')
    saver.save()

if __name__ == "__main__":
    main()
