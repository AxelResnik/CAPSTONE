import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the data
df = pd.read_excel('datos_ie.xlsx')

# Drop unnecessary columns
df.drop(columns=["NPI_Deactivation_Reason_Code", "NPI_Deactivation_Date", "NPI_Reactivation_Date"], inplace=True)

# List of columns to be dropped
columns_to_drop = [
    "AH_Sign language services for the deaf and hard of hearing",
    "EN_English",
    "SP_Spanish",
    "NX_American Indian or Alaska Native languages",
    "FX_Other languages (excluding Spanish)",
    "F4_Arabic",
    "F17_Any Chinese Language",
    "F19_Creole",
    "F25_Farsi",
    "F28_French",
    "F30_German",
    "F31_Greek",
    "F35_Hebrew",
    "F36_Hindi",
    "F37_Hmong",
    "F42_Italian",
    "F43_Japanese",
    "F47_Korean",
    "F66_Polish",
    "F67_Portuguese",
    "F70_Russian",
    "F81_Tagalog",
    "F92_Vietnamese",
    "N24_Ojibwa",
    "N40_Yupik"
]

# Drop the columns from the DataFrame
df.drop(columns=columns_to_drop, inplace=True)

# Further drop columns
df.drop(columns=['owner_identification', 'Classification_1', 'Specialization_1', 'census_tract'], inplace=True)

# Columns to be imputed
columns_to_impute = [
    "Weekend Availability",
    "Availability Before 8AM",
    "Availability After 5PM"
]

# Replace "Info Not Available" with "No" in the specified columns
df[columns_to_impute] = df[columns_to_impute].replace("Info Not Available", "No")

df.drop(columns=['working_hours_cleaned'], inplace=True)

# Find columns with only one unique value and drop them
columns_with_one_unique_value = [col for col in df.columns if df[col].nunique() == 1]
df.drop(columns=columns_with_one_unique_value, inplace=True)

df.drop(columns=['Healthcare_Provider_Taxonomy_Code_1', 'countyname'], inplace=True)

# Create the new column 'minority_owned'
columns_to_check = ['black_owned', 'latino_owned', 'lgbtq+_owned', 'veteran_owned', 'women_owned']
df['minority_owned'] = df[columns_to_check].apply(lambda row: 1 if row.sum() > 0 else 0, axis=1)
df.drop(columns=columns_to_check, inplace=True)

# Adjust the Entity_Type_Code
df['Entity_Type_Code'] = df['Entity_Type_Code'] - 1

# Drop columns that start with '%'
columns_to_drop = [col for col in df.columns if col.startswith('%')]
df.drop(columns=columns_to_drop, inplace=True)

# Impute business_status
npi_to_impute = df.loc[
    (df['business_status'] == 'Info Not Available') & 
    (df['service_year'] == 2020) & 
    (df['service_quarter'] == 4), 
    'npi'
]
df.loc[df['npi'].isin(npi_to_impute), 'business_status'] = 'OPERATIONAL'

# Further drop columns
df.drop(columns=['zip5', 'lat', 'long'], inplace=True)

# Drop columns that end with 'year' but exclude 'service_year'
columns_to_drop = [col for col in df.columns if col.endswith('year') and col != 'service_year']
df.drop(columns=columns_to_drop, inplace=True)

df['service_year'] = df['service_year'].astype(int)
df['service_quarter'] = df['service_quarter'].astype(int)

quarter_end_month = {1: '03/31', 2: '06/30', 3: '09/30', 4: '12/31'}
df['date'] = df.apply(lambda row: f"{row['service_year']}/{quarter_end_month[row['service_quarter']]}", axis=1)
df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')

# Columns to apply label encoding
label_encode_columns = ['Grouping_1', 'Display_Name_1']

# Columns to apply one-hot encoding
one_hot_encode_columns = ['urbanity', 'provider_size', 'business_status', 
                          'Weekend Availability', 'Availability Before 8AM', 'Availability After 5PM']

# Apply label encoding to specified columns
label_encoders = {}
for col in label_encode_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Apply one-hot encoding to specified columns, dropping the first category
ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first'), one_hot_encode_columns)
    ],
    remainder='passthrough'
)

# Fit and transform the data
df_encoded_array = ct.fit_transform(df)

# Get the column names after transformation
encoded_columns = ct.get_feature_names_out()

# Remove "remainder__" from the column names
cleaned_columns = [col.replace("remainder__", "") for col in encoded_columns]

# Convert the result to a DataFrame
df_encoded = pd.DataFrame(df_encoded_array, columns=cleaned_columns, index=df.index)

# Concatenate the non-transformed columns with the encoded DataFrame
df = df_encoded

# Aggregations
npi_count = df.groupby(['countyID', 'service_year', 'service_quarter'])['npi'].nunique().reset_index().rename(columns={'npi': 'npi_county'})
df = df.merge(npi_count, on=['countyID', 'service_year', 'service_quarter'], how='left')

patients_sum = df.groupby(['countyID', 'service_year', 'service_quarter'])['total_patients'].sum().reset_index().rename(columns={'total_patients': 'total_patients_county'})
df = df.merge(patients_sum, on=['countyID', 'service_year', 'service_quarter'], how='left')

df['patients_per_county'] = df['total_patients_county'] / df['npi_county']

claims_quarter_sum = df.groupby(['service_year', 'service_quarter'])['total_claims'].sum().reset_index().rename(columns={'total_claims': 'claims_quarter'})
patients_quarter_sum = df.groupby(['service_year', 'service_quarter'])['total_patients'].sum().reset_index().rename(columns={'total_patients': 'patients_quarter'})
npi_quarter_count = df.groupby(['service_year', 'service_quarter'])['npi'].nunique().reset_index().rename(columns={'npi': 'npi_quarter'})

df = df.merge(claims_quarter_sum, on=['service_year', 'service_quarter'], how='left')
df = df.merge(patients_quarter_sum, on=['service_year', 'service_quarter'], how='left')
df = df.merge(npi_quarter_count, on=['service_year', 'service_quarter'], how='left')

# Ensure no division by zero
df['claims_per_patient'] = df.apply(lambda row: row['total_claims'] / row['total_patients'] if row['total_patients'] != 0 else 0, axis=1)
df['tele_claims_per_patient'] = df.apply(lambda row: row['tele_services_claims'] / row['tele_services_patients'] if row['tele_services_patients'] != 0 else 0, axis=1)
df['standard_claims_per_patient'] = df.apply(lambda row: row['standard_services_claims'] / row['standard_services_patients'] if row['standard_services_patients'] != 0 else 0, axis=1)

agg_by_county = df.groupby(['countyID', 'service_year', 'service_quarter']).agg({
    'total_claims': 'sum',
    'total_patients': 'sum',
    'tele_services_claims': 'sum',
    'tele_services_patients': 'sum',
    'standard_services_claims': 'sum',
    'standard_services_patients': 'sum'
}).reset_index()

agg_by_county['claims_per_patient_by_county'] = agg_by_county.apply(lambda row: row['total_claims'] / row['total_patients'] if row['total_patients'] != 0 else 0, axis=1)
agg_by_county['tele_claims_per_patient_by_county'] = agg_by_county.apply(lambda row: row['tele_services_claims'] / row['tele_services_patients'] if row['tele_services_patients'] != 0 else 0, axis=1)
agg_by_county['standard_claims_per_patient_by_county'] = agg_by_county.apply(lambda row: row['standard_services_claims'] / row['standard_services_patients'] if row['standard_services_patients'] != 0 else 0, axis=1)

df = df.merge(agg_by_county[['countyID', 'service_year', 'service_quarter', 
                             'claims_per_patient_by_county', 
                             'tele_claims_per_patient_by_county', 
                             'standard_claims_per_patient_by_county']],
              on=['countyID', 'service_year', 'service_quarter'], how='left')

agg_by_grouping = df.groupby(['Grouping_1', 'service_year', 'service_quarter']).agg({
    'total_claims': 'sum',
    'total_patients': 'sum',
    'tele_services_claims': 'sum',
    'tele_services_patients': 'sum',
    'standard_services_claims': 'sum',
    'standard_services_patients': 'sum'
}).reset_index()

agg_by_grouping['claims_per_patient_by_grouping'] = agg_by_grouping.apply(lambda row: row['total_claims'] / row['total_patients'] if row['total_patients'] != 0 else 0, axis=1)
agg_by_grouping['tele_claims_per_patient_by_grouping'] = agg_by_grouping.apply(lambda row: row['tele_services_claims'] / row['tele_services_patients'] if row['tele_services_patients'] != 0 else 0, axis=1)
agg_by_grouping['standard_claims_per_patient_by_grouping'] = agg_by_grouping.apply(lambda row: row['standard_services_claims'] / row['standard_services_patients'] if row['standard_services_patients'] != 0 else 0, axis=1)

df = df.merge(agg_by_grouping[['Grouping_1', 'service_year', 'service_quarter', 
                               'claims_per_patient_by_grouping', 
                               'tele_claims_per_patient_by_grouping', 
                               'standard_claims_per_patient_by_grouping']],
              on=['Grouping_1', 'service_year', 'service_quarter'], how='left')

agg_by_quarter = df.groupby(['service_year', 'service_quarter']).agg({
    'total_claims': 'sum',
    'total_patients': 'sum',
    'tele_services_claims': 'sum',
    'tele_services_patients': 'sum',
    'standard_services_claims': 'sum',
    'standard_services_patients': 'sum'
}).reset_index()

agg_by_quarter['claims_per_patient_by_quarter'] = agg_by_quarter.apply(lambda row: row['total_claims'] / row['total_patients'] if row['total_patients'] != 0 else 0, axis=1)
agg_by_quarter['tele_claims_per_patient_by_quarter'] = agg_by_quarter.apply(lambda row: row['tele_services_claims'] / row['tele_services_patients'] if row['tele_services_patients'] != 0 else 0, axis=1)
agg_by_quarter['standard_claims_per_patient_by_quarter'] = agg_by_quarter.apply(lambda row: row['standard_services_claims'] / row['standard_services_patients'] if row['standard_services_patients'] != 0 else 0, axis=1)

df = df.merge(agg_by_quarter[['service_year', 'service_quarter', 
                               'claims_per_patient_by_quarter', 
                               'tele_claims_per_patient_by_quarter', 
                               'standard_claims_per_patient_by_quarter']],
              on=['service_year', 'service_quarter'], how='left')

# Save the processed DataFrame to a CSV file
df.to_csv('processed_data.csv', index=False)
