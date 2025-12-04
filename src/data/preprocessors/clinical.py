import pandas as pd
import numpy as np
import os
import json
from typing import Tuple, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer


# ============================================================
# Feature Engineering Functions
# ============================================================

def load_clinical_data(data_root: str) -> pd.DataFrame:
    path = os.path.join(data_root, "StructuredData", "clinical_data.json")
    with open(path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features:
    - sex: binary (male=1, female=0)
    - smoking_status: one-hot encoding
    - primarily_metastasis: one-hot encoding
    """
    df = df.copy()
    df['sex'] = (df['sex'] == 'male').fillna(1.).astype(float)
    df = pd.get_dummies(df, columns=["smoking_status", "primarily_metastasis"], dummy_na=True, dtype=int)
    
    return df


def encode_metastasis_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Convert metastasis_1_locations column to dummy columns for each location."""
    df = df.copy()
    
    possible_locations = [
        "Lung", "Bones", "LymphNodes", "Liver", "SoftTissue", 
        "Peritoneum", "Skin", "Pleura", "Brain", "Adrenal", "OtherOrgans",
        "Nan"
    ]
    
    # Initialize all dummy columns with 0
    for location in possible_locations:
        col_name = f"metastasis_1_{location.lower()}"
        df[col_name] = 0
    
    # Process each row
    for idx, row in df.iterrows():
        locations_str = row['metastasis_1_locations']
        
        if pd.isna(locations_str) or locations_str == '':
            df.at[idx, 'metastasis_1_nan'] = 1
            continue
            
        locations = str(locations_str).split()
        for location in locations:
            if location in possible_locations:
                col_name = f"metastasis_1_{location.lower()}"
                df.at[idx, col_name] = 1
    
    df = df.drop('metastasis_1_locations', axis=1)
    return df


def select_clinical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select final columns of interest."""
    columns = [
        "patient_id", 
        "survival_status", "survival_status_with_cause", "days_to_last_information",
        "sex", "smoking_status_former", "smoking_status_non-smoker",
        "smoking_status_smoker", "smoking_status_nan",
        "primarily_metastasis_no", "primarily_metastasis_yes",
        "primarily_metastasis_nan", "age_at_initial_diagnosis",
        "metastasis_1_lung", "metastasis_1_bones", "metastasis_1_lymphnodes",
        "metastasis_1_liver", "metastasis_1_softtissue", "metastasis_1_peritoneum",
        "metastasis_1_skin", "metastasis_1_pleura", "metastasis_1_brain",
        "metastasis_1_adrenal", "metastasis_1_otherorgans", "metastasis_1_nan"
    ]
    return df[columns]


# ============================================================
# Main API
# ============================================================

def prepare_clinical_data_features(data_root: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Feature engineering for clinical data.
    
    Args:
        data_root: Root directory of HANCOCK dataset
        
    Returns:
        Tuple of:
            - df: DataFrame with encoded categorical features and unnormalized continuous features
            - continuous_columns: List of column names to transform
    """
    # Load data
    df = load_clinical_data(data_root)    
    df = encode_categorical_features(df)
    df = encode_metastasis_locations(df)    
    df = select_clinical_columns(df)
    
    # Identify continuous columns to normalize
    continuous_columns = [
        'age_at_initial_diagnosis',
    ]
    
    return df, continuous_columns


def create_clinical_transformer(continuous_columns: List[str]) -> ColumnTransformer:
    """
    Create sklearn ColumnTransformer for clinical data transformation.
    
    Args:
        continuous_columns: List of column names to apply transformation
    
    Returns:
        ColumnTransformer (not fitted yet)
    """
    # Pipeline: scaling (min-max) only (no imputation needed for clinical)
    continuous_pipeline = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])
    
    # ColumnTransformer: apply pipeline to continuous columns only
    ct = ColumnTransformer(
        transformers=[
            ('continuous', continuous_pipeline, continuous_columns)
        ],
        remainder='passthrough',  # Keep categorical columns unchanged
        verbose_feature_names_out=False
    )
    
    return ct


def apply_clinical_transformer(
    df: pd.DataFrame, 
    transformer: ColumnTransformer, 
    continuous_columns: List[str]
) -> pd.DataFrame:
    """
    Apply fitted transformer and reconstruct DataFrame.
    
    Args:
        df: DataFrame (from prepare_clinical_data_features)
        transformer: Fitted ColumnTransformer
        continuous_columns: List of continuous column names (for reconstruction)
    
    Returns:
        Transformed DataFrame with normalized continuous columns
    """
    # Transform
    transformed_array = transformer.transform(df)
    
    # Reconstruct DataFrame
    # ColumnTransformer outputs: [transformed_cols] + [remainder_cols]
    remainder_columns = [col for col in df.columns if col not in continuous_columns]
    all_columns = continuous_columns + remainder_columns
    
    df_transformed = pd.DataFrame(
        transformed_array,
        columns=all_columns,
        index=df.index
    )
    
    # Rename continuous columns to _normalized
    rename_dict = {
        'age_at_initial_diagnosis': 'age_normalized',
    }
    df_transformed = df_transformed.rename(columns=rename_dict)
    
    return df_transformed

