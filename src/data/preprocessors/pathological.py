import pandas as pd
import numpy as np
import os
import json
from typing import Tuple, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


# ============================================================
# Feature Engineering Functions
# ============================================================

def load_pathological_data(data_root: str) -> pd.DataFrame:
    path = os.path.join(data_root, "StructuredData", "pathological_data.json")
    with open(path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using one-hot encoding.
    
    TODO: Consider ordinal encoding for ordered variables to reduce dimensionality:
    - resection_status: RX < R0 < R1 < R2
    - resection_status_carcinoma_in_situ: CIS Absent < Ris0 < Ris1
    - closest_resection_margin_in_cm: ordinal bins
    """
    categorical_columns = [
        "primary_tumor_site",
        "pT_stage",
        "pN_stage",
        "grading",
        "hpv_association_p16",
        "perinodal_invasion",
        "lymphovascular_invasion_L",
        "vascular_invasion_V",
        "perineural_invasion_Pn",
        "resection_status_carcinoma_in_situ",
        "carcinoma_in_situ",
        "closest_resection_margin_in_cm",
        "histologic_type"
    ]
    
    for col in categorical_columns:
        df = pd.get_dummies(df, columns=[col], dummy_na=True, dtype=int)
    
    return df


# ============================================================
# Main API
# ============================================================

def prepare_pathological_data_features(data_root: str) -> Tuple[pd.DataFrame, Tuple[List[str], ...]]:
    """
    Feature engineering for pathological data.
    
    Args:
        data_root: Root directory of HANCOCK dataset
        
    Returns:
        Tuple of:
            - df: DataFrame with encoded categorical features and unnormalized continuous features
            - column_groups: Tuple of (infiltration_columns, positive_lymph_columns, resected_lymph_columns)
                - infiltration_columns: median imputation + add_indicator
                - positive_lymph_columns: constant=0 imputation (clinical meaning)
                - resected_lymph_columns: scaling only (never missing)
    """
    # Load data
    df = load_pathological_data(data_root)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Drop resection_status column
    df = df.drop(['resection_status'], axis=1, errors='ignore')
    
    # Separate continuous columns by imputation strategy
    #HACK: this strategy may need to be confirmed
    infiltration_columns = ['infiltration_depth_in_mm']  # median + add_indicator
    positive_lymph_columns = ['number_of_positive_lymph_nodes']  # constant=0 (clinical meaning)
    resected_lymph_columns = ['number_of_resected_lymph_nodes']  # no imputation (never missing)
    
    column_groups = (infiltration_columns, positive_lymph_columns, resected_lymph_columns)
    
    return df, column_groups


def create_pathological_transformer(
    infiltration_columns: List[str],
    positive_lymph_columns: List[str],
    resected_lymph_columns: List[str]
) -> ColumnTransformer:
    """
    Create sklearn ColumnTransformer for pathological data transformation.
    
    Different imputation strategies for different variables:
    - infiltration_depth: median + add_indicator (random missing)
    - positive_lymph_nodes: constant=0 (clinical meaning when resected=0)
    - resected_lymph_nodes: no imputation (never missing)
    
    Args:
        infiltration_columns: Columns for median imputation with indicator
        positive_lymph_columns: Columns for constant=0 imputation
        resected_lymph_columns: Columns for scaling only
    
    Returns:
        ColumnTransformer (not fitted yet)
    """
    # Pipeline 1: median + add_indicator + scaling (infiltration_depth)
    infiltration_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', MinMaxScaler())
    ])
    
    # Pipeline 2: constant=0 + scaling (positive_lymph_nodes)
    positive_lymph_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('scaler', MinMaxScaler())
    ])
    
    # Pipeline 3: scaling only (resected_lymph_nodes)
    resected_lymph_pipeline = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])
    
    # ColumnTransformer with 3 different pipelines
    ct = ColumnTransformer(
        transformers=[
            ('infiltration', infiltration_pipeline, infiltration_columns),
            ('positive_lymph', positive_lymph_pipeline, positive_lymph_columns),
            ('resected_lymph', resected_lymph_pipeline, resected_lymph_columns)
        ],
        remainder='passthrough',  # Keep categorical columns unchanged
        verbose_feature_names_out=False
    )
    
    return ct


def apply_pathological_transformer(
    df: pd.DataFrame, 
    transformer: ColumnTransformer,
    infiltration_columns: List[str],
    positive_lymph_columns: List[str],
    resected_lymph_columns: List[str]
) -> pd.DataFrame:
    """
    Apply fitted transformer and reconstruct DataFrame.
    
    Args:
        df: DataFrame (from prepare_pathological_data_features)
        transformer: Fitted ColumnTransformer
        infiltration_columns: Columns with median + indicator
        positive_lymph_columns: Columns with constant=0
        resected_lymph_columns: Columns with scaling only
    
    Returns:
        Transformed DataFrame with:
        - infiltration_depth_in_mm (scaled)
        - missingindicator_infiltration_depth_in_mm (from add_indicator=True)
        - numb_of_positive_lymph_nodes (scaled)
        - numb_of_resected_lymph_nodes (scaled)
        - categorical columns (unchanged)
    """
    # Transform
    transformed_array = transformer.transform(df)
    
    # Reconstruct columns
    # Order: [infiltration_scaled, infiltration_indicator, positive_lymph, resected_lymph, remainder]
    
    # Infiltration columns
    infiltration_scaled = infiltration_columns  # scaled values
    infiltration_indicator = [f'missingindicator_{col}' for col in infiltration_columns]
    
    # Other continuous columns
    positive_lymph_scaled = positive_lymph_columns
    resected_lymph_scaled = resected_lymph_columns
    
    # Remainder (categorical)
    all_continuous = infiltration_columns + positive_lymph_columns + resected_lymph_columns
    remainder_columns = [col for col in df.columns if col not in all_continuous]
    
    # All columns in order
    all_columns = (infiltration_scaled + infiltration_indicator + 
                   positive_lymph_scaled + resected_lymph_scaled + 
                   remainder_columns)
    
    df_transformed = pd.DataFrame(
        transformed_array,
        columns=all_columns,
        index=df.index
    )
    
    # Rename for consistency
    rename_dict = {
        'number_of_positive_lymph_nodes': 'numb_of_positive_lymph_nodes',
        'number_of_resected_lymph_nodes': 'numb_of_resected_lymph_nodes',
    }
    df_transformed = df_transformed.rename(columns=rename_dict)
    
    return df_transformed

