import pandas as pd
import numpy as np
import os
from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


# ============================================================
# Feature Engineering Functions
# ============================================================

def load_tma_measurements(data_root: str) -> pd.DataFrame:
    """Load TMA cell density measurements."""
    path = os.path.join(data_root, "TMA_CellDensityMeasurements", "TMA_celldensity_measurements.csv")
    tma_cdm = pd.read_csv(path)
    tma_cdm['Image'] = tma_cdm['Image'].str.extract(r'(block\d+)')
    return tma_cdm


def load_tma_maps(data_root: str) -> pd.DataFrame:
    """Load and concatenate all TMA map files."""
    maps_folder = os.path.join(data_root, "TMA_Maps")
    dfs = []
    
    for i, subdir in enumerate(os.listdir(maps_folder)):
        num_block = i + 1
        map_path = os.path.join(maps_folder, subdir)
        map_df = pd.read_csv(map_path)
        map_df['Image'] = [f"block{num_block}"] * len(map_df)
        dfs.append(map_df)
    
    map_df = pd.concat(dfs, ignore_index=True)
    map_df = map_df.dropna(subset=['Case ID'])
    map_df['Case ID'] = map_df['Case ID'].astype(int).astype(str).str.zfill(3)
    map_df = map_df.rename(columns={"Case ID": "patient_id"})
    
    return map_df


def merge_tma_data(tma_cdm: pd.DataFrame, tma_maps: pd.DataFrame) -> pd.DataFrame:
    """Merge cell density measurements with TMA maps."""
    return tma_cdm.merge(
        tma_maps, 
        left_on=['Name', 'Image'], 
        right_on=['core', 'Image'], 
        how='inner'
    )


def aggregate_tma_by_patient(tma_merged: pd.DataFrame) -> pd.DataFrame:
    """Aggregate TMA metrics by patient."""
    # Count and density aggregation
    tma_agg = tma_merged.groupby('patient_id').agg({
        'Num Positive': 'sum',
        'Num Positive per mm^2': 'mean'
    }).reset_index()
    
    # Rename columns
    tma_agg = tma_agg.rename(columns={
        'Num Positive': 'count_positive',
        'Num Positive per mm^2': 'avg_density_positive'
    })
    
    return tma_agg


def fill_missing_patients(df: pd.DataFrame, clinical_patient_ids: List[str]) -> pd.DataFrame:
    """Add missing patients"""
    complete_patients = pd.DataFrame({'patient_id': clinical_patient_ids})
    df = complete_patients.merge(df, on='patient_id', how='left')
    return df


# ============================================================
# Main API
# ============================================================

def prepare_tma_data_features(data_root: str, clinical_patient_ids: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Feature engineering for TMA data.
    
    Args:
        data_root: Root directory of HANCOCK dataset
        clinical_patient_ids: List of patient IDs from clinical data
        
    Returns:
        Tuple of:
            - df: DataFrame with unnormalized TMA features
            - continuous_columns: List of column names to transform
    """
    # Load data
    tma_cdm = load_tma_measurements(data_root)
    tma_maps = load_tma_maps(data_root)
    
    # Merge and aggregate
    tma_merged = merge_tma_data(tma_cdm, tma_maps)
    tma_agg = aggregate_tma_by_patient(tma_merged)
    
    # Fill missing patients (no imputation yet)
    df = fill_missing_patients(tma_agg, clinical_patient_ids)
    
    # Limit to 763 patients
    df = df.head(763)
    
    # Identify continuous columns to normalize
    continuous_columns = ['count_positive', 'avg_density_positive']
    
    return df, continuous_columns


def create_tma_transformer(continuous_columns: List[str]) -> ColumnTransformer:
    """
    Create sklearn ColumnTransformer for TMA data transformation.
    
    Args:
        continuous_columns: List of column names to apply transformation
    
    Returns:
        ColumnTransformer (not fitted yet)
    """
    # Pipeline: imputation (median) + scaling (min-max)
    continuous_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', MinMaxScaler())
    ])
    
    # ColumnTransformer: apply pipeline to continuous columns only
    ct = ColumnTransformer(
        transformers=[
            ('continuous', continuous_pipeline, continuous_columns)
        ],
        remainder='passthrough',  # Keep patient_id
        verbose_feature_names_out=False
    )
    
    return ct


def apply_tma_transformer(
    df: pd.DataFrame, 
    transformer: ColumnTransformer, 
    continuous_columns: List[str]
) -> pd.DataFrame:
    """
    Apply fitted transformer and reconstruct DataFrame.
    
    Args:
        df: DataFrame (from prepare_tma_data_features)
        transformer: Fitted ColumnTransformer
        continuous_columns: List of continuous column names (for reconstruction)
    
    Returns:
        Transformed DataFrame with:
        - count_pos, avg_density_pos (scaled)
        - missingindicator_count_positive, missingindicator_avg_density_positive
        - patient_id
    """
    # Transform
    transformed_array = transformer.transform(df)
    
    # Reconstruct columns
    # [value_1_scaled, value_2_scaled, indicator_1, indicator_2]
    
    # Scaled continuous columns
    scaled_continuous_cols = continuous_columns.copy()
    
    # Indicator columns (added by add_indicator=True)
    indicator_cols = [f'missingindicator_{col}' for col in continuous_columns]
    
    # Remainder columns (patient_id)
    remainder_columns = [col for col in df.columns if col not in continuous_columns]
    
    # All columns in order: scaled values + indicators + remainder
    all_columns = scaled_continuous_cols + indicator_cols + remainder_columns
    
    df_transformed = pd.DataFrame(
        transformed_array,
        columns=all_columns,
        index=df.index
    )
    
    # Rename continuous columns: *_positive -> *_pos
    rename_dict = {
        'count_positive': 'count_pos',
        'avg_density_positive': 'avg_density_pos',
    }
    df_transformed = df_transformed.rename(columns=rename_dict)
    
    return df_transformed

