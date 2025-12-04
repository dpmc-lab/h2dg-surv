import os
import json
import numpy as np
import pandas as pd

def create_cv_splits(patient_ids, n_folds=5, train_ratio=0.7, val_ratio=0.15, random_seed=42):
    """
    Create n-fold cross-validation splits with specified train/val/test ratios.

    Args:
        patient_ids (list): List of patient IDs.
        n_folds (int): Number of folds.
        train_ratio (float): Fraction for training set.
        val_ratio (float): Fraction for validation set.
        test_ratio (float): Fraction for test set.
        random_seed (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with patient_id and fold columns.
    """
    np.random.seed(random_seed)
    patient_ids = np.array(patient_ids)
    n_patients = len(patient_ids)
    
    # Shuffle patient IDs
    shuffled_ids = np.random.permutation(patient_ids)
    
    # Calculate split sizes
    train_size = int(np.floor(train_ratio * n_patients))
    val_size = int(np.floor(val_ratio * n_patients))
    test_size = n_patients - train_size - val_size  # Ensure all patients are included
    
    # Prepare DataFrame
    df = pd.DataFrame({"patient_id": patient_ids})
    
    for fold in range(1, n_folds + 1):
        # Rotate shuffled IDs for each fold to get different splits
        rotated_ids = np.roll(shuffled_ids, shift=fold * test_size)
        
        # Assign splits as full strings
        split = ['train'] * train_size + ['validation'] * val_size + ['test'] * test_size
        
        # Map split back to patient_id order
        fold_assignment = pd.Series(split, index=rotated_ids)
        df[f'fold_{fold}'] = df['patient_id'].map(fold_assignment)
    
    return df

def save_splits(data_root, n_folds, df):
    # Folder
    split_path = os.path.join(data_root, "Split")
    os.makedirs(split_path, exist_ok=True)

    # CSV
    csv_path = os.path.join(split_path, f"folds_{n_folds}.csv")

    # Save
    df.to_csv(csv_path, index=0)
    print(f'File successfully saved at: {csv_path}')

def build_folds(args):

    # Parameters
    data_root   = args.data_root
    n_folds     = args.n_folds
    train_ratio = args.train_ratio
    val_ratio   = args.val_ratio
    random_seed = args.random_seed

    # Source Clinical Data
    clinical_path = os.path.join(data_root, "StructuredData", "clinical_data.json")
    with open(clinical_path, 'r') as file:
        clinical_data = json.load(file)
    df_clinical = pd.DataFrame(clinical_data)

    print(df_clinical.shape)

    # List of patients
    all_patient_ids = list(df_clinical.patient_id)

    # Create splits
    df_splits = create_cv_splits(
        patient_ids = all_patient_ids,
        n_folds     = n_folds,
        train_ratio = train_ratio,
        val_ratio   = val_ratio,
        random_seed = random_seed
        )
    
    # Save dataframe including splits for each fold
    save_splits(data_root, n_folds, df_splits)