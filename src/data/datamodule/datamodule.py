import lightning as L
from torch.utils.data import DataLoader, random_split
from typing import Optional, Tuple, Dict, List
import torch
import os
import json
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.data.dataset.dataset import Dataset, collate_fn
from src.data.preprocessors.clinical import (
    prepare_clinical_data_features,
    create_clinical_transformer,
    apply_clinical_transformer
)
from src.data.preprocessors.pathological import (
    prepare_pathological_data_features,
    create_pathological_transformer,
    apply_pathological_transformer
)
from src.data.preprocessors.tma import (
    prepare_tma_data_features,
    create_tma_transformer,
    apply_tma_transformer
)
from src.data.preprocessors.blood import (
    prepare_blood_data_features,
    create_blood_transformer,
    apply_blood_transformer
)
import random

#TODO: this is a bit ugly and not Lighning friendly, but it works for now
class HANCOCKDataModule(L.LightningDataModule):
    """
    Lightning DataModule for HANCOCK dataset.
    Handles data loading, splitting, and DataLoader creation.
    """
    
    def __init__(
        self,
        # Tokenization parameters
        max_tokens_history: int,
        max_tokens_surgery: int,
        max_tokens_report: int,
        path_lm: str,
        # Training parameters
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        # Split parameters
        k: int = 5,
        fold: int = 1,
        data_root: str = "./data/HANCOCK",
        data_fraction: float = 1.0,
        **kwargs
    ):
        super().__init__()
        # Tokenization config
        self.max_tokens_history = max_tokens_history
        self.max_tokens_surgery = max_tokens_surgery
        self.max_tokens_report = max_tokens_report
        self.path_lm = path_lm
        self.data_root = data_root
        self.data_fraction = data_fraction
        
        # Training config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        print(f'self.seed: {seed}')
        
        # Split config
        self.k    = k
        self.fold = fold
        
        # Initialize datasets
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        
        # Initialize patient ID lists for each split
        self.train_patient_ids: Optional[List[str]] = None
        self.val_patient_ids: Optional[List[str]] = None
        self.test_patient_ids: Optional[List[str]] = None
        
        # Initialize transformers
        self.clinical_transformer: Optional[ColumnTransformer] = None
        self.clinical_continuous_columns: Optional[List[str]] = None
        
        self.blood_transformer: Optional[ColumnTransformer] = None
        self.blood_value_columns: Optional[List[str]] = None
        
        self.pathological_transformer: Optional[ColumnTransformer] = None
        self.pathological_column_groups: Optional[Tuple[List[str], ...]] = None
        
        self.tma_transformer: Optional[ColumnTransformer] = None
        self.tma_continuous_columns: Optional[List[str]] = None
        
        # Track if data has been prepared to avoid redundant calls
        self._data_is_prepared: bool = False
        

    def prepare_data(self) -> None:
        """
        Preprocess data and create patient ID splits. Called only once per node.
        """
        # Skip if already prepared (handles manual setup() call + Lightning's automatic calls)
        if self._data_is_prepared:
            return
            
        # Read dataframe containing splits for each patient
        split_data = os.path.join(self.data_root, "Split", f"folds_{self.k}.csv")
        df_splits = pd.read_csv(split_data, dtype=str)
        
        # Split patient IDs directly
        self.train_patient_ids = df_splits[df_splits[f"fold_{self.fold}"] == "train"].patient_id.tolist()
        self.val_patient_ids   = df_splits[df_splits[f"fold_{self.fold}"] == "validation"].patient_id.tolist()
        self.test_patient_ids  = df_splits[df_splits[f"fold_{self.fold}"] == "test"].patient_id.tolist()
        
        print(f"Patient ID splits created - Train: {len(self.train_patient_ids)}, "
              f"Val: {len(self.val_patient_ids)}, Test: {len(self.test_patient_ids)}")
        
        # Mark as prepared
        self._data_is_prepared = True

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Create train/val/test datasets. Called on every process in DDP.
        
        Args:
            stage: Not used, all datasets are created at once
        """
        print(f"Datamodule setup - stage: {stage}")
        # Ensure patient ID splits are available
        if not self._data_is_prepared:
            self.prepare_data()
        
        # Create all datasets at once
        if self.train_dataset is None:
            print("\n>>> Preparing TRAIN dataset...")
            
            # ======= CLINICAL =======
            # Clinical data: Feature engineering
            print("  - Clinical: Feature engineering...")
            clinical_train, self.clinical_continuous_columns = prepare_clinical_data_features(self.data_root)
            clinical_train = clinical_train[clinical_train.patient_id.isin(self.train_patient_ids)]
            # Clinical data: Create and fit transformer
            print(f"  - Clinical: Creating transformer for {len(self.clinical_continuous_columns)} continuous columns...")
            self.clinical_transformer = create_clinical_transformer(self.clinical_continuous_columns)
            print("  - Clinical: Fitting transformer on train data...")
            self.clinical_transformer.fit(clinical_train)
            # Clinical data: Transform
            print("  - Clinical: Transforming train data...")
            clinical_train = apply_clinical_transformer(
                clinical_train,
                self.clinical_transformer,
                self.clinical_continuous_columns
            )
            
            # ======= BLOOD =======
            # Blood data: Feature engineering
            print("  - Blood: Feature engineering...")
            blood_train, self.blood_value_columns = prepare_blood_data_features(
                data_root=self.data_root,
                data_clinical=clinical_train
            )
            # Blood data: Create and fit transformer
            print(f"  - Blood: Creating transformer for {len(self.blood_value_columns)} value columns...")
            self.blood_transformer = create_blood_transformer(self.blood_value_columns)
            print("  - Blood: Fitting transformer on train data...")
            self.blood_transformer.fit(blood_train)
            # Blood data: Transform
            print("  - Blood: Transforming train data...")
            blood_train = apply_blood_transformer(
                blood_train, 
                self.blood_transformer, 
                self.blood_value_columns
            )
            
            # ======= PATHOLOGICAL =======
            # Pathological data: Feature engineering
            print("  - Pathological: Feature engineering...")
            patho_train, self.pathological_column_groups = prepare_pathological_data_features(
                data_root=self.data_root
            )
            patho_train = patho_train[patho_train.patient_id.isin(self.train_patient_ids)]
            # Pathological data: Create and fit transformer
            all_patho_cols = sum([list(g) for g in self.pathological_column_groups], [])
            print(f"  - Pathological: Creating transformer for {len(all_patho_cols)} continuous columns...")
            self.pathological_transformer = create_pathological_transformer(*self.pathological_column_groups)
            print("  - Pathological: Fitting transformer on train data...")
            self.pathological_transformer.fit(patho_train)
            # Pathological data: Transform
            print("  - Pathological: Transforming train data...")
            patho_train = apply_pathological_transformer(
                patho_train,
                self.pathological_transformer,
                *self.pathological_column_groups
            )
            
            # ======= TMA =======
            # TMA data: Feature engineering
            print("  - TMA: Feature engineering...")
            tma_train, self.tma_continuous_columns = prepare_tma_data_features(
                data_root=self.data_root,
                clinical_patient_ids=list(clinical_train.patient_id.unique())
            )
            # TMA data: Create and fit transformer
            print(f"  - TMA: Creating transformer for {len(self.tma_continuous_columns)} continuous columns...")
            self.tma_transformer = create_tma_transformer(self.tma_continuous_columns)
            print("  - TMA: Fitting transformer on train data...")
            self.tma_transformer.fit(tma_train)
            # TMA data: Transform
            print("  - TMA: Transforming train data...")
            tma_train = apply_tma_transformer(
                tma_train,
                self.tma_transformer,
                self.tma_continuous_columns
            )
            
            # Create Dataset with pre-processed data
            self.train_dataset = Dataset(
                max_tokens_history=self.max_tokens_history,
                max_tokens_surgery=self.max_tokens_surgery,
                max_tokens_report=self.max_tokens_report,
                path_lm=self.path_lm,
                split="train",
                data_root=self.data_root,
                list_patient_id_sample=self.train_patient_ids,
                data_clinical=clinical_train,
                data_blood=blood_train,
                data_pathological=patho_train,
                tma_cdm=tma_train
            )
            print(self.train_dataset)
        
        if self.val_dataset is None:
            print("\n>>> Preparing VAL dataset...")
            
            # ======= CLINICAL =======
            # Clinical data: Feature engineering
            print("  - Clinical: Feature engineering...")
            clinical_val, _ = prepare_clinical_data_features(self.data_root)
            clinical_val = clinical_val[clinical_val.patient_id.isin(self.val_patient_ids)]
            # Clinical data: Transform with fitted transformer
            print("  - Clinical: Transforming val data with train transformer...")
            clinical_val = apply_clinical_transformer(
                clinical_val,
                self.clinical_transformer,
                self.clinical_continuous_columns
            )
            
            # ======= BLOOD =======
            # Blood data: Feature engineering
            print("  - Blood: Feature engineering...")
            blood_val, _ = prepare_blood_data_features(
                data_root=self.data_root,
                data_clinical=clinical_val
            )
            # Blood data: Transform with fitted transformer
            print("  - Blood: Transforming val data with train transformer...")
            blood_val = apply_blood_transformer(
                blood_val,
                self.blood_transformer,
                self.blood_value_columns
            )
            
            # ======= PATHOLOGICAL =======
            # Pathological data: Feature engineering
            print("  - Pathological: Feature engineering...")
            patho_val, _ = prepare_pathological_data_features(self.data_root)
            patho_val = patho_val[patho_val.patient_id.isin(self.val_patient_ids)]
            # Pathological data: Transform with fitted transformer
            print("  - Pathological: Transforming val data with train transformer...")
            patho_val = apply_pathological_transformer(
                patho_val,
                self.pathological_transformer,
                *self.pathological_column_groups
            )
            
            # ======= TMA =======
            # TMA data: Feature engineering
            print("  - TMA: Feature engineering...")
            tma_val, _ = prepare_tma_data_features(
                data_root=self.data_root,
                clinical_patient_ids=list(clinical_val.patient_id.unique())
            )
            # TMA data: Transform with fitted transformer
            print("  - TMA: Transforming val data with train transformer...")
            tma_val = apply_tma_transformer(
                tma_val,
                self.tma_transformer,
                self.tma_continuous_columns
            )
            
            # Create Dataset
            self.val_dataset = Dataset(
                max_tokens_history=self.max_tokens_history,
                max_tokens_surgery=self.max_tokens_surgery,
                max_tokens_report=self.max_tokens_report,
                path_lm=self.path_lm,
                split="val",
                data_root=self.data_root,
                list_patient_id_sample=self.val_patient_ids,
                data_clinical=clinical_val,
                data_blood=blood_val,
                data_pathological=patho_val,
                tma_cdm=tma_val
            )
            print(self.val_dataset)
        
        if self.test_dataset is None:
            print("\n>>> Preparing TEST dataset...")
            
            # ======= CLINICAL =======
            # Clinical data: Feature engineering
            print("  - Clinical: Feature engineering...")
            clinical_test, _ = prepare_clinical_data_features(self.data_root)
            clinical_test = clinical_test[clinical_test.patient_id.isin(self.test_patient_ids)]
            # Clinical data: Transform with fitted transformer
            print("  - Clinical: Transforming test data with train transformer...")
            clinical_test = apply_clinical_transformer(
                clinical_test,
                self.clinical_transformer,
                self.clinical_continuous_columns
            )
            
            # ======= BLOOD =======
            # Blood data: Feature engineering
            print("  - Blood: Feature engineering...")
            blood_test, _ = prepare_blood_data_features(
                data_root=self.data_root,
                data_clinical=clinical_test
            )
            
            # Blood data: Transform with fitted transformer
            print("  - Blood: Transforming test data with train transformer...")
            blood_test = apply_blood_transformer(
                blood_test,
                self.blood_transformer,
                self.blood_value_columns
            )
            
            # ======= PATHOLOGICAL =======
            # Pathological data: Feature engineering
            print("  - Pathological: Feature engineering...")
            patho_test, _ = prepare_pathological_data_features(self.data_root)
            patho_test = patho_test[patho_test.patient_id.isin(self.test_patient_ids)]
            # Pathological data: Transform with fitted transformer
            print("  - Pathological: Transforming test data with train transformer...")
            patho_test = apply_pathological_transformer(
                patho_test,
                self.pathological_transformer,
                *self.pathological_column_groups
            )
            
            # ======= TMA =======
            # TMA data: Feature engineering
            print("  - TMA: Feature engineering...")
            tma_test, _ = prepare_tma_data_features(
                data_root=self.data_root,
                clinical_patient_ids=list(clinical_test.patient_id.unique())
            )
            # TMA data: Transform with fitted transformer
            print("  - TMA: Transforming test data with train transformer...")
            tma_test = apply_tma_transformer(
                tma_test,
                self.tma_transformer,
                self.tma_continuous_columns
            )
            
            # Create Dataset
            self.test_dataset = Dataset(
                max_tokens_history=self.max_tokens_history,
                max_tokens_surgery=self.max_tokens_surgery,
                max_tokens_report=self.max_tokens_report,
                path_lm=self.path_lm,
                split="test",
                data_root=self.data_root,
                list_patient_id_sample=self.test_patient_ids,
                data_clinical=clinical_test,
                data_blood=blood_test,
                data_pathological=patho_test,
                tma_cdm=tma_test
            )
            print(self.test_dataset)
        
        print(f"\nDatasets created - Train: {len(self.train_dataset)}, "
              f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}\n")

    def train_dataloader(self, batch_size: int = None, num_workers: int = None) -> DataLoader:
        """Create training DataLoader."""
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_workers = num_workers if num_workers is not None else self.num_workers
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

    def val_dataloader(self, batch_size: int = None, num_workers: int = None) -> DataLoader:
        """Create validation DataLoader."""
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_workers = num_workers if num_workers is not None else self.num_workers
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

    def test_dataloader(self, batch_size: int = None, num_workers: int = None) -> DataLoader:
        """Create test DataLoader."""
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_workers = num_workers if num_workers is not None else self.num_workers
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

    def predict_dataloader(self) -> DataLoader:
        """Create prediction DataLoader (same as test for here)."""
        return self.test_dataloader()

    def get_input_dims(self) -> Dict[str, int]:
        """
        Get input dimensions for each modality.
        
        This method examines the actual dataset to determine the real dimensions
        of each modality after preprocessing.
        
        Returns:
            Dictionary with input dimensions for each modality
            
        Raises:
            RuntimeError: If called before setup() or if no datasets available
        """
        # Check if we have a dataset available (preferably train, but any will do)
        available_dataset = None
        
        if self.train_dataset is not None:
            available_dataset = self.train_dataset
        elif self.val_dataset is not None:
            available_dataset = self.val_dataset
        elif self.test_dataset is not None:
            available_dataset = self.test_dataset
        else:
            raise ValueError("No datasets have been created yet")
        
        # Get dimensions from available dataset
        return available_dataset.get_input_dims()
    
    def get_tmax(self) -> int:
        """
        Get maximum time from the train set.
        
        This method examines the actual dataset to determine the time
        of the dataset.
        
        Returns:
            Integer value with the corresponding maximum time.
            
        Raises:
            RuntimeError: If called before setup() or if no datasets available
        """
        # Check if we have a dataset available (preferably train, but any will do)
        available_dataset = None
        
        if self.train_dataset is not None:
            available_dataset = self.train_dataset
        elif self.val_dataset is not None:
            available_dataset = self.val_dataset
        elif self.test_dataset is not None:
            available_dataset = self.test_dataset
        else:
            raise ValueError("No datasets have been created yet")
        
        # Get dimensions from available dataset
        return available_dataset.get_tmax()
