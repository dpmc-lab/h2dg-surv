from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from typing import Optional, List

from src.data.datamodule.datamodule import HANCOCKDataModule
from src.data.dataset.h2dg_surv_dataset import H2DGSurvDataset
from src.data.dataset.dataset import Dataset


class H2DGSurvDataModule(HANCOCKDataModule):
    """
    Lightning DataModule for the heterogeneous hierarchical directed survival graph (h2dg_surv).
    
    Inherits all preprocessing from HANCOCKDataModule.
    Wraps base datasets into PyG heterogeneous graphs via H2DGSurvDataset.
    """
    
    def __init__(self, aggregate_images: bool = True, **kwargs):
        """
        Initialize datamodule.
        
        Args:
            aggregate_images: If True, average images into one node per modality (default).
                            If False, create one node per image (GNN learns aggregation).
        """
        super().__init__(**kwargs)
        self.aggregate_images = aggregate_images
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets by reusing parent preprocessing then wrapping in graph datasets.
        """
        # Call parent setup to do all preprocessing and create base datasets
        super().setup(stage)
        
        # Store base datasets before wrapping (needed for evaluation utils)
        # Only save once - don't overwrite on subsequent setup() calls
        if not hasattr(self, 'base_train_dataset'):
            self.base_train_dataset = self.train_dataset
            self.base_val_dataset = self.val_dataset
            self.base_test_dataset = self.test_dataset
        
        # Now wrap the datasets to return HeteroData instead of tuples (only if not already wrapped)
        needs_wrapping = (
            (self.train_dataset is not None and not isinstance(self.train_dataset, H2DGSurvDataset)) or
            (self.val_dataset is not None and not isinstance(self.val_dataset, H2DGSurvDataset)) or
            (self.test_dataset is not None and not isinstance(self.test_dataset, H2DGSurvDataset))
        )
        
        if needs_wrapping:
            print("\n>>> Converting to H2DGSurv datasets...")
        
        if self.train_dataset is not None and not isinstance(self.train_dataset, H2DGSurvDataset):
            self.train_dataset = self._wrap_as_hetero_dataset(self.train_dataset)
            print(f"   - Train: {len(self.train_dataset)} patient graphs")
        
        if self.val_dataset is not None and not isinstance(self.val_dataset, H2DGSurvDataset):
            self.val_dataset = self._wrap_as_hetero_dataset(self.val_dataset)
            print(f"   - Val: {len(self.val_dataset)} patient graphs")
        
        if self.test_dataset is not None and not isinstance(self.test_dataset, H2DGSurvDataset):
            self.test_dataset = self._wrap_as_hetero_dataset(self.test_dataset)
            print(f"   - Test: {len(self.test_dataset)} patient graphs")
    
    def _wrap_as_hetero_dataset(self, base_dataset: Dataset) -> H2DGSurvDataset:
        """
        Wrap a base Dataset into H2DGSurvDataset.
        
        This preserves all preprocessed data and just changes __getitem__ behavior.
        """
        hetero_dataset = H2DGSurvDataset(
            max_tokens_history=base_dataset.max_tokens_history,
            max_tokens_surgery=base_dataset.max_tokens_surgery,
            max_tokens_report=base_dataset.max_tokens_report,
            path_lm=base_dataset.path_lm,
            split=base_dataset.split,
            data_clinical=base_dataset.data_clinical,
            data_blood=base_dataset.data_blood,
            data_pathological=base_dataset.data_pathological,
            tma_cdm=base_dataset.tma_cdm,
            data_root=base_dataset.path_data,
            list_patient_id_sample=base_dataset.list_patient_id_sample,
            aggregate_images=self.aggregate_images,  # Pass the config parameter
        )
        return hetero_dataset
    
    # Override dataloaders to use PyG Batch
    
    def train_dataloader(self, batch_size: int = None, num_workers: int = None) -> DataLoader:
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_workers = num_workers if num_workers is not None else self.num_workers
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
    
    def val_dataloader(self, batch_size: int = None, num_workers: int = None) -> DataLoader:
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_workers = num_workers if num_workers is not None else self.num_workers
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    
    def test_dataloader(self, batch_size: int = None, num_workers: int = None) -> DataLoader:
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_workers = num_workers if num_workers is not None else self.num_workers
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

