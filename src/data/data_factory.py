import lightning as L
from dataclasses import asdict

from src.utils.config import DataConfig, TrainingConfig
from src.data.datamodule.datamodule import HANCOCKDataModule
from src.data.datamodule.hierarchical_directed_survival_graph_datamodule import HierarchicalDirectedSurvivalGraphDataModule


class DataFactory:
    """
    Factory pattern for creating DataModules.
    """
    
    @staticmethod
    def create_datamodule(
        data_config: DataConfig,
        training_config: TrainingConfig,
        **kwargs
    ) -> L.LightningDataModule:
        """
        Create appropriate DataModule based on data configuration.
        
        Args:
            data_config: Data configuration object
            training_config: Training configuration object (for batch_size, num_workers, seed)
            **kwargs: Additional arguments for specific DataModules
            
        Returns:
            Lightning DataModule instance
            
        Raises:
            ValueError: If datamodule type is not supported
        """
        datamodule_type = data_config.datamodule_type
        
        if datamodule_type == "HANCOCK_HierarchicalDirectedSurvivalGraph":
            print("Using HierarchicalDirectedSurvivalGraphDataModule (PyG hierarchical directed survival graphs)")
            return HierarchicalDirectedSurvivalGraphDataModule(
                **asdict(data_config),
                **asdict(training_config),
                **kwargs
            )
        elif datamodule_type == "HANCOCK":
            print("Using HANCOCKDataModule")
            return HANCOCKDataModule(
                **asdict(data_config),
                **asdict(training_config),
                **kwargs
            )
        else:
            raise ValueError(
                f"Unsupported datamodule_type: {datamodule_type}. "
                f"Supported types: {DataFactory.get_supported_datasets()}"
            )

    @staticmethod
    def get_supported_datasets() -> list:
        """Return list of supported dataset types."""
        return ["HANCOCK", "HANCOCK_HierarchicalDirectedSurvivalGraph"]
