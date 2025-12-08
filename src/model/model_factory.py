import torch.nn as nn
from typing import Dict

from src.utils.config import ModelConfig, DataConfig
from src.model.h2dg_surv_logistic_hazard import H2DGSurvLogisticHazard


class ModelFactory:
    """
    Factory for creating survival prediction models.
    Takes a ModelConfig and DataConfig and returns the appropriate PyTorch model.
    """
    
    # Registry of available models
    _MODELS = {
        "h2dg_surv": H2DGSurvLogisticHazard,
    }
    
    @staticmethod
    def create_model(
        model_config: ModelConfig,
        data_config: DataConfig,
        input_dims: Dict[str, int],
        t_max: int,
        **kwargs
    ) -> nn.Module:
        """
        Create a model based on the configuration.
        
        Args:
            model_config: Model configuration object
            data_config: Data configuration object (for text model path)
            input_dims: Dictionary with input dimensions for each modality
                       e.g., {"clinical": 10, "blood": 200, "patho": 150, "cdm": 2}
            t_max: Integer value corresponding to the maximum time.
            **kwargs: Additional arguments for specific models
            
        Returns:
            PyTorch model instance
            
        Raises:
            ValueError: If model type is not supported
        """
        model_name = model_config.name.lower()
        
        if model_name not in ModelFactory._MODELS:
            raise ValueError(
                f"Unsupported model type: {model_name}. "
                f"Supported models: {list(ModelFactory._MODELS.keys())}"
            )
        model_class = ModelFactory._MODELS[model_name]
        
        # Use num_time_bins if specified, otherwise use t_max (1 bin per day)
        num_bins = data_config.num_time_bins if data_config.num_time_bins is not None else t_max
        kwargs["num_bins"] = num_bins
        
        # Create model with input dimensions and text model path
        model = model_class(
            input_dims=input_dims,
            lm_path=data_config.path_lm,
            **model_config.args,
            **kwargs
        )
            
        return model
    
    @staticmethod
    def get_supported_models() -> list:
        """Return list of supported model types."""
        return list(ModelFactory._MODELS.keys())
