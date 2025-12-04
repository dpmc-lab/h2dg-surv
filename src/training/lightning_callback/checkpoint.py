import os
import json
from typing import Any, Dict, Optional

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from src.training.lightning_module.base_module import BaseLightningModule
from src.training.utils.training_visualizer import TrainingVisualizer

class EnhancedModelCheckpoint(ModelCheckpoint):
    """
    Enhanced ModelCheckpoint with additional features:
    - Training visualization (plots generation)
    - Model state dict export (best/last models) as separate pt files at the end of training
    - Metrics history export (JSON format) at each checkpoint save
    """
    
    # Class attributes for state dict filenames
    BEST_MODEL_STATE_DICT_FILENAME = "best_model_state_dict.pt"
    LAST_MODEL_STATE_DICT_FILENAME = "last_model_state_dict.pt"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visualizer = TrainingVisualizer(save_dir=self.dirpath)

    def on_save_checkpoint(self, trainer: L.Trainer, pl_module: BaseLightningModule, checkpoint: Dict[str, Any]) -> None:
        # Called automatically at each checkpoint save
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        # trainer.save_checkpoint(path=self.dirpath)
        
        metrics = pl_module.metric_history
        self.visualizer.save_all_plots(metrics)
        # Save metrics history to a separate file
        if metrics:
            export_path = os.path.join(self.dirpath, f"metrics_history.json")
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: BaseLightningModule):
        super().on_train_epoch_end(trainer, pl_module)
        
        path = os.path.join(self.dirpath, f"last.ckpt")
        trainer.save_checkpoint(path)
        self.last_model_path = path
            
    def on_fit_end(self, trainer: L.Trainer, pl_module: BaseLightningModule) -> None:
        super().on_fit_end(trainer, pl_module)
        self.export_state_dicts(pl_module)
    
    def export_state_dicts(self, pl_module: BaseLightningModule) -> None:
        # Export best model state dict
        if self.best_model_path:
            try:
                self._export_state_dict(self.best_model_path, self.BEST_MODEL_STATE_DICT_FILENAME, pl_module)
            except Exception as e:
                print(f"Error exporting best model state dict: {e}")
        
        # Export last model state dict  
        try:
            self._export_state_dict(self.last_model_path, self.LAST_MODEL_STATE_DICT_FILENAME, pl_module)
        except Exception as e:
            print(f"Error exporting last model state dict: {e}")
    
    def _export_state_dict(self, ckpt_path: str, export_name: str, pl_module: BaseLightningModule):
        """
        Extract and export model state dict from checkpoint.
        Extracts only the model weights by filtering the checkpoint's state_dict.
        """
        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        # Extract model state dict directly from checkpoint (without loading into module)
        # The checkpoint's state_dict contains both module and model weights with "model." prefix
        full_state_dict = ckpt["state_dict"]
        model_state_dict = {
            k.replace("model.", ""): v 
            for k, v in full_state_dict.items() 
            if k.startswith("model.")
        }
        
        # Save model state dict
        export_path = os.path.join(self.dirpath, export_name)
        torch.save(model_state_dict, export_path)
        print(f"Exported {export_name} to {export_path} from {ckpt_path}")

    def on_exception(self, trainer: L.Trainer, pl_module: BaseLightningModule, exception: Exception) -> None:
        super().on_exception(trainer, pl_module, exception)
        print("🚨 Detected Exception, attempting custom graceful shutdown...")
        self.export_state_dicts(pl_module)

    @staticmethod
    def load_model_state_dict_from_path(checkpoint_path: str, load_best: bool = True, device: Optional[torch.device] = None) -> Optional[Dict[str, Any]]:
        """
        Helper static method for one-shot model state dict loading.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            load_best: If True load best model, if False load last model
            device: Device to load model to
            
        Returns:
            Model state dict or None if not found
        """
        if load_best:
            file_path = os.path.join(checkpoint_path, EnhancedModelCheckpoint.BEST_MODEL_STATE_DICT_FILENAME)
        else:
            file_path = os.path.join(checkpoint_path, EnhancedModelCheckpoint.LAST_MODEL_STATE_DICT_FILENAME)
        
        if os.path.exists(file_path):
            model_state_dict = torch.load(file_path, map_location=device, weights_only=False)
            return model_state_dict
        else:
            return None