from typing import Any, Optional

import lightning as L
from lightning.pytorch.callbacks import Callback
from tqdm import tqdm
from src.training.lightning_module.base_module import BaseLightningModule


class CustomProgressBarCallback(Callback):
    """Custom Progress Bar Callback that shows metrics like legacy trainer."""

    def __init__(self):
        """Initialize the callback."""
        super().__init__()
        self.train_progress_bar: Optional[tqdm] = None
        self.val_progress_bar: Optional[tqdm] = None
        self.test_progress_bar: Optional[tqdm] = None
        self.predict_progress_bar: Optional[tqdm] = None

    def _get_progress_description(self, prefix: str, pl_module: BaseLightningModule) -> str:
        """
        Generate progress bar description with current metrics.
        
        Args:
            prefix: Prefix for the description ("TRAIN" or "VAL")
            pl_module: Lightning module to get metrics from            
        Returns:
            Formatted description string
        """
        # Get current phase from prefix
        phase = "train" if "TRAIN" in prefix else "val"
        
        # Get current loss using the new helper method
        current_loss = pl_module.get_current_loss(phase)
        desc = f"{prefix} : Loss={current_loss:.4f}"
        
        # Add accuracy metrics if available using helper methods
        acc = pl_module.get_current_metric("accuracy", phase)
        f1_score = pl_module.get_current_metric("f1_score", phase)
        
        if acc is not None:
            desc += f", Acc={acc:.2%}"
        if f1_score is not None:
            desc += f", F1={f1_score:.2%}"
        return desc

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: BaseLightningModule) -> None:
        """Called at the start of training epoch."""
        # Print epoch separator
        current_epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs
        print(f"\n{'='*100}")
        print(f"Epoch {current_epoch}/{max_epochs}")
        print('='*100)
        
        # Create training progress bar
        total_batches = trainer.num_training_batches
        if isinstance(total_batches, int):
            self.train_progress_bar = tqdm(
                total=total_batches,
                desc="TRAIN",
                colour="green",
            )

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: BaseLightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Called at the end of each training batch."""
        if self.train_progress_bar:
            desc = self._get_progress_description("TRAIN", pl_module)
            self.train_progress_bar.set_description(desc)
            self.train_progress_bar.update(1)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: BaseLightningModule) -> None:
        """Called at the end of training epoch."""
        if self.train_progress_bar:
            self.train_progress_bar.close()
            self.train_progress_bar = None

        if self.val_progress_bar:
            self.val_progress_bar.close()
            self.val_progress_bar = None

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: BaseLightningModule) -> None:
        """Called at the start of validation epoch."""
        # Create validation progress bar
        total_batches = trainer.num_val_batches
        
        # Handle case where num_val_batches is a list (multiple dataloaders) or int (single dataloader)
        if isinstance(total_batches, list):
            total_batches = total_batches[0] if total_batches else 0
            
        if isinstance(total_batches, int) and total_batches > 0:
            self.val_progress_bar = tqdm(
                total=total_batches,
                desc="VAL",
                colour="green",
            )
            
    def on_validation_batch_end(self, trainer: L.Trainer, pl_module: BaseLightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Called at the end of each validation batch."""
        if self.val_progress_bar:
            desc = self._get_progress_description("VAL  ", pl_module)
            self.val_progress_bar.set_description(desc)
            self.val_progress_bar.update(1)

    def on_train_end(self, trainer: L.Trainer, pl_module: BaseLightningModule) -> None:
        """Called at the end of training."""
        # Clean up any remaining progress bars
        if self.train_progress_bar:
            self.train_progress_bar.close()
            self.train_progress_bar = None
        if self.val_progress_bar:
            self.val_progress_bar.close()
            self.val_progress_bar = None
            
        # Print final separator
        print(f"\n{'='*100}")
        print("Training completed!")
        print('='*100, '\n') 

    def on_test_start(self, trainer: L.Trainer, pl_module: BaseLightningModule, **kwargs) -> None:
        """Called at the start of each test batch."""
        total_batches = trainer.num_test_batches
        if isinstance(total_batches, list):
            total_batches = total_batches[0] if total_batches else 0
            
        if isinstance(total_batches, int) and total_batches > 0 and trainer.is_global_zero:
            self.test_progress_bar = tqdm(
                    total=total_batches,
                    desc=f"TEST",
                    colour="cyan",
                )
            
    def on_test_batch_end(self, trainer: L.Trainer, pl_module: BaseLightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.test_progress_bar and trainer.is_global_zero:
            self.test_progress_bar.update(1)

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: BaseLightningModule, dataloader_idx: int = 0) -> None:
        """Called at the end of test epoch."""
        if self.test_progress_bar and trainer.is_global_zero:
            self.test_progress_bar.close()
            self.test_progress_bar = None

    def on_predict_start(self, trainer: L.Trainer, pl_module: BaseLightningModule, **kwargs) -> None:
        """Called at the start of each predict batch."""
        total_batches = trainer.num_predict_batches
        if isinstance(total_batches, list):
            total_batches = total_batches[0] if total_batches else 0
            
        if isinstance(total_batches, int) and total_batches > 0 and trainer.is_global_zero:
            self.predict_progress_bar = tqdm(
                    total=total_batches,
                    desc=f"PREDICT",
                    colour="cyan",
                )
            
    def on_predict_batch_end(self, trainer: L.Trainer, pl_module: BaseLightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.predict_progress_bar and trainer.is_global_zero:
            self.predict_progress_bar.update(1)

    def on_predict_epoch_end(self, trainer: L.Trainer, pl_module: BaseLightningModule, dataloader_idx: int = 0) -> None:
        """Called at the end of predict epoch."""
        if self.predict_progress_bar and trainer.is_global_zero:
            self.predict_progress_bar.close()
            self.predict_progress_bar = None