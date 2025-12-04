import os
import warnings
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

# Filter out matplotlib legend warnings when no labels are available
warnings.filterwarnings("ignore", message="No artists with labels found to put in legend*", category=UserWarning)

class TrainingVisualizer:
    """Visualize training progress with various plot types."""

    def __init__(
        self,
        save_dir: str,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        style: str = "seaborn-v0_8",
        save_format: str = "png"
    ):
        """
        Args:
            save_dir: Directory to save figures
            figsize: Figure size (width, height)
            dpi: Figure resolution
            style: Matplotlib style
            save_format: File format for saved figures (png, pdf, svg)
        """
        self.save_dir = save_dir
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        self.save_format = save_format
        
        # Create figures directory path
        self.figures_dir = os.path.join(save_dir, "figures")        
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Set matplotlib style
        try:
            mplstyle.use(self.style)
        except OSError:
            # Fallback if style not available
            mplstyle.use("default")

    def save_all_plots(self, log_dict: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]) -> None:
        """
        Save all available plots.
        Args:
            log_dict: Training metric history from MetricsMixin
        """
        if log_dict and all(isinstance(v, list) for v in log_dict.values()):
            log_dict = self._parse_flat_to_nested(log_dict)
        self.plot_losses(log_dict)
        self.plot_metrics(log_dict) 

    def plot_losses(self, log_dict: Dict[str, List[float]]) -> None:
        """
        Plot epoch and step losses.
        Args:
            log_dict: Training metric history from MetricsMixin
        """
        self._plot_epoch_losses(log_dict)
        self._plot_step_losses(log_dict)

    def plot_metrics(self, log_dict: Dict[str, Dict[str, List[float]]]) -> None:
        """
        Plot all available metrics.
        Args:
            log_dict: Training metric history from MetricsMixin
        """
        available_metrics = set()
        for phase in ["train", "val"]:
            if phase in log_dict:
                available_metrics.update(log_dict[phase].keys())
        
        # Remove loss and step_losses as they have their own plots
        available_metrics.discard("loss_epoch")
        available_metrics.discard("loss_step")
        
        # Plot each metric
        for metric in available_metrics:
            self._plot_metric(log_dict, metric)

    def _plot_epoch_losses(self, log_dict: Dict[str, Dict[str, List[float]]]) -> None:
        """Plot epoch-wise losses."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot train and validation losses
        train_losses = log_dict.get("train", {}).get("loss_epoch", [])
        val_losses = log_dict.get("val", {}).get("loss_epoch", [])
        
        epochs = list(range(1, len(train_losses) + 1))
        
        if train_losses:
            ax.plot(epochs, train_losses, label="Train Loss", marker='o', linewidth=2)
        
        if val_losses:
            val_epochs = list(range(1, len(val_losses) + 1))
            ax.plot(val_epochs, val_losses, label="Validation Loss", marker='s', linewidth=2)
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        
        title = "Training and Validation Loss"
        ax.set_title(title)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure using FileSystemHandler
        filename = f"loss_epoch.{self.save_format}"
        self._save_figure(filename)

    def _plot_step_losses(self, log_dict: Dict[str, Dict[str, List[float]]]) -> None:
        """Plot step-wise losses."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot step losses
        train_step_losses = log_dict.get("train", {}).get("loss_step", [])
        val_step_losses = log_dict.get("val", {}).get("loss_step", [])
        
        if train_step_losses:
            steps = list(range(1, len(train_step_losses) + 1))
            ax.plot(steps, train_step_losses, label="Train", alpha=0.7, linewidth=1)
        
        if val_step_losses:
            val_steps = list(range(1, len(val_step_losses) + 1))
            ax.plot(val_steps, val_step_losses, label="Validation", alpha=0.7, linewidth=1)
        
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        
        title = "Step-wise Loss"
        ax.set_title(title)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure using FileSystemHandler
        filename = f"loss_step.{self.save_format}"
        self._save_figure(filename)

    def _plot_metric(self, log_dict: Dict[str, Dict[str, List[float]]], metric: str) -> None:
        """Plot a specific metric."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot train and validation metrics
        train_values = log_dict.get("train", {}).get(metric, [])
        val_values = log_dict.get("val", {}).get(metric, [])
        
        epochs = list(range(1, len(train_values) + 1))
        
        if train_values:
            ax.plot(epochs, train_values, label=f"Train", marker='o', linewidth=2)
        
        if val_values:
            val_epochs = list(range(1, len(val_values) + 1))
            ax.plot(val_epochs, val_values, label=f"Validation", marker='s', linewidth=2)
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.title())
        title = f"Training and Validation {metric.title()}"
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        filename = f"{metric}.{self.save_format}"
        self._save_figure(filename)

    def _save_figure(self, filename: str) -> None:
        """
        Save the current matplotlib figure
        
        Args:
            filename: Name of the file (with extension)
        """
        filepath = os.path.join(self.figures_dir, filename)        
        plt.savefig(filepath, format=self.save_format, dpi=self.dpi, bbox_inches='tight')        
        plt.close()

    def _parse_flat_to_nested(self, log_dict: Dict[str, List[float]]) -> Dict[str, Dict[str, List[float]]]:
        """
        Convert flat log_dict format to nested format for.
        
        Args:
            log_dict: Flat format like {"train/loss_epoch": [...], "val/accuracy_micro": [...]}
            
        Returns:
            Nested format like {"train": {"loss_epoch": [...], "accuracy_micro": [...], "loss_step": [...], "iteration_losses": [...]}
        """
        nested_dict = {}
        
        for key, values in log_dict.items():
            if "/" in key:
                phase, metric_name = key.split("/", 1)
                if phase not in nested_dict:
                    nested_dict[phase] = {}
                nested_dict[phase][metric_name] = values
                
        return nested_dict