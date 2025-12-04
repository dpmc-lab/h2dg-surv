"""
HANCOCK Survival Analysis - Main Training and Evaluation Script

Usage Examples:

1. Training (default command):
   python main.py --config config/hdhg.yaml
   python main.py train --config config/hdhg.yaml --debug
   python main.py train --config config/hdhg.yaml --k 2

2. Evaluation:
   python main.py eval --checkpoint-dirs results/hdhg/contrib/CV_5/fold_1/2025-01-01_00-00-00

3. K-Folds builder:
    python main.py folds --data_root ./data/HANCOCK --random_seed 42 --n_folds 5
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random
import torch
import lightning as L
import warnings
import pandas as pd

# Suppress specific PyTorch Lightning warnings
warnings.filterwarnings("ignore", ".*GPU available but not used.*")
warnings.filterwarnings("ignore", ".*tensorboardX.*has been removed.*")

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.folds import build_folds
from src.utils.config import load_config, Config
from src.data.data_factory import DataFactory
from src.model.model_factory import ModelFactory
from src.training.trainer_factory import TrainerFactory
from src.eval.survival_eval import SurvivalEval


def parse_args():
    """Parse command line arguments with subcommands."""
    # If no subcommand provided, inject 'train' as default
    if len(sys.argv) > 1 and sys.argv[1] not in ['train', 'eval', 'folds', '-h', '--help']:
        sys.argv.insert(1, 'train')
    
    parser = argparse.ArgumentParser(description="HANCOCK Survival Analysis Training and Evaluation")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)
    
    # Train command (default)
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", type=str, default="config/train/default.yaml", help="Path to the YAML config file")
    train_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    train_parser.add_argument("--resume_training", action="store_true", help="Resume training from checkpoint")
    train_parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name for logging")
    train_parser.add_argument("--k", type=int, default=None, help="Fold number for CV (overrides config)")
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained model(s)")
    eval_parser.add_argument("--checkpoint-dirs", nargs="+", required=True, help="List of checkpoint directories to evaluate")
    eval_parser.add_argument("--stage", type=str, default="test", choices=["train", "val", "test"], help="Stage to evaluate on")
    
    # K-fold builder
    build_folds_parser = subparsers.add_parser("folds", help="Build K-folds for Cross-Validation")
    build_folds_parser.add_argument("--data_root", type=str, default="./data/HANCOCK", help="Path of data")
    build_folds_parser.add_argument("--random_seed", type=int, default=42, help="Random seed to make folds")
    build_folds_parser.add_argument("--n_folds", type=int, default=10, help="Number of folds")
    build_folds_parser.add_argument("--val_ratio", type=float, default=0.2, help="Ratio of validation set from non-test data")

    return parser.parse_args()


def override_config(config: Config, args: argparse.Namespace) -> Config:
    """Override config from command line arguments."""
    # Fold number
    if hasattr(args, 'k') and args.k is not None:
        config.data.fold = args.k
        print(f"Overriding fold number from config: fold = {args.k}")
        # Reset checkpoint_path to None to force reconstruction with new fold
        config.training.checkpoints.checkpoint_path = None
    # Debug mode
    if args.debug:
        config.mode_debug = True
    # Apply debug mode overrides if enabled
    if config.mode_debug:
        print("\nDEBUG MODE ENABLED: Overriding config for fast development runs.\n")
        config.training.epochs = 2
        config.training.batch_size = 4
        config.training.checkpoints.checkpoint_dir = "results/debug"
        config.data.data_fraction = 0.1
    # Update all sub-configs that need post-processing
    config.__post_init__()
    return config


def train_command(args):
    """Execute training command."""
    print("Starting HANCOCK Training")
    
    #########################################################
    # 1. Prepare all elements
    #########################################################
    
    print("=" * 60)
    # Load + override configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    config = override_config(config, args)
    os.makedirs(config.training.checkpoints.checkpoint_path, exist_ok=True)
    config.save_to_yaml(os.path.join(config.training.checkpoints.checkpoint_path, "config.yaml"))
    
    # Set random seed for reproducibility
    random.seed(config.training.seed)
    L.seed_everything(config.training.seed, workers=True)
    print(f"Random seed set to: {config.training.seed}")
    
    # Create data module
    print("Creating data module...")
    datamodule = DataFactory.create_datamodule(
        data_config=config.data,
        training_config=config.training
    )
    print("Setup data module...")
    datamodule.setup()

    # Create model
    print("Creating model...")
    model = ModelFactory.create_model(
        model_config=config.model,
        data_config=config.data,
        input_dims=datamodule.get_input_dims(),
        t_max=datamodule.get_tmax(),
    )
    print(model)
    
    # Create trainer and lightning module
    print("Creating trainer and module...")
    trainer, lightning_module = TrainerFactory.create_trainer_and_module(
        config=config,
        model=model,
    )
    
    #########################################################
    # 2. Training and predictions
    #########################################################    
    # 2.1 Training
    print("\nStarting training...")
    print(f"Checkpoint path: {config.training.checkpoints.checkpoint_path}")
    ckpt_path = os.path.join(config.training.checkpoints.checkpoint_path, "last.ckpt")
    if config.training.resume_training and os.path.exists(ckpt_path):
        trainer.fit(lightning_module, datamodule, ckpt_path=ckpt_path)
    else:
        trainer.fit(lightning_module, datamodule)
        
    print("\nTraining completed successfully!\n")
    
    # 2.2 Predict on LAST checkpoint (current model state)
    print("\n" + "=" * 60)
    print("Running Predictions on LAST checkpoint...")
    print("=" * 60)
    trainer.predict(lightning_module, [datamodule.val_dataloader(), datamodule.test_dataloader()])
    
    # Copy predictions to prediction_last/
    import shutil
    pred_dir = os.path.join(config.training.checkpoints.checkpoint_path, "prediction")
    pred_last_dir = os.path.join(config.training.checkpoints.checkpoint_path, "prediction_last")
    os.makedirs(pred_last_dir, exist_ok=True)
    shutil.copy(os.path.join(pred_dir, "predictions_val.csv"), os.path.join(pred_last_dir, "predictions_val.csv"))
    shutil.copy(os.path.join(pred_dir, "predictions_test.csv"), os.path.join(pred_last_dir, "predictions_test.csv"))
    print(" LAST predictions saved\n")
    
    # 2.3 Predict on BEST checkpoint
    print("=" * 60)
    print("Running Predictions on BEST checkpoint...")
    print("=" * 60)
    best_state_dict_path = os.path.join(config.training.checkpoints.checkpoint_path, "best_model_state_dict.pt")
    if os.path.exists(best_state_dict_path):
        print(f"Loading best model weights from: {best_state_dict_path}")
        
        # Load best model weights
        best_state_dict = torch.load(best_state_dict_path, map_location=next(model.parameters()).device, weights_only=False)
        lightning_module.model.load_state_dict(best_state_dict)
        
        # Predict with best checkpoint
        trainer.predict(lightning_module, [datamodule.val_dataloader(), datamodule.test_dataloader()])
        
        # Copy predictions to prediction_best/
        pred_best_dir = os.path.join(config.training.checkpoints.checkpoint_path, "prediction_best")
        os.makedirs(pred_best_dir, exist_ok=True)
        shutil.copy(os.path.join(pred_dir, "predictions_val.csv"), os.path.join(pred_best_dir, "predictions_val.csv"))
        shutil.copy(os.path.join(pred_dir, "predictions_test.csv"), os.path.join(pred_best_dir, "predictions_test.csv"))
        print(" BEST predictions saved\n")
    else:
        print(f"  WARNING: best_model_state_dict.pt not found at {best_state_dict_path}")
        print("Skipping BEST predictions\n")
    
    #########################################################
    # 3. Evaluation on LAST and BEST
    #########################################################
    print("\n" + "=" * 60)
    print("Evaluating model performance...")
    print("=" * 60)
    
    # 3.1 Evaluate LAST checkpoint
    print("\n--- Evaluating LAST checkpoint ---")
    pred_last_path = os.path.join(config.training.checkpoints.checkpoint_path, "prediction_last", "predictions_test.csv")
    if os.path.exists(pred_last_path):
        predictions_last = pd.read_csv(pred_last_path)
        evaluator_last = SurvivalEval(
            datamodule=datamodule,
            checkpoint_dir=config.training.checkpoints.checkpoint_path,
            tau=5 * 365
        )
        metrics_last = evaluator_last.evaluate(predictions=predictions_last, stage="test")
        evaluator_last.save(output_path=os.path.join(config.training.checkpoints.checkpoint_path, "eval", "metrics_test_last.json"))
        
        print("\n" + "-" * 60 + "\nLAST Checkpoint Metrics:\n" + "-" * 60)
        for metric_name, metric_value in metrics_last.items():
            print(f"  {metric_name:15s}: {metric_value:.4f}")
        print("-" * 60)
    
    # 3.2 Evaluate BEST checkpoint
    print("\n--- Evaluating BEST checkpoint ---")
    pred_best_path = os.path.join(config.training.checkpoints.checkpoint_path, "prediction_best", "predictions_test.csv")
    if os.path.exists(pred_best_path):
        predictions_best = pd.read_csv(pred_best_path)
        evaluator_best = SurvivalEval(
            datamodule=datamodule,
            checkpoint_dir=config.training.checkpoints.checkpoint_path,
            tau=5 * 365
        )
        metrics_best = evaluator_best.evaluate(predictions=predictions_best, stage="test")
        evaluator_best.save(output_path=os.path.join(config.training.checkpoints.checkpoint_path, "eval", "metrics_test_best.json"))
        
        print("\n" + "-" * 60 + "\nBEST Checkpoint Metrics:\n" + "-" * 60)
        for metric_name, metric_value in metrics_best.items():
            print(f"  {metric_name:15s}: {metric_value:.4f}")
        print("-" * 60)
        
        # 3.3 Comparison
        if os.path.exists(pred_last_path):
            print("\n" + "=" * 60 + "\nLAST vs BEST Comparison:\n" + "=" * 60)
            for metric_name in metrics_best.keys():
                diff = metrics_best[metric_name] - metrics_last[metric_name]
                symbol = "down" if diff < 0 else "up"
                print(f"  {metric_name:15s}: LAST={metrics_last[metric_name]:.4f}, BEST={metrics_best[metric_name]:.4f}, Diff={diff:+.4f} {symbol}")
            print("=" * 60)
    
    print("\nTraining script completed!")


def eval_command(args):
    """Execute evaluation command on multiple checkpoint directories."""
    print("Starting HANCOCK Evaluation\n" + "=" * 60)
    print(f"Number of checkpoints to evaluate: {len(args.checkpoint_dirs)}")
    print(f"Stage: {args.stage}")
    print("=" * 60)
    
    # Evaluate each checkpoint
    all_results = {}
    for checkpoint_dir in args.checkpoint_dirs:
        print("\n" + "=" * 60 + f"\nEvaluating: {checkpoint_dir}\n" + "=" * 60)
        
        # Load config from checkpoint directory
        config_path = os.path.join(checkpoint_dir, "config.yaml")
        if not os.path.exists(config_path):
            print(
                f"No config file found at {config_path}. "
                f"Make sure checkpoint directory contains config.yaml"
            )
            continue
        
        print(f"\nLoading configuration from: {config_path}")
        config = load_config(config_path)
        
        # Create data module
        print("Creating data module...")
        datamodule = DataFactory.create_datamodule(data_config=config.data, training_config=config.training)
        print("Setup data module...")
        datamodule.setup()
        
        # Check if predictions exist
        pred_path = os.path.join(checkpoint_dir, "prediction", f"predictions_{args.stage}.csv")
        if not os.path.exists(pred_path):
            print(f"WARNING: Predictions not found at {pred_path}. Skipping.")
            continue
        
        # Evaluate
        evaluator = SurvivalEval(
            datamodule=datamodule,
            checkpoint_dir=checkpoint_dir,
            tau=5 * 365
        )
        metrics = evaluator.evaluate(stage=args.stage)
        evaluator.save()
        
        all_results[checkpoint_dir] = metrics
        
        print("\n" + "-" * 60 + "\nMetrics:\n" + "-" * 60)
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name:15s}: {metric_value:.4f}")
        print("-" * 60)
    
    # Summary
    print("\n" + "=" * 60 + "\nEVALUATION SUMMARY\n" + "=" * 60)
    if len(all_results) == 0:
        print("No checkpoints were evaluated.")
    else:
        # Create summary table
        metric_names = list(next(iter(all_results.values())).keys())
        print(f"\n{'Checkpoint':<50} | " + " | ".join([f"{m:>10s}" for m in metric_names]))
        print("-" * (50 + 3 + (13 * len(metric_names))))
        for checkpoint_dir, metrics in all_results.items():
            values = " | ".join([f"{metrics[m]:>10.4f}" for m in metric_names])
            print(f"{checkpoint_dir} | {values}")
        
        print("=" * 60)
    
    print("\nEvaluation script completed!")


def build_folds_command(args):
    """Build folds for K-folds Cross-Validation."""
    print("Starting HANCOCK K-folds\n" + "=" * 60)
    print(f"Random seed: {args.random_seed}")
    print(f"Folds number: {args.n_folds}")
    print(f"Data root: {args.data_root}")
    print("=" * 60)

    build_folds(args)

    print("=" * 60)
    
    print("Build folds script completed!")


if __name__ == "__main__":
    args = parse_args()
    
    if args.command == "train":
        train_command(args)
    elif args.command == "eval":
        eval_command(args)
    elif args.command == "folds":
        build_folds_command(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)
