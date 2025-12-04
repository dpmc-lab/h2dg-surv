import os
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve,
    auc,
    f1_score, 
    precision_score, 
    recall_score, 
    accuracy_score
)


class ClassificationEvaluator:
    """
    Binary classification evaluator at a specific time horizon.
    
    Evaluates survival predictions as binary classification:
    - Predict whether patient is alive or dead at time t
    
    Methodology:
    1. Exclude patients censored before t (unknown status at t)
    2. Positive class (y=1): patients who died at or before t
    3. Negative class (y=0): patients censored after t OR died after t
    4. Score: 1 - S(t) where S(t) is survival probability at time t
    
    Threshold selection:
    - Validation set: Find threshold that maximizes F1-score
    - Test set: Use threshold from validation set
    
    Metrics computed:
    - AUROC: Area Under ROC Curve
    - AP: Average Precision (sklearn's average_precision_score)
    - AUPRC: Area Under Precision-Recall Curve (computed with precision_recall_curve + auc)
    - F1-score, Precision, Recall, Accuracy (threshold-dependent)
    """
    
    def __init__(self, checkpoint_dir: str, horizons: List[int] | int):
        """
        Initialize ClassificationEvaluator.
        
        Args:
            checkpoint_dir: Path to checkpoint directory for loading predictions
            horizons: List of time horizons in days for classification (e.g., [365, 1095, 1825])
        """
        self.checkpoint_dir = checkpoint_dir
        self.horizons = horizons if isinstance(horizons, list) else [horizons]
        
        # Computed from validation data (one threshold per horizon)
        self.best_thresholds = {}
    
    def find_val_best_threshold(self):
        """
        Compute optimal classification thresholds from validation data by maximizing F1 score.
        One threshold per horizon, stored in self.best_thresholds.
        """
        # Load validation predictions
        predictions_val = self._load_predictions(stage="val")
        
        # Find best threshold for each horizon
        for horizon in self.horizons:
            y_val, risk_score_val = self._extract_labels_and_scores(predictions_val, horizon)
            if y_val is not None and risk_score_val is not None:
                self.best_thresholds[horizon] = self._find_best_threshold(y_val, risk_score_val)
                print(f"  Best threshold for t={horizon}: {self.best_thresholds[horizon]:.4f}")
    
    def evaluate(self, predictions: pd.DataFrame) -> Dict[str, float]:
        """
        Compute classification metrics on given predictions for all horizons.
        
        Args:
            predictions: DataFrame with predictions
        
        Returns:
            Dictionary with metrics (auroc_t_{horizon}, auprc_t_{horizon}, f1_score_t_{horizon}, etc.)
        """
        all_metrics = {}
        
        # Compute metrics for each horizon
        for horizon in self.horizons:
            print(f"\nClassification metrics at t={horizon} days ({horizon/365:.1f} years)...")
            
            # Extract labels and scores for this horizon
            y_true, risk_score = self._extract_labels_and_scores(predictions, horizon)
            
            if y_true is None or risk_score is None:
                print(f"  Skipping t={horizon} (no valid data)")
                continue
            
            # Compute threshold-independent metrics
            metrics = {}
            metrics["auroc"] = float(roc_auc_score(y_true, risk_score))
            metrics["ap"] = float(average_precision_score(y_true, risk_score))        
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, risk_score)
            metrics["auprc"] = float(auc(recall_curve, precision_curve))
            
            # Compute threshold-dependent metrics
            threshold = self.best_thresholds.get(horizon, 0.5)
            y_pred = (risk_score >= threshold).astype(int)
            
            metrics["f1_score"] = float(f1_score(y_true, y_pred, zero_division=0))
            metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            
            # Additional information
            metrics["n_valid_classif"] = int(len(y_true))
            metrics["n_positive_classif"] = int(np.sum(y_true))
            metrics["best_val_threshold"] = float(threshold)
            
            # Display metrics
            for metric_name in ['auroc', 'ap', 'auprc', 'f1_score', 'precision', 'recall', 'accuracy']:
                if metric_name in metrics:
                    print(f"  {metric_name}: {metrics[metric_name]:.4f}")
            
            # Add to all_metrics with _t_{horizon} suffix
            for key, value in metrics.items():
                all_metrics[f"{key}_t_{horizon}"] = value
        
        return all_metrics
    
    def _extract_labels_and_scores(self, df: pd.DataFrame, horizon: int) -> tuple:
        """
        Extract binary labels and risk scores from predictions DataFrame.
        
        Args:
            df: DataFrame with predictions
            horizon: Time horizon in days
        
        Returns:
            Tuple of (y_true, risk_score) or (None, None) if invalid
        """
        time = df["time"].values
        event = df["event"].values
        
        # Get S(t) column
        S_t_col = f"S_{int(horizon)}"
        if S_t_col not in df.columns:
            raise ValueError(f"Survival column {S_t_col} not found in predictions")
        
        S_t = df[S_t_col].values
        
        # 1. Exclude patients censored before t
        valid_mask = ~((event == 0) & (time < horizon))
        
        # 2. Define labels:
        #    - y=1 (positive): died at or before t
        #    - y=0 (negative): censored after t OR died after t
        y_true = np.zeros(len(df), dtype=int)
        y_true[(event == 1) & (time <= horizon)] = 1
        
        # Apply valid mask
        y_true_valid = y_true[valid_mask]
        S_t_valid = S_t[valid_mask]
        
        # Check for edge cases
        if len(y_true_valid) == 0:
            print(f"  Warning: No valid patients at t={horizon}")
            return None, None
        
        if len(np.unique(y_true_valid)) < 2:
            print(f"  Warning: Only one class present at t={horizon}")
            return None, None
        
        # Compute risk score (1 - S(t))
        risk_score = 1 - S_t_valid
        
        return y_true_valid, risk_score
    
    def _load_predictions(self, stage: str) -> pd.DataFrame:
        """Load predictions CSV from checkpoint_dir."""
        pred_path = os.path.join(
            self.checkpoint_dir, 
            "prediction", 
            f"predictions_{stage}.csv"
        )
        
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Predictions file not found: {pred_path}")
        
        return pd.read_csv(pred_path)
    
    def _find_best_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        Find threshold that maximizes F1 score.
        
        Args:
            y_true: True labels
            y_score: Predicted scores
        
        Returns:
            Best threshold value
        """
        thresholds = np.linspace(0, 1, 101)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold

