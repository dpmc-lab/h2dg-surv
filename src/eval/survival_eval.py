import os
import json
import pandas as pd
import numpy as np
from scipy.stats import chi2 
from typing import Optional, Literal, Dict, List
from sksurv.util import Surv
from sksurv.nonparametric import SurvivalFunctionEstimator
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
from SurvivalEVAL.Evaluations.util import predict_median_st, predict_rmst
from SurvivalEVAL.Evaluations.MeanError import mean_error

from src.data.datamodule.datamodule import HANCOCKDataModule
from src.data.utils import process_to_array
from src.eval.classification_eval import ClassificationEvaluator


class SurvivalEval:
    """
    Evaluation class for survival models.
    
    Computes standard survival analysis metrics:
    - C-IPCW: Concordance Index with Inverse Probability of Censoring Weighting
    - td-AUC: Time-dependent Area Under the Curve
    - IBS: Integrated Brier Score
    - KM Calibration: KL divergence between predicted and Kaplan-Meier survival curves
    - D-Calibration: Chi-squared calibration test
    - MAE: Mean Absolute Error (Margin, Hinge, Pseudo-observations, Uncensored methods)
    - MSE: Mean Squared Error (Margin, Hinge, Pseudo-observations, Uncensored methods)
    - RMSE: Root Mean Squared Error (Margin, Hinge, Pseudo-observations, Uncensored methods)
    
    Also computes binary classification metrics at specified time horizons:
    - AUROC: Area Under ROC Curve
    - AP: Average Precision
    - AUPRC: Area Under Precision-Recall Curve
    - F1-score, Precision, Recall, Accuracy (using optimal threshold from validation set)
    
    Expected prediction format (CSV):
    ```
    patient_id,time,event,risk_score,S_1,S_2,...,S_{T_max}
    P001,365,1,38.83,1.0,0.999,...,0.489
    P002,730,0,20.88,1.0,1.0,...,0.663
    ```
    
    Required columns:
    - patient_id: Patient identifier
    - time: Observed time (ground truth)
    - event: Event indicator (1=event occurred, 0=censored)
    - risk_score: Predicted risk scores (higher = higher risk)
    - S_1, S_2, ..., S_{T_max}: Survival probabilities at times 1, 2, ..., T_max
    """
    
    def __init__(
        self,
        datamodule: HANCOCKDataModule,
        checkpoint_dir: str,
        tau: int = 5 * 365,
        classification_horizon: Optional[List[int]] = [3*365, 5*365],
        T_max_prediction: int = 3650, # 10 * 365 days
    ):
        """
        Initialize SurvivalEval and prepare train/val data.
        
        Args:
            datamodule: HANCOCK datamodule (used to extract train data for IPCW)
            checkpoint_dir: Path to checkpoint directory (for loading predictions and saving results)
            tau: Truncation time for C-IPCW computation (default: 5 years)
            classification_horizon: Time horizon(s) for binary classification metrics in days.
                Can be an int or a list (e.g., [1095, 1825] for 3 and 5 years).
                If None, defaults to [3*365, 5*365]
            T_max_prediction: Maximum time for prediction filtering (default: 3650 days = 10 * 365 days)
        """
        self.datamodule = datamodule
        self.checkpoint_dir = checkpoint_dir
        self.tau = tau
        self.T_max_prediction = T_max_prediction
        self.classification_horizons = classification_horizon or [3*365, 5*365]
        if isinstance(self.classification_horizons, int):
            self.classification_horizons = [self.classification_horizons]
        
        # Prepare train data (for IPCW)
        self._prepare_train_data()
        
        # Initialize classification evaluator and find best threshold from validation data
        self.classification_evaluator = ClassificationEvaluator(
            checkpoint_dir=self.checkpoint_dir,
            horizons=self.classification_horizons
        )
        self.classification_evaluator.find_val_best_threshold()
        
        # Results storage
        self.metrics = {}
        self.current_stage = None
    
    def evaluate(
        self, 
        predictions: Optional[pd.DataFrame] = None,
        stage: Literal["train", "val", "test"] = "test"
    ) -> Dict[str, float]:
        """
        Compute all survival metrics.
        
        Args:
            predictions: DataFrame with predictions (if None, load from checkpoint_dir/prediction/predictions_{stage}.csv)
            stage: Which stage to evaluate (used only if predictions is None)
        
        Returns:
            Dictionary with metrics:
                - c_ipcw: Concordance index with IPCW
                - td_auc: Mean time-dependent AUC
                - ibs: Integrated Brier Score
                - km_calib: KM calibration (KL divergence)
                - d_calib_chi2: D-Calibration chi-squared statistic
                - d_calib_pvalue: D-Calibration p-value
                - mae_{method}_{pred}: MAE with different methods and predictions
                - mse_{method}_{pred}: MSE with different methods and predictions
                - rmse_{method}_{pred}: RMSE with different methods and predictions
                  (e.g. rmse_margin_median, rmse_margin_rmst, rmse_uncensored_median, etc.)
                - auroc_t_{horizon}: AUROC at time horizon (binary classification)
                - ap_t_{horizon}: Average Precision at time horizon (binary classification)
                - auprc_t_{horizon}: Area Under Precision-Recall Curve at time horizon (binary classification)
                - f1_score_t_{horizon}: F1-score at time horizon (binary classification)
                - precision_t_{horizon}: Precision at time horizon (binary classification)
                - recall_t_{horizon}: Recall at time horizon (binary classification)
                - accuracy_t_{horizon}: Accuracy at time horizon (binary classification)
                - n_valid_classif_t_{horizon}: Number of valid patients at time horizon
                - n_positive_classif_t_{horizon}: Number of positive cases at time horizon
                - best_val_threshold_t_{horizon}: Threshold used from validation set
        """
        print(f"\n{'='*60}\nSurvivalEval - Computing Metrics\n{'='*60}")
        
        # Store current stage
        self.current_stage = stage
        
        # Load predictions
        if predictions is None:
            print(f"Loading predictions from checkpoint_dir (stage={stage})...")
            predictions = self._load_predictions(stage)
        else:
            print("Using provided predictions DataFrame...")
            print(f"Stage: {stage}")
        
        print(f"Predictions shape: {predictions.shape}")
        
        # Validate predictions format
        self._validate_predictions(predictions)
        
        # Compute metrics
        print("\nComputing C-IPCW...")
        self._compute_c_ipcw(predictions)
        print(f"  C-IPCW: {self.metrics['c_ipcw']:.4f}")
        
        print("Computing td-AUC...")
        self._compute_td_auc(predictions)
        print(f"  td-AUC: {self.metrics['td_auc']:.4f}")
        
        print("Computing IBS...")
        self._compute_ibs(predictions)
        print(f"  IBS: {self.metrics['ibs']:.4f}")
        
        print("Computing KM Calibration...")
        self._compute_km_calibration(predictions)
        print(f"  KM Calibration: {self.metrics['km_calib']:.4f}")
        
        print("Computing D-Calibration...")
        self._compute_d_calibration(predictions)
        print(f"  D-Calibration: {self.metrics['d_calib_chi2']:.4f}")
        print(f"  D-Calibration p-value: {self.metrics['d_calib_pvalue']:.4f}")
        
        # Compute MAE for all combinations of methods and prediction types
        print("\nComputing MAE metrics...")
        mae_methods = ["Margin", "Hinge", "Pseudo_obs", "Uncensored"]
        prediction_methods = ["median", "rmst"]
        
        for mae_method in mae_methods:
            for pred_method in prediction_methods:
                metric_key = f"mae_{mae_method.lower()}_{pred_method}"
                print(f"  Computing {metric_key}...")
                self._compute_mae(predictions, method=mae_method, prediction_method=pred_method)
                print(f"    {metric_key}: {self.metrics[metric_key]:.2f}")
        
        # Compute MSE and RMSE for all combinations of methods and prediction types
        print("\nComputing MSE and RMSE metrics...")
        for mse_method in mae_methods:
            for pred_method in prediction_methods:
                mse_key = f"mse_{mse_method.lower()}_{pred_method}"
                rmse_key = f"rmse_{mse_method.lower()}_{pred_method}"
                print(f"  Computing {mse_key}...")
                self._compute_mae(predictions, method=mse_method, prediction_method=pred_method, error_type="squared")
                print(f"    {mse_key}: {self.metrics[mse_key]:.2f}")
                # Calculate RMSE as sqrt(MSE)
                self.metrics[rmse_key] = np.sqrt(self.metrics[mse_key])
                print(f"    {rmse_key}: {self.metrics[rmse_key]:.2f}")
        
        # Compute binary classification metrics at specified time horizons
        print("\nComputing Classification Metrics at Time Horizons...")
        classif_metrics = self.classification_evaluator.evaluate(predictions)
        self.metrics.update(classif_metrics)
        
        print(f"\n{'='*60}\n")
        
        return self.metrics
    
    def save(self, output_path: Optional[str] = None):
        """
        Save metrics to JSON file.
        
        Args:
            output_path: Path to save JSON file (if None, saves to checkpoint_dir/eval/metrics_{stage}.json)
        """
        if output_path is None:
            stage_suffix = f"_{self.current_stage}" if self.current_stage else ""
            output_path = os.path.join(self.checkpoint_dir, "eval", f"metrics{stage_suffix}.json")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Metrics saved to: {output_path}")
    
    def _prepare_train_data(self):
        """Extract y_train from datamodule using process_to_array."""
        _, time_train, event_train, _ = process_to_array(self.datamodule, stage="train")
        self.y_train = Surv.from_arrays(event=event_train, time=time_train)
        self.time_train = time_train
        self.event_train = event_train
        self.t_max_train = float(np.max(time_train))  # Max time for RMST
    
    def _load_predictions(self, stage: str) -> pd.DataFrame:
        """
        Load predictions CSV from checkpoint_dir/prediction/predictions_{stage}.csv.
        Filter survival columns S_t to keep only those with t <= T_max_prediction.
        """
        pred_path = os.path.join(
            self.checkpoint_dir, 
            "prediction", 
            f"predictions_{stage}.csv"
        )
        
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Predictions file not found: {pred_path}")
        
        df = pd.read_csv(pred_path)
        
        # Filter survival columns by T_max_prediction
        survival_cols = [c for c in df.columns if c.startswith("S_")]
        cols_to_keep = []
        cols_to_drop = []
        
        for col in survival_cols:
            time_point = int(col.split("_")[1])
            if time_point <= self.T_max_prediction:
                cols_to_keep.append(col)
            else:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            print(f"  Filtering survival columns: keeping {len(cols_to_keep)} columns (t <= {self.T_max_prediction}), dropping {len(cols_to_drop)} columns")
            df = df.drop(columns=cols_to_drop)
        
        return df
    
    def _validate_predictions(self, df: pd.DataFrame):
        """Validate that predictions DataFrame has required columns and covers T_max_prediction."""
        required_cols = ["patient_id", "time", "event", "risk_score"]
        survival_cols = [c for c in df.columns if c.startswith("S_")]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if len(survival_cols) == 0:
            raise ValueError("No survival columns (S_1, S_2, ...) found in predictions")
        
        # Check that predictions cover at least T_max_prediction
        max_time_predicted = max([int(c.split("_")[1]) for c in survival_cols])
        if max_time_predicted < self.T_max_prediction:
            raise ValueError(
                f"Predictions do not cover T_max_prediction: "
                f"max predicted time = {max_time_predicted} days, "
                f"T_max_prediction = {self.T_max_prediction} days. "
                f"Predictions must cover at least T_max_prediction."
            )
        
        print(f"  Validated: {len(required_cols)} required columns + {len(survival_cols)} survival columns")
        print(f"  Max predicted time: {max_time_predicted} days (T_max_prediction: {self.T_max_prediction} days)")
    
    def _compute_c_ipcw(self, df: pd.DataFrame) -> float:
        """
        Compute Concordance Index with IPCW.
        
        Measures discrimination: probability that model correctly ranks pairs of patients.
        Higher is better (range: 0.5-1.0, 0.5=random, 1.0=perfect).
        """
        y_test = Surv.from_arrays(event=df["event"].values, time=df["time"].values)
        risk_scores = df["risk_score"].values
        
        c_index_ipcw = concordance_index_ipcw(
            survival_train=self.y_train,
            survival_test=y_test,
            estimate=risk_scores,
            tau=self.tau
        )
        
        self.metrics["c_ipcw"] = float(c_index_ipcw[0])
        return float(c_index_ipcw[0])
    
    def _compute_td_auc(self, df: pd.DataFrame) -> float:
        """
        Compute time-dependent AUC (mean over time grid).
        
        Measures discrimination at different time points.
        Higher is better (range: 0.5-1.0).
        """
        y_train_times = np.array([x[1] for x in self.y_train])
        times = np.linspace(365, np.percentile(y_train_times, 95), num=100)
        
        y_test = Surv.from_arrays(event=df["event"].values, time=df["time"].values)
        risk_scores = df["risk_score"].values
        
        _, cd_auc = cumulative_dynamic_auc(
            survival_train=self.y_train,
            survival_test=y_test,
            estimate=risk_scores,
            times=times
        )
        
        self.metrics["td_auc"] = float(np.mean(cd_auc))
        return float(np.mean(cd_auc))
    
    def _compute_ibs(self, df: pd.DataFrame) -> float:
        """
        Compute Integrated Brier Score.
        
        Measures calibration: squared difference between predicted and actual survival.
        Lower is better (range: 0-1, 0=perfect).
        """
        y_test = Surv.from_arrays(event=df["event"].values, time=df["time"].values)
        
        # Extract survival columns and times
        survival_cols = sorted(
            [c for c in df.columns if c.startswith("S_")],
            key=lambda x: int(x.split("_")[1])
        )
        all_times = [int(c.split("_")[1]) for c in survival_cols]
        
        # Filter times to be within test data range
        y_test_times = df["time"].values
        min_time, max_time = y_test_times.min(), y_test_times.max()
        
        valid_cols = [c for c, t in zip(survival_cols, all_times) if min_time <= t < max_time]
        valid_times = [t for t in all_times if min_time <= t < max_time]
        
        if len(valid_times) == 0:
            raise ValueError("No valid time points found for IBS computation")
        
        survs = df[valid_cols].values
        
        ibs = integrated_brier_score(
            survival_train=self.y_train,
            survival_test=y_test,
            estimate=survs,
            times=valid_times
        )
        
        self.metrics["ibs"] = float(ibs)
        return float(ibs)
    
    def _compute_mae(
        self, 
        df: pd.DataFrame, 
        method: Literal["Margin", "Hinge", "Pseudo_obs", "Uncensored"] = "Margin",
        prediction_method: Literal["median", "rmst"] = "median",
        error_type: Literal["absolute", "squared"] = "absolute"
    ) -> float:
        """
        Compute Mean Absolute Error (MAE) or Mean Squared Error (MSE) using SurvivalEVAL.
        
        This metric measures the difference between predicted and observed survival times,
        accounting for censoring using various methods.
        
        Args:
            df: DataFrame with predictions
            method: Method for handling censoring:
                - "Margin": Margin-based method (recommended)
                - "Hinge": Hinge-based method
                - "Pseudo_obs": Pseudo-observations method
                - "Uncensored": Only on uncensored patients (baseline)
            prediction_method: How to derive predicted time from survival curve:
                - "median": Use median survival time (default)
                - "rmst": Use restricted mean survival time
            error_type: Type of error to compute:
                - "absolute": Mean Absolute Error (MAE)
                - "squared": Mean Squared Error (MSE)
        
        Returns:
            Error value (lower is better, MAE in days, MSE in days²)
        
        Reference: https://github.com/shi-ang/SurvivalEVAL
        """
        # Extract survival columns and times
        survival_cols = sorted(
            [c for c in df.columns if c.startswith("S_")],
            key=lambda x: int(x.split("_")[1])
        )
        times_coordinates = np.array([int(c.split("_")[1]) for c in survival_cols])
        
        # Extract survival curves [n_samples, n_times]
        survival_curves = df[survival_cols].values
        
        # Predict survival times using specified method
        if prediction_method == "median":
            predicted_times = predict_median_st(survival_curves, times_coordinates)
        elif prediction_method == "rmst":
            # For RMST, use max time from training set
            predicted_times = predict_rmst(survival_curves, times_coordinates, interpolation="None")
        else:
            raise ValueError(f"Unknown prediction_method: {prediction_method}")
        
        # Extract test event times and indicators
        event_times_test = df["time"].values
        event_indicators_test = df["event"].values.astype(int)
        
        # Extract train event times and indicators
        event_times_train = self.time_train
        event_indicators_train = self.event_train.astype(int)
        
        # Compute error using SurvivalEVAL
        error_value = mean_error(
            predicted_times=predicted_times,
            event_times=event_times_test,
            event_indicators=event_indicators_test,
            train_event_times=event_times_train,
            train_event_indicators=event_indicators_train,
            error_type=error_type,
            method=method,
            weighted=False,
            log_scale=False,
            verbose=False,
            truncated_time=None,
        )
        
        # Store in metrics with appropriate key: {error_type}_{method}_{prediction}
        error_prefix = "mae" if error_type == "absolute" else "mse"
        metric_key = f"{error_prefix}_{method.lower()}_{prediction_method}"
        self.metrics[metric_key] = float(error_value)
        
        return float(error_value)

    ####################################
    #   Calibration Metrics 
    ####################################

    def _compute_d_calibration(self, df: pd.DataFrame, bins: int = 10) -> float:
        """Compute D-Calibration chi-squared statistic."""
        # NOTE: No interpolation -> assumes each observed time is exactly on the survival grid.
        df = df.copy()
        df = df[df["time"] <= self.T_max_prediction]
        times = df["time"].values.astype(int)

        preds_at_obs_time = np.array([
           df.iloc[i][f"S_{int(t)}"] 
           for i, t in enumerate(times) 
        ])  # S_i(T_i)
        preds_at_obs_time = np.clip(preds_at_obs_time, 1e-12, 1)
        events = df["event"].values.astype(bool)

        out = self.d_calibration(event_indicators=events, predictions=preds_at_obs_time, bins=bins)

        self.metrics["d_calib_chi2"] = float(out["chi2_statistic"])
        self.metrics["d_calib_pvalue"] = float(out["p_value"])

        return float(out["chi2_statistic"])

    def d_calibration(
        self,
        event_indicators,
        predictions,
        bins: int = 10,
    ) -> dict:
        """
        D-Calibration by Haider et al.
        From the authors' implementation:
            https://github.com/haiderstats/survival_evaluation/blob/70e3a4d/survival_evaluation/evaluations.py#L111
        Returns the original outputs + the chi2 statistic `s`.
        """
        # include minimum to catch if probability = 1.
        bin_index = np.minimum(np.floor(predictions * bins), bins - 1).astype(int)
        censored_bin_indexes = bin_index[~event_indicators]
        uncensored_bin_indexes = bin_index[event_indicators]

        censored_predictions = predictions[~event_indicators]
        censored_contribution = 1 - (censored_bin_indexes / bins) * (
            1 / censored_predictions
        )
        censored_following_contribution = 1 / (bins * censored_predictions)

        contribution_pattern = np.tril(np.ones([bins, bins]), k=-1).astype(bool)

        following_contributions = np.matmul(
            censored_following_contribution, contribution_pattern[censored_bin_indexes]
        )
        single_contributions = np.matmul(
            censored_contribution, np.eye(bins)[censored_bin_indexes]
        )
        uncensored_contributions = np.sum(np.eye(bins)[uncensored_bin_indexes], axis=0)
        bin_count = (
            single_contributions + following_contributions + uncensored_contributions
        )
        chi2_statistic = np.sum(
            np.square(bin_count - len(predictions) / bins) / (len(predictions) / bins)
        )
        return dict(
            chi2_statistic=chi2_statistic,
            p_value=1 - chi2.cdf(chi2_statistic, bins - 1),
            bin_proportions=bin_count / len(predictions),
            censored_contributions=(single_contributions + following_contributions)
            / len(predictions),
            uncensored_contributions=uncensored_contributions / len(predictions),
        )
    
    def _compute_km_calibration(self, df: pd.DataFrame) -> float:
        """
        Compute KM calibration using KL divergence.
        
        Measures how well predicted survival matches Kaplan-Meier estimate.
        Lower is better (0=perfect calibration).
        
        Reference: Yanagisawa et al. (ICML 2023)
        """
        y_test = Surv.from_arrays(event=df["event"].values, time=df["time"].values)
        
        # Extract survival columns
        survival_cols = sorted(
            [c for c in df.columns if c.startswith("S_")],
            key=lambda x: int(x.split("_")[1])
        )
        all_times = [int(c.split("_")[1]) for c in survival_cols]
        
        # Filter times to be within test data range
        y_test_times = df["time"].values
        min_time, max_time = y_test_times.min(), y_test_times.max()
        
        valid_cols = [c for c, t in zip(survival_cols, all_times) if min_time <= t < max_time]
        valid_times = [t for t in all_times if min_time <= t < max_time]
        
        if len(valid_times) == 0:
            raise ValueError("No valid time points found for KM calibration computation")
        
        S_pred = df[valid_cols].values
        
        # Compute Kaplan-Meier on test set
        km_estimator = SurvivalFunctionEstimator(conf_type=None)
        km_estimator.fit(y_test)
        S_km = km_estimator.predict_proba(valid_times, return_conf_int=False)
        
        # KL divergence between KM and predicted survival
        kl_div = self._km_calibration_kl(S_pred, S_km, valid_times, B=30)
        self.metrics["km_calib"] = float(kl_div)
        return float(kl_div)
    
    def _km_calibration_kl(
        self, 
        S_pred: np.ndarray, 
        S_km: np.ndarray, 
        all_times: list, 
        B: Optional[int] = None, 
        eps: float = 1e-6
    ) -> float:
        """
        KM-calibration KL divergence metric.
        
        Args:
            S_pred: Predicted survival matrix [n_samples, n_times]
            S_km: Kaplan-Meier survival curve [n_times]
            all_times: Time grid
            B: Number of bins for time discretization (if None, use all times)
            eps: Small value for numerical stability
        
        Returns:
            KL divergence (lower is better)
        """
        all_times_arr = np.asarray(all_times, float)
        
        if B is None:
            # Use all available times
            grid = all_times_arr
        else:
            # Create B evenly spaced time points
            grid_times = np.linspace(all_times[0], all_times[-1], B)
            
            # Map grid times to nearest indices in all_times
            indices = np.searchsorted(all_times_arr, grid_times, side='left')
            indices = np.clip(indices, 0, len(all_times) - 1)
            
            # Interpolate to grid using indices
            S_pred = S_pred[:, indices]
            S_km = S_km[indices]
            grid = all_times_arr[indices]
        
        # Mean predicted survival
        S_bar = S_pred.mean(axis=0)
        
        # Augment with 0 at the end to close distribution
        S_km_aug = np.append(S_km, 0.0)
        S_bar_aug = np.append(S_bar, 0.0)
        
        # Event masses per interval
        p = S_km_aug[:-1] - S_km_aug[1:]
        q = S_bar_aug[:-1] - S_bar_aug[1:]
        
        # Clip for numerical stability
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        
        # KL divergence: KL(p || q) = sum(p * log(p / q))
        kl_div = np.sum(p * np.log(p / q))
        
        return float(kl_div)
