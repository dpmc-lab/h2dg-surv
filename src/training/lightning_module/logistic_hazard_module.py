import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, Tuple, Optional, Literal
from src.training.lightning_module.survival_base_module import SurvivalBaseLightningModule
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models.loss import NLLLogistiHazardLoss
from src.utils.xcal import d_calibration


class LogisticHazardModule(SurvivalBaseLightningModule):
    """
    Lightning Module for HANCOCK multimodal Logistic Hazard survival prediction.
    
    Logistic Hazard is a discrete-time survival model that:
    - Discretizes time into intervals/bins
    - Models the hazard probability in each interval via logistic regression
    - Uses NLL loss for training
    
    Approach:
    - Time discretization: Configured via num_bins (from model)
    - Model produces num_bins logits, one per time interval
    """
    
    def __init__(
        self,
        model: nn.Module,
        # Optimizer parameters
        optimizer_name: str = "AdamW",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        # Scheduler parameters
        scheduler_name: str = "ConstantLR",
        step_size: int = 30,
        gamma: float = 0.1,
        factor: float = 1.0,
        last_epoch: int = -1,
        # Metrics parameters
        main_metric: str = "val_loss",
        # Parameters specific to LogisticHazard
        T_max: int = 10 * 365,  # T-max for survival function grid
        scheme: Literal["equidistant", "quantiles"] = "equidistant",
        # Temperature scaling for calibration
        temperature: float = 1.0,
        # X-CAL calibration parameters
        lambda_xcal: float = 0.0,  # Weight for X-CAL calibration penalty (0 = disabled)
        xcal_nbins: int = 20,       # Number of bins for X-CAL histogram (authors use 20)
        xcal_gamma: float = 1e3,    # Softness of X-CAL bin membership (higher = sharper)
        # Ranking loss parameters (concordance)
        lambda_ranking: float = 0.0,  # Weight for ranking loss (0 = disabled)
        ranking_margin: float = 0.0,  # Margin for hinge loss (0 = no margin)
        **kwargs
    ):
        # Initialize base class with all training configuration
        super().__init__(
            model=model,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            scheduler_name=scheduler_name,
            step_size=step_size,
            gamma=gamma,
            factor=factor,
            last_epoch=last_epoch,
            main_metric=main_metric,
            **kwargs
        )
        
        self.T_max = T_max
        self.scheme = scheme
        
        # Temperature scaling for calibration
        self.temperature = temperature
        
        # Discretization cuts (fitted on training data)
        # Will be set in on_fit_start()
        self.register_buffer("cuts", torch.tensor([]), persistent=True)
        
        # Logistic hazard loss
        self.loss_fn = NLLLogistiHazardLoss(reduction="mean")
        
        # X-CAL calibration parameters
        self.lambda_xcal = lambda_xcal
        self.xcal_nbins = xcal_nbins
        self.xcal_gamma = xcal_gamma
        
        # Ranking loss parameters
        self.lambda_ranking = lambda_ranking
        self.ranking_margin = ranking_margin

    ########################################################
    # Lightning Hooks
    ########################################################
    
    def _unpack_batch(self, batch) -> Tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor]:
        """
        Unpack batch into components.
        
        Override this method in subclasses for different batch formats (e.g., PyG HeteroData).
        
        Args:
            batch: Batch from dataloader
            
        Returns:
            Tuple of (patient_id, inputs, time, event)
        """
        patient_id, inputs, targets = batch
        time, event = targets
        return patient_id, inputs, time, event
    
    def forward(self, inputs: Any) -> torch.Tensor:
        """
        Forward pass through the model with temperature scaling.
        
        Args:
            inputs: Model inputs (format depends on model type)
        
        Returns:
            Logits for discrete time bins [batch_size, num_bins]
        """
        logits = self.model(inputs)  # [batch_size, num_bins]
        logits = logits / self.temperature
        
        return logits
    
    def on_fit_start(self) -> None:
        """
        Fit LabTransDiscreteTime on training data at the beginning of training.
        
        This creates the discretization bins and stores the cut points.
        Uses pycox for fitting (correct bin placement) but stores cuts as torch tensor
        for GPU-native transformation during training.
        """
        print("\n" + "="*60)
        print("Fitting LabTransDiscreteTime on training data...")
        print("="*60)
        
        # Extract training times and events
        train_times, train_events = self._extract_train_labels()
        
        # Fit LabTransDiscreteTime with pycox
        labtrans = LabTransDiscreteTime(self.model.num_bins, scheme=self.scheme)
        labtrans.fit(train_times, train_events)
        
        # HACK: With "quantiles" scheme, pycox may create fewer bins than requested due to duplicate
        # survival times in the training data (e.g., requesting 50 bins but getting only 39 unique cuts).
        # Since the model was already created with num_bins output logits, we need to
        # adjust the model's last layer to match the actual number of bins created by pycox.
        # This ensures dimension compatibility during training (model outputs must match labtrans.out_features).
        actual_num_bins = labtrans.out_features
        if actual_num_bins != self.model.num_bins:
            print(f"WARNING: Requested {self.model.num_bins} bins, but pycox created only {actual_num_bins} unique bins.")
            print(f"   Adjusting model output layer from {self.model.num_bins} to {actual_num_bins} bins...")
            if self.scheme != "quantiles":
                print("This should not happen if scheme is not 'quantiles'!!!")
            
            # Recreate the last layer of hazard_head with correct output dimension
            old_linear = self.model.hazard_head[-1]
            new_linear = torch.nn.Linear(old_linear.in_features, actual_num_bins)
            
            new_linear = new_linear.to(next(self.model.parameters()).device)
            
            self.model.hazard_head[-1] = new_linear
            self.model.num_bins = actual_num_bins
            
            # Update optimizer to include the new layer's parameters
            # The optimizer was already created in configure_optimizers(), so we need to add
            # the new parameters to it
            if self.trainer and self.trainer.optimizers:
                optimizer = self.trainer.optimizers[0]
                optimizer.add_param_group({'params': new_linear.parameters()})
        
        # Store cuts as torch tensor (for GPU-native transformation)
        # Add T_max as the final cut to close the last bin
        cuts_with_final = np.append(labtrans.cuts, train_times.max())
        self.cuts = torch.from_numpy(cuts_with_final).float()
        
        print(f"Discretization completed:")
        print(f"  - Number of bins: {self.model.num_bins}")
        print(f"  - Number of cuts: {len(self.cuts)}")
        print(f"  - Cut points (first 10): {self.cuts[:10].tolist()}")
        print(f"  - Cut points (last 10): {self.cuts[-10:].tolist()}")
        print(f"  - Time range: [{self.cuts[0]:.1f}, {self.cuts[-1]:.1f}] days")
        
        # Setup survival metrics (IBS, etc.) using parent class
        time_coordinates = labtrans.cuts  # Use cuts as time coordinates for IBS
        self.setup_survival_metrics(train_times, train_events, time_coordinates)
        
        print("="*60 + "\n")
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step with survival metrics computation."""
        outputs = self._shared_step(batch, "train")
        
        # Cache predictions for survival metrics computation at epoch end
        self.cache_survival_predictions(
            survival_probs=outputs["survival_probs"],
            times=outputs["times"],
            events=outputs["events"],
            risk_scores=outputs["risk_scores"],
            stage="train"
        )
        
        return outputs["loss"]
    
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step with survival metrics computation."""
        outputs = self._shared_step(batch, "val")
        
        # Cache predictions for survival metrics computation at epoch end
        self.cache_survival_predictions(
            survival_probs=outputs["survival_probs"],
            times=outputs["times"],
            events=outputs["events"],
            risk_scores=outputs["risk_scores"],
            stage="val"
        )
        
        return outputs["loss"]
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Generate predictions for a single batch.
        
        Note: Predictions aggregation and saving to disk are handled by 
        `SurvivalPredictionWriter` callback (see `callbacks.py`).
        
        Returns:
            Dictionary with predictions and targets for the batch.
            Includes survival probabilities at each discrete time interval.
        """
        patient_id, inputs, time, event = self._unpack_batch(batch)
        with torch.no_grad():
            logits = self.forward(inputs)  # [batch_size, num_bins]
            
            # Compute hazard probabilities from logits
            hazard_probs = torch.sigmoid(logits)  # [batch_size, num_bins]
            
            # Compute survival function: S(t) = prod_{j<=t} (1 - h_j)
            # where h_j is the hazard probability at interval j
            survival_probs = torch.cumprod(1 - hazard_probs, dim=1)  # [batch_size, num_bins]
            
            # Interpolate survival to fine-grained grid [1, 2, ..., T_max]
            t_grid = torch.arange(1, self.T_max + 1, device=survival_probs.device)
            
            # Map each time point to its corresponding bin
            surv_interpolated = self._interpolate_survival(
                survival_probs, 
                self.cuts, 
                t_grid
            )
            
            # Convert discrete survival to dictionary for storage
            # Keys are 'S_1', 'S_2', ..., 'S_{T_max}' for days 1, 2, ..., T_max
            S_of_t_dict = {}
            for i, t_k in enumerate(t_grid):
                S_of_t_dict[f"S_{int(t_k)}"] = surv_interpolated[:, i]
            
            # Expected survival time: E[T] = sum_t S(t) * dt
            # Here t_grid has step=1, so dt=1 for all bins
            T_hat = torch.sum(surv_interpolated, dim=1)  # [batch_size]
            
            output = {
                "p_id": patient_id,
                "time": time,
                "event": event,
                "risk_scores": -T_hat,  # Higher risk = lower expected survival
            }
            
            output.update(S_of_t_dict)
            return output
    
    ########################################################
    # Utility Methods
    ########################################################
    
    def _shared_step(
        self, 
        batch: Tuple[Tuple[torch.Tensor, ...], torch.Tensor], 
        stage: str
    ) -> Dict[str, torch.Tensor]:
        """
        Shared step for train/val/test.
        
        Args:
            batch: Batch from dataloader (format depends on dataset type)
                  - For standard: Tuple of (patient_id, inputs, (time, event))
            stage: "train", "val", or "test"
            
        Returns:
            Dictionary with loss
        """
        patient_id, inputs, time, event = self._unpack_batch(batch)
        logits = self.forward(inputs)  # [batch_size, num_bins]
        
        # Get batch size
        batch_size = patient_id.size(0) if torch.is_tensor(patient_id) else len(patient_id)
        
        # Transform continuous times to discrete bin indices (GPU-native)
        time_discrete = self._transform_torch(time)  # [batch_size]
        
        # Logistic hazard loss
        # NLLLogistiHazardLoss expects:
        #   - phi: [batch_size, num_bins] logits
        #   - idx_durations: [batch_size] time indices (long)
        #   - events: [batch_size] event indicators (bool or float)
        nll_loss = self.loss_fn(
            phi=logits,
            idx_durations=time_discrete.long(),
            events=event.bool()
        )
        
        # X-CAL calibration penalty (only during training if enabled)
        if self.lambda_xcal > 0 and stage == "train":
            # Compute survival probabilities
            hazard_probs = torch.sigmoid(logits)  # [batch_size, num_bins]
            survival_probs = torch.cumprod(1 - hazard_probs, dim=1)  # [batch_size, num_bins]
            
            # Compute CDF at observed times
            cdf_at_times = self._compute_cdf_at_observed_times(survival_probs, time)
            
            # X-CAL convention: is_alive = 1 for censored, 0 for event
            # Our convention: event = 1 for event, 0 for censored
            # So: is_alive = 1 - event
            is_alive = 1.0 - event.float()
            
            # Compute X-CAL penalty
            xcal_penalty = d_calibration(
                points=cdf_at_times,
                is_alive=is_alive,
                args=None,
                nbins=self.xcal_nbins,
                differentiable=True,
                gamma=self.xcal_gamma,
                device=self.device
            )
            
            # Total loss with X-CAL regularization
            loss = nll_loss + self.lambda_xcal * xcal_penalty
            
            # Log both losses separately
            self.log(f'{stage}/nll_loss', nll_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log(f'{stage}/xcal_penalty', xcal_penalty, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        else:
            # No X-CAL: just use NLL
            loss = nll_loss
        
        # Compute risk scores (needed for ranking loss and metrics)
        # Risk score = -E[T], where E[T] is expected survival time
        hazard_probs = torch.sigmoid(logits)  # [batch_size, num_bins]
        survival_probs = torch.cumprod(1 - hazard_probs, dim=1)  # [batch_size, num_bins]
        bin_widths = self._compute_bin_widths()  # [num_bins]
        expected_survival_time = torch.sum(survival_probs * bin_widths.unsqueeze(0), dim=1)  # [batch_size]
        risk_scores = -expected_survival_time  # Higher risk = shorter expected survival
        
        # Ranking loss (concordance penalty, only during training if enabled)
        if self.lambda_ranking > 0 and stage == "train":
            # Compute ranking matrix
            rank_mat = self._compute_ranking_matrix(time, event)
            
            # Compute ranking loss using risk scores
            ranking_loss = self._compute_ranking_loss_hinge(
                risk_scores=risk_scores,
                rank_mat=rank_mat,
                margin=self.ranking_margin
            )
            
            # Add to total loss
            loss = loss + self.lambda_ranking * ranking_loss
            
            # Log ranking loss separately
            self.log(f'{stage}/ranking_loss', ranking_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        
        # Loss logging
        self.accumulate_and_log_loss(loss, phase=stage, batch_size=event.size(0))
        
        outputs = {
            "loss": loss,
            "survival_probs": survival_probs,
            "times": time,
            "events": event,
            "risk_scores": risk_scores
        }
        
        return outputs
    
    def _compute_bin_widths(self) -> torch.Tensor:
        """
        Compute the width of each bin from cuts.
        
        With N bins, we have N+1 cuts (including T_max as the final cut).
        Bin i has width = cuts[i+1] - cuts[i]
        
        Returns:
            bin_widths: [num_bins] width of each bin
        """
        # cuts: [cut_0, cut_1, ..., cut_N] where N = num_bins
        # bin i has width = cuts[i+1] - cuts[i] for i in [0, N-1]
        bin_widths = self.cuts[1:] - self.cuts[:-1]  # [num_bins]
        
        return bin_widths.to(self.device)
    
    def _transform_torch(self, times: torch.Tensor) -> torch.Tensor:
        """
        Transform continuous times to discrete bin indices (GPU-native).
        
        This is a PyTorch implementation of LabTransDiscreteTime.transform()
        that avoids CPU/GPU transfers during training.
        
        Args:
            times: [batch_size] continuous times (torch.Tensor on GPU)
        
        Returns:
            bin_indices: [batch_size] discrete indices in [0, num_bins-1]
        """
        # Move cuts to same device as times
        cuts = self.cuts.to(times.device)
        
        # searchsorted: find where to insert times in cuts
        # right=False: if times[i] == cuts[j], returns j
        idx = torch.searchsorted(cuts, times, right=False)
        
        # Convert cut index -> bin index
        # and clamp to handle edge cases (times < cuts[0] or > cuts[-1])
        bin_idx = torch.clamp(idx - 1, min=0, max=self.model.num_bins - 1)
        
        return bin_idx
    
    def _interpolate_survival(
        self, 
        survival_at_bins: torch.Tensor,
        cuts: torch.Tensor,
        t_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate survival probabilities from discrete bins to fine-grained time grid.
        
        Uses step function (right-continuous): S(t) = S(bin_j) for t in [cut_j, cut_{j+1})
        
        Args:
            survival_at_bins: [batch_size, num_bins] survival at each bin
            cuts: [num_bins + 1] bin boundaries
            t_grid: [T] target time points
        
        Returns:
            survival_at_grid: [batch_size, T] survival at each time point
        """
        device = survival_at_bins.device
        batch_size, num_bins = survival_at_bins.shape
        T = t_grid.size(0)
        
        # Move tensors to same device
        cuts = cuts.to(device)
        t_grid = t_grid.to(device).float()
        
        # For each time point in t_grid, find which bin it belongs to
        # searchsorted with right=True gives right-continuous step function
        bin_indices = torch.searchsorted(cuts, t_grid, right=True)
        bin_indices = torch.clamp(bin_indices - 1, min=0, max=num_bins - 1)
        
        # Gather survival values from corresponding bins
        # bin_indices: [T] -> expand to [batch_size, T] for gather
        bin_indices_expanded = bin_indices.unsqueeze(0).expand(batch_size, -1)
        survival_at_grid = torch.gather(survival_at_bins, 1, bin_indices_expanded)
        
        return survival_at_grid
    
    def _compute_cdf_at_observed_times(
        self,
        survival_probs: torch.Tensor,  # [batch_size, num_bins]
        times: torch.Tensor,            # [batch_size] observed times
    ) -> torch.Tensor:
        """
        Compute CDF F(u_i|x_i) = 1 - S(u_i|x_i) at observed times u_i.
        
        Required for X-CAL calibration, which needs the CDF value at the exact
        observed time (event or censoring time) for each patient.
        
        Uses step function interpolation: for time u_i in bin j, we use S(bin_j).
        
        Args:
            survival_probs: [batch_size, num_bins] S(t_j) at bin end boundaries
            times: [batch_size] observed times u_i (event or censoring)
        
        Returns:
            cdf_at_times: [batch_size] F(u_i|x_i) for each patient, clamped to [eps, 1-eps]
        """
        batch_size = survival_probs.shape[0]
        device = survival_probs.device
        
        # Get current bin boundaries
        cuts = self.cuts.to(device)
        
        # For each time, find which bin it belongs to
        # searchsorted with right=True: bin_j = smallest j where cuts[j] > time
        # Then bin_indices - 1 gives the bin that contains the time
        bin_indices = torch.searchsorted(cuts, times, right=True)
        bin_indices = torch.clamp(bin_indices - 1, min=0, max=self.model.num_bins - 1)
        
        # Gather survival values at corresponding bins
        # survival_probs[i, bin_indices[i]] gives S(bin_j) where u_i is in bin j
        survival_at_times = survival_probs.gather(1, bin_indices.unsqueeze(1)).squeeze(1)
        
        # Convert to CDF: F(t) = 1 - S(t)
        cdf_at_times = 1.0 - survival_at_times
        
        # Clamp to [eps, 1-eps] for numerical stability in X-CAL
        # (avoids log(0) and division by 0 issues)
        eps = 1e-6
        cdf_at_times = cdf_at_times.clamp(eps, 1.0 - eps)
        
        return cdf_at_times
    
    def _compute_ranking_matrix(
        self,
        times: torch.Tensor,
        events: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ranking matrix R for concordance/ranking loss.
        
        R[i, j] = 1 if patient i should be ranked higher risk than patient j:
                = 1 if event[i]=1 AND time[i] < time[j]
                = 0 otherwise
        
        This identifies valid pairs where:
        - Patient i has an observed event (not censored)
        - Patient i's event time is earlier than patient j's time
        
        For these pairs, we expect: risk[i] > risk[j]
        (higher risk = shorter expected survival)
        
        Args:
            times: [batch_size] observed times (event or censoring)
            events: [batch_size] event indicators (1=event, 0=censored)
        
        Returns:
            rank_mat: [batch_size, batch_size] binary matrix
        """
        batch_size = times.size(0)
        
        # time[i] < time[j] for all pairs (i, j)
        # Shape: [batch_size, batch_size]
        time_i_lt_time_j = (times.unsqueeze(1) < times.unsqueeze(0)).float()
        
        # event[i] = 1 for all pairs (i, j)
        # Shape: [batch_size, batch_size]
        event_i = events.unsqueeze(1).expand(batch_size, batch_size).float()
        
        # R[i, j] = 1 if event[i]=1 AND time[i] < time[j]
        rank_mat = event_i * time_i_lt_time_j
        
        return rank_mat
    
    def _compute_ranking_loss_hinge(
        self,
        risk_scores: torch.Tensor,
        rank_mat: torch.Tensor,
        margin: float = 0.0
    ) -> torch.Tensor:
        """
        Compute hinge ranking loss for concordance.
        
        For pairs where rank_mat[i,j]=1, we want risk[i] > risk[j].
        Hinge loss penalizes violations: max(0, risk[j] - risk[i] + margin)
        
        This is a margin-based ranking loss similar to SVM:
        - If risk[i] > risk[j] + margin (correct order with margin), loss = 0
        - If risk[i] ≤ risk[j] + margin (wrong or marginal order), loss > 0
        
        Args:
            risk_scores: [batch_size] predicted risk scores (higher = worse prognosis)
            rank_mat: [batch_size, batch_size] binary matrix of valid pairs
            margin: Margin parameter for hinge loss (default: 0.0)
        
        Returns:
            Scalar ranking loss
        """
        # Pairwise risk differences: risk[j] - risk[i]
        # Shape: [batch_size, batch_size]
        # If risk[i] > risk[j], this is negative (good)
        # If risk[i] < risk[j], this is positive (bad)
        diff = risk_scores.unsqueeze(0) - risk_scores.unsqueeze(1)
        
        # Hinge loss: max(0, risk[j] - risk[i] + margin)
        # Penalizes when risk[i] is not sufficiently larger than risk[j]
        violations = torch.relu(diff + margin)
        
        # Apply only to valid pairs (rank_mat=1)
        ranking_loss = (violations * rank_mat).sum()
        
        # Normalize by number of valid pairs
        n_valid_pairs = rank_mat.sum().clamp(min=1.0)
        
        return ranking_loss / n_valid_pairs
    
    def _extract_train_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract training times and events from datamodule.
        
        Uses _unpack_batch() to handle different batch formats (standard tuples vs PyG HeteroData).
        
        Returns:
            train_times: [N] numpy array of training times
            train_events: [N] numpy array of training event indicators
        """
        train_dataset = self.trainer.datamodule.train_dataset
        
        train_times = []
        train_events = []
        
        print(f"Extracting labels from {len(train_dataset)} training samples...")
        
        for i in range(len(train_dataset)):
            # Use _unpack_batch to handle different formats
            _, _, time, event = self._unpack_batch(train_dataset[i])
            train_times.append(time.item() if torch.is_tensor(time) else time)
            train_events.append(event.item() if torch.is_tensor(event) else event)
        
        return np.array(train_times), np.array(train_events)

