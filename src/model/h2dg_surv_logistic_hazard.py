import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from torch_geometric.nn import HeteroConv, GATv2Conv, LayerNorm
from torch_geometric.utils import scatter
from transformers import AutoModel


class H2DGSurvLogisticHazard(nn.Module):
    """
    H2DGSurv: Hierarchical Heterogeneous Directed Graph survival model for HANCOCK multimodal prediction.
    
    Architecture follows the clinical pathway with temporal progression:
    - Step 1: Diagnostic (clinical + blood)
    - Step 2: Pathology & Imaging (pathological + TMA + lymph + tumor)
    - Step 3: History
    - Step 4: Surgery (surgery report + description)
    
    Compatible with LogisticHazardModule (outputs logits for discrete time bins).
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        num_bins: int,
        hidden_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.3,
        lm_path: str = "./data/models/Bio_ClinicalBERT",
        freeze_text_encoder: bool = True,
        hidden_dims: Optional[List[int]] = None,
        hidden_reduction_factors: Optional[List[int]] = None,
        **kwargs
    ):
        """
        Initialize H2DGSurv model.
        
        Args:
            input_dims: Dictionary with input dimensions for each leaf node type
            num_bins: Number of discrete time bins for logistic hazard
            hidden_dim: Hidden dimension for GNN layers (default: 512)
            num_heads: Number of attention heads for GATv2Conv (default: 4)
            dropout: Dropout rate (default: 0.3)
            lm_path: Path to language model for text encoding
            freeze_text_encoder: Whether to freeze text encoder weights
            hidden_dims: List of hidden dimensions for hazard head (alternative to hidden_reduction_factors)
            hidden_reduction_factors: List of reduction factors for hazard head (e.g., [2, 4] means hidden_dim//2, hidden_dim//4)
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # ============================================
        # Text encoder (BioClinical BERT) - shared for all text nodes
        # ============================================
        self.text_model_path = lm_path
        self.text_encoder = AutoModel.from_pretrained(lm_path, local_files_only=True)
        
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        self.text_dim = self.text_encoder.config.hidden_size  # 768 for BERT-base
        
        # ============================================
        # Leaf node encoders - separate for each sub-type
        # ============================================
        self.leaf_encoders = nn.ModuleDict({
            # Structured leaves (separate encoders)
            'clinical': nn.Linear(input_dims['clinical'], hidden_dim),
            'blood': nn.Linear(input_dims['blood'], hidden_dim),
            'pathological': nn.Linear(input_dims['pathological'], hidden_dim),
            'tma': nn.Linear(input_dims['cdm'], hidden_dim),
            # Image leaves (separate encoders)
            'lymph': nn.Linear(input_dims['lymphnode'], hidden_dim),
            'tumor': nn.Linear(input_dims['primarytumor'], hidden_dim),
            # Text leaves (separate encoders after BERT)
            'history': nn.Linear(self.text_dim, hidden_dim),
            'surgery_report': nn.Linear(self.text_dim, hidden_dim),
            'surgery_desc': nn.Linear(self.text_dim, hidden_dim),
        })
        
        # ============================================
        # Layer 1: Leaf nodes -> Steps
        # ============================================
        # All leaves have hidden_dim after encoding, steps start with hidden_dim
        self.conv1 = HeteroConv({
            # Step 1: clinical + blood
            ('clinical', 'to_step1', 'step1'): 
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False),
            ('blood', 'to_step1', 'step1'): 
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False),
            
            # Step 2: pathological + tma + images
            ('pathological', 'to_step2', 'step2'): 
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False),
            ('tma', 'to_step2', 'step2'): 
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False),
            ('lymph', 'to_step2', 'step2'): 
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False),
            ('tumor', 'to_step2', 'step2'): 
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False),
            
            # Step 3: history
            ('history', 'to_step3', 'step3'): 
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False),
            
            # Step 4: surgery reports
            ('surgery_report', 'to_step4', 'step4'): 
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False),
            ('surgery_desc', 'to_step4', 'step4'): 
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False),
        }, aggr='mean')
        
        # ============================================
        # Layer 2: Temporal Progression + Skip Connections + Self-loops
        # ============================================        
        temporal_conv = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False)
        skip_conv = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False)
        self_conv = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False)
    
        self.conv2 = HeteroConv({
            # Temporal progression
            ('step1', 'temporal_next', 'step2'): temporal_conv,
            ('step2', 'temporal_next', 'step3'): temporal_conv,
            ('step3', 'temporal_next', 'step4'): temporal_conv,
            
            # Skip connections
            ('step1', 'skip_2', 'step3'): skip_conv,
            ('step1', 'skip_3', 'step4'): skip_conv,
            ('step2', 'skip_2', 'step4'): skip_conv,
            
            # Self-loops
            ('step1', 'self', 'step1'): self_conv,
            ('step2', 'self', 'step2'): self_conv,
            ('step3', 'self', 'step3'): self_conv,
            ('step4', 'self', 'step4'): self_conv,
        }, aggr='mean')
        
        # ============================================
        # Layer normalization for Layer 2
        # ============================================
        self.layer_norms_conv2 = nn.ModuleDict({
            'step1': LayerNorm(hidden_dim),
            'step2': LayerNorm(hidden_dim),
            'step3': LayerNorm(hidden_dim),
            'step4': LayerNorm(hidden_dim),
        })
        
        # ============================================
        # Layer 3: All Steps -> Master (Final Aggregation)
        # ============================================
        # Each step type aggregates to master with specialized attention
        self.conv3 = HeteroConv({
            ('step1', 'to_master', 'master'): 
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False),
            ('step2', 'to_master', 'master'): 
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False),
            ('step3', 'to_master', 'master'): 
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False),
            ('step4', 'to_master', 'master'): 
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False),
            ('master', 'self', 'master'): 
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False),
        }, aggr='mean')
        
        # ============================================
        # Activation and Dropout
        # ============================================
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        
        # ============================================
        # Hazard head (for logistic hazard survival prediction)
        # Same logic as MLPLogisticHazard
        # ============================================
        
        # Determine hidden dimensions for hazard head
        if hidden_dims is not None and hidden_reduction_factors is not None:
            print("WARNING: Both 'hidden_dims' and 'hidden_reduction_factors' are specified.")
            print("         Using 'hidden_reduction_factors' and ignoring 'hidden_dims'.")
            final_hidden_dims = [
                hidden_dim // factor 
                for factor in hidden_reduction_factors
            ]
        elif hidden_reduction_factors is not None:
            final_hidden_dims = [
                hidden_dim // factor 
                for factor in hidden_reduction_factors
            ]
        elif hidden_dims is not None:
            final_hidden_dims = hidden_dims
        else:
            # Default behavior
            hidden_reduction_factors = [2, 4]
            final_hidden_dims = [
                hidden_dim // factor 
                for factor in hidden_reduction_factors
            ]
        
        # Build hazard head dynamically
        layers = []
        prev_dim = hidden_dim
        
        for hid_dim in final_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hid_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hid_dim
        
        # Final layer to num_bins
        layers.append(nn.Linear(prev_dim, num_bins))
        
        self.hazard_head = nn.Sequential(*layers)
        
        # Log architecture
        print(f"\n>>> H2DGSurvLogisticHazard initialized:")
        print(f"    - Architecture: Fully Heterogeneous (9 leaf types) + Temporal Steps")
        print(f"    - GNN hidden dim: {hidden_dim}")
        print(f"    - Layer 2: LayerNorm + Conv + Residual")
        print(f"    - Num heads: {num_heads}")
        print(f"    - Dropout: {dropout}")
        print(f"    - Num bins: {num_bins}")
        print(f"    - Text encoder: {lm_path} (frozen: {freeze_text_encoder})")
        print(f"    - Leaf types: clinical, blood, pathological, tma, lymph, tumor, history, surgery_report, surgery_desc")
        print(f"    - Hazard head architecture:")
        print(f"        Input dim: {hidden_dim}")
        print(f"        Hidden dims: {final_hidden_dims}")
        print(f"        Output dim: {num_bins}")
        print(f"        Total layers: {len(final_hidden_dims) + 1}")
    
    def _encode_text_nodes(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode text using BioClinical BERT.
        
        Args:
            input_ids: Token IDs [num_text_nodes, seq_len]
            attention_mask: Attention mask [num_text_nodes, seq_len]
            
        Returns:
            Text embeddings [num_text_nodes, text_dim]
        """
        with torch.set_grad_enabled(self.text_encoder.training):
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Use [CLS] token representation
            text_embeddings = outputs.last_hidden_state[:, 0, :]  # [num_text_nodes, text_dim]
        
        return text_embeddings
    
    def forward(
        self, 
        batch, 
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through hierarchical directed survival GNN.
        
        Args:
            batch: PyG HeteroBatch object containing all node features
            edge_index_dict: Edge indices dict (from PyG Batch)
            
        Returns:
            Logits for discrete time bins [batch_size, num_bins]
        """
        device = next(self.parameters()).device
        
        # Get batch size (number of patients in batch)
        # Each patient has 1 master node
        batch_size = batch['master'].x.size(0)
        
        # ============================================
        # Encode leaf nodes to hidden_dim
        # ============================================
        x_dict_encoded = {}
        
        # ═══════════════════════════════════════════
        # STRUCTURED NODES 
        # ═══════════════════════════════════════════
        x_dict_encoded['clinical'] = self.dropout_layer(
            self.activation(self.leaf_encoders['clinical'](batch['clinical'].x))
        )  # [batch_size, hidden_dim]
        
        x_dict_encoded['blood'] = self.dropout_layer(
            self.activation(self.leaf_encoders['blood'](batch['blood'].x))
        )  # [batch_size, hidden_dim]
        
        x_dict_encoded['pathological'] = self.dropout_layer(
            self.activation(self.leaf_encoders['pathological'](batch['pathological'].x))
        )  # [batch_size, hidden_dim]
        
        x_dict_encoded['tma'] = self.dropout_layer(
            self.activation(self.leaf_encoders['tma'](batch['tma'].x))
        )  # [batch_size, hidden_dim]
        
        # ═══════════════════════════════════════════
        # IMAGE NODES 
        # ═══════════════════════════════════════════
        x_dict_encoded['lymph'] = self.dropout_layer(
            self.activation(self.leaf_encoders['lymph'](batch['lymph'].x))
        )  # [total_num_lymph_images_in_batch, hidden_dim]
        
        x_dict_encoded['tumor'] = self.dropout_layer(
            self.activation(self.leaf_encoders['tumor'](batch['tumor'].x))
        )  # [total_num_tumor_images_in_batch, hidden_dim]
        
        # ═══════════════════════════════════════════
        # TEXT NODES 
        # ═══════════════════════════════════════════
        # History
        history_embeddings = self._encode_text_nodes(
            batch['history'].input_ids, 
            batch['history'].attention_mask
        )  # [batch_size, 768]
        x_dict_encoded['history'] = self.dropout_layer(
            self.activation(self.leaf_encoders['history'](history_embeddings))
        )  # [batch_size, hidden_dim]
        
        # Surgery report
        surgery_report_embeddings = self._encode_text_nodes(
            batch['surgery_report'].input_ids,
            batch['surgery_report'].attention_mask
        )  # [batch_size, 768]
        x_dict_encoded['surgery_report'] = self.dropout_layer(
            self.activation(self.leaf_encoders['surgery_report'](surgery_report_embeddings))
        )  # [batch_size, hidden_dim]
        
        # Surgery description
        surgery_desc_embeddings = self._encode_text_nodes(
            batch['surgery_desc'].input_ids,
            batch['surgery_desc'].attention_mask
        )  # [batch_size, 768]
        x_dict_encoded['surgery_desc'] = self.dropout_layer(
            self.activation(self.leaf_encoders['surgery_desc'](surgery_desc_embeddings))
        )  # [batch_size, hidden_dim]
        
        # ═══════════════════════════════════════════
        # STEP NODES (Heterogeneous - each step is distinct)
        # ═══════════════════════════════════════════
        # Initialize with zeros
        # Each patient has 1 node per step type
        
        x_dict_encoded['step1'] = torch.zeros(batch_size, self.hidden_dim, device=device)
        x_dict_encoded['step2'] = torch.zeros(batch_size, self.hidden_dim, device=device)
        x_dict_encoded['step3'] = torch.zeros(batch_size, self.hidden_dim, device=device)
        x_dict_encoded['step4'] = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # ═══════════════════════════════════════════
        # MASTER NODE
        # ═══════════════════════════════════════════
        # Pool image nodes per patient if needed
        # Check if lymph has multiple nodes per patient
        if x_dict_encoded['lymph'].size(0) != batch_size:
            lymph_pooled = scatter(
                x_dict_encoded['lymph'], 
                batch['lymph'].batch, 
                dim=0, 
                reduce='mean'
            )  # [batch_size, hidden_dim]
        else:
            lymph_pooled = x_dict_encoded['lymph']
        
        if x_dict_encoded['tumor'].size(0) != batch_size:
            tumor_pooled = scatter(
                x_dict_encoded['tumor'], 
                batch['tumor'].batch, 
                dim=0, 
                reduce='mean'
            )  # [batch_size, hidden_dim]
        else:
            tumor_pooled = x_dict_encoded['tumor']
        
        leaf_encoded = [
            x_dict_encoded['clinical'],
            x_dict_encoded['blood'],
            x_dict_encoded['pathological'],
            x_dict_encoded['tma'],
            lymph_pooled,
            tumor_pooled,
            x_dict_encoded['history'],
            x_dict_encoded['surgery_report'],
            x_dict_encoded['surgery_desc'],
        ]
        # mean pool: [9, batch_size, hidden_dim] -> [batch_size, hidden_dim]
        x_dict_encoded['master'] = torch.stack(leaf_encoded, dim=0).mean(dim=0)
        
        # ============================================
        # Layer 1: Leaves -> Steps
        # ============================================
        x_dict_1 = self.conv1(x_dict_encoded, edge_index_dict)
        x_dict_1 = {
            key: self.dropout_layer(self.activation(x)) 
            for key, x in x_dict_1.items()
        }
        
        # ============================================
        # Layer 2
        # ============================================
        
        # Pre-LayerNorm before conv2
        x_dict_1_norm = {}
        for key in x_dict_1.keys():
            if key in self.layer_norms_conv2:
                x_dict_1_norm[key] = self.layer_norms_conv2[key](x_dict_1[key])
            else:
                x_dict_1_norm[key] = x_dict_1[key]
        
        # Conv2: Temporal + Skip Connections + Self-loops
        x_dict_2_raw = self.conv2(x_dict_1_norm, edge_index_dict)
        
        # Residual connection
        x_dict_2 = {}
        for key in x_dict_2_raw.keys():
            if key in x_dict_1:  # Only add residual for step nodes
                x_dict_2[key] = x_dict_1[key] + x_dict_2_raw[key]
            else:
                x_dict_2[key] = x_dict_2_raw[key]
        
        x_dict_2 = {
            key: self.dropout_layer(self.activation(x)) 
            for key, x in x_dict_2.items()
        }
        
        # Preserve master for next layer (it wasn't updated in conv2)
        x_dict_2['master'] = x_dict_encoded['master']
        
        # ============================================
        # Layer 3: Steps -> Master
        # ============================================
        x_dict_3 = self.conv3(x_dict_2, edge_index_dict)
        
        # ============================================
        # Extract master node representation
        # ============================================
        master_repr = x_dict_3['master']  # [batch_size, hidden_dim]
        
        # ============================================
        # Survival prediction via hazard head
        # ============================================
        logits = self.hazard_head(master_repr)  # [batch_size, num_bins]
        
        return logits
    
    def __str__(self):
        # Extract hidden dims from hazard_head
        hidden_dims_info = []
        for i, layer in enumerate(self.hazard_head):
            if isinstance(layer, nn.Linear):
                hidden_dims_info.append(f"{layer.in_features} -> {layer.out_features}")
        
        return (
            f"\n--- H2DGSurvLogisticHazard ---\n"
            f"  - Input dims: {self.input_dims}\n"
            f"  - GNN hidden dim: {self.hidden_dim}\n"
            f"  - Num heads: {self.num_heads}\n"
            f"  - Dropout: {self.dropout}\n"
            f"  - Num bins: {self.num_bins}\n"
            f"  - Leaf types (9 distinct): clinical, blood, pathological, tma, lymph, tumor, history, surgery_report, surgery_desc\n"
            f"  - Text model: {self.text_model_path}\n"
            f"  - GNN architecture:\n"
            f"      * Layer 1: Fully Heterogeneous Leaves -> Temporal Steps\n"
            f"         - clinical, blood -> step1\n"
            f"         - pathological, tma, lymph, tumor -> step2\n"
            f"         - history -> step3\n"
            f"         - surgery_report, surgery_desc -> step4\n"
            f"      * Layer 2: LayerNorm + Conv + Residual\n"
            f"         - Pre-LayerNorm + GATv2Conv (Temporal + Skip + Self)\n"
            f"         - Temporal: step1->step2->step3->step4 (shared weights)\n"
            f"         - Skip: step1->step3, step1->step4, step2->step4 (shared weights)\n"
            f"         - Self-loops: all steps (shared weights)\n"
            f"      * Layer 3: All Steps -> Master (final aggregation)\n"
            f"  - Hazard head: {' -> '.join(hidden_dims_info)}\n\n"
        )

