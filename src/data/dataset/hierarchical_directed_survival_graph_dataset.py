import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from torch_geometric.data import HeteroData

from src.data.dataset.dataset import Dataset


class HierarchicalDirectedSurvivalGraphDataset(Dataset):
    """
    PyG-compatible Dataset that creates hierarchical directed survival graphs.
    
    Graph structure follows clinical pathway:
    - Step 1: Diagnostic (clinical + blood)
    - Step 2: Pathology & Imaging (pathological + TMA + lymph + tumor)
    - Step 3: History
    - Step 4: Surgery (surgery report + surgery description)
    
    Inherits everything from Dataset, only overrides __getitem__ to build graphs.
    """
    
    def __init__(self, aggregate_images: bool = True, **kwargs):
        """
        Initialize dataset.
        
        Args:
            aggregate_images: If True, average images into one node per modality (default).
                            If False, create one node per image (GNN learns aggregation).
        """
        super().__init__(**kwargs)
        self.aggregate_images = aggregate_images
    
    def __get_lymphnode__(self, patient_id):
        """
        Get lymphnode images.
        
        Returns:
            - If aggregate_images=True: Tensor of shape [1024] (averaged)
            - If aggregate_images=False: Tensor of shape [num_images, 1024]
        """
        file_path = self.lymph_nodes_paths[patient_id]
        features = self.read_h5(file_path)  # Shape: [num_images, 1024]
        
        if self.aggregate_images:
            # Average all images into one vector
            features = features.mean(0)  # [1024]
            features = torch.from_numpy(features).float()
        else:
            # Keep all images separate
            if features.ndim == 1:
                features = features.reshape(1, -1)
            features = torch.from_numpy(features).float()  # [num_images, 1024]
        
        return features
    
    def __get_primarytumor__(self, patient_id):
        """
        Get primary tumor images.
        
        Returns:
            - If aggregate_images=True: Tensor of shape [1024] (averaged)
            - If aggregate_images=False: Tensor of shape [num_images, 1024]
        """
        list_file_paths = self.primary_tumors_paths[patient_id]
        
        if list_file_paths is not None:
            features = []
            for file_path in self.primary_tumors_paths[patient_id]:
                features.append(self.read_h5(file_path))
            features = np.vstack(features)  # Stack all images: [num_images, 1024]
        else:
            # No tumor images available
            features = np.zeros((1, 1024))
        
        if self.aggregate_images:
            # Average all images into one vector
            features = features.mean(0)  # [1024]
            features = torch.from_numpy(features).float()
        else:
            # Keep all images separate
            features = torch.from_numpy(features).float()  # [num_images, 1024]
        
        return features
    
    def get_input_dims(self) -> Dict[str, int]:
        """
        Override to return correct dimensions for image nodes.        
        """
        if len(self.samples) == 0:
            raise ValueError("Dataset is empty, cannot determine input dimensions")
        
        # Get first patient to determine dimensions
        first_patient_id = self.samples[0]
        
        # Get actual dimensions from processed data
        clinical_dim     = self.__get_clinical__(first_patient_id).shape[0]
        blood_dim        = self.__get_blood__(first_patient_id).shape[0]
        patho_dim        = self.__get_pathological__(first_patient_id).shape[0]
        cdm_dim          = self.__get_tma_cdm__(first_patient_id).shape[0]
        
        # For images: depends on aggregate_images flag
        lymph_features = self.__get_lymphnode__(first_patient_id)
        tumor_features = self.__get_primarytumor__(first_patient_id)
        
        if self.aggregate_images:
            # Averaged: shape is [1024]
            lymphnode_dim = lymph_features.shape[0]
            primarytumor_dim = tumor_features.shape[0]
        else:
            # Multiple nodes: shape is [num_images, 1024], return feature dim
            lymphnode_dim = lymph_features.shape[1] if lymph_features.ndim > 1 else lymph_features.shape[0]
            primarytumor_dim = tumor_features.shape[1] if tumor_features.ndim > 1 else tumor_features.shape[0]

        return {
            "clinical": clinical_dim,
            "blood": blood_dim,
            "pathological": patho_dim,
            "cdm": cdm_dim,
            "lymphnode": lymphnode_dim,
            "primarytumor": primarytumor_dim,
        }
    
    def _create_hetero_graph(
        self,
        patient_id: str,
        clinical: torch.Tensor,
        blood: torch.Tensor,
        patho: torch.Tensor,
        cdm: torch.Tensor,
        h_ids: torch.Tensor,
        h_mask: torch.Tensor,
        s_ids: torch.Tensor,
        s_mask: torch.Tensor,
        r_ids: torch.Tensor,
        r_mask: torch.Tensor,
        lymph: torch.Tensor,
        tumor: torch.Tensor,
    ) -> HeteroData:
        """
        Create hierarchical directed survival graph for one patient.
        
        Graph structure:
        - Master node at the top level
        - Four step nodes representing clinical pathway stages
        - Leaf nodes: clinical, blood, pathological, tma, lymph, tumor, history, surgery_report, surgery_desc
        - Directed edges: leaves -> steps -> temporal sequence -> master
        """
        data = HeteroData()
        
        # ═══════════════════════════════════════════════════════════════
        # LEAF NODES
        # ═══════════════════════════════════════════════════════════════
        
        # Structured leaves
        data['clinical'].x = clinical.unsqueeze(0)           
        data['clinical'].num_nodes = 1
        
        data['blood'].x = blood.unsqueeze(0)                    
        data['blood'].num_nodes = 1
        
        data['pathological'].x = patho.unsqueeze(0)          
        data['pathological'].num_nodes = 1
        
        data['tma'].x = cdm.unsqueeze(0)                     
        data['tma'].num_nodes = 1
        
        # Image leaves (configurable: one node per image OR aggregated)
        if self.aggregate_images:
            # Aggregated: lymph and tumor are [1024], need unsqueeze
            data['lymph'].x = lymph.unsqueeze(0)              # [1, 1024]
            data['lymph'].num_nodes = 1
            
            data['tumor'].x = tumor.unsqueeze(0)              # [1, 1024]
            data['tumor'].num_nodes = 1
        else:
            # Multiple nodes: lymph and tumor are [num_images, 1024]
            data['lymph'].x = lymph                           # [num_lymph_images, 1024]
            data['lymph'].num_nodes = lymph.shape[0]
            
            data['tumor'].x = tumor                           # [num_tumor_images, 1024]
            data['tumor'].num_nodes = tumor.shape[0]
        
        # Text leaves
        max_text_len = 512
        h_ids_padded = F.pad(h_ids.squeeze(0), (0, max_text_len - h_ids.shape[1]))
        s_ids_padded = F.pad(s_ids.squeeze(0), (0, max_text_len - s_ids.shape[1]))
        r_ids_padded = F.pad(r_ids.squeeze(0), (0, max_text_len - r_ids.shape[1]))
        h_mask_padded = F.pad(h_mask.squeeze(0), (0, max_text_len - h_mask.shape[1]))
        s_mask_padded = F.pad(s_mask.squeeze(0), (0, max_text_len - s_mask.shape[1]))
        r_mask_padded = F.pad(r_mask.squeeze(0), (0, max_text_len - r_mask.shape[1]))
        
        data['history'].input_ids = h_ids_padded.unsqueeze(0)
        data['history'].attention_mask = h_mask_padded.unsqueeze(0)
        data['history'].num_nodes = 1
        
        data['surgery_report'].input_ids = s_ids_padded.unsqueeze(0)
        data['surgery_report'].attention_mask = s_mask_padded.unsqueeze(0)
        data['surgery_report'].num_nodes = 1
        
        data['surgery_desc'].input_ids = r_ids_padded.unsqueeze(0)
        data['surgery_desc'].attention_mask = r_mask_padded.unsqueeze(0)
        data['surgery_desc'].num_nodes = 1
        
        # ═══════════════════════════════════════════════════════════════
        # STEP AND MASTER NODES
        # ═══════════════════════════════════════════════════════════════
        data['step1'].x = torch.zeros(1, 1)  # Clinical pathway step 1 (clinical + blood)
        data['step1'].num_nodes = 1
        
        data['step2'].x = torch.zeros(1, 1)  # Clinical pathway step 2 (patho + tma + images)
        data['step2'].num_nodes = 1
        
        data['step3'].x = torch.zeros(1, 1)  # Clinical pathway step 3 (history)
        data['step3'].num_nodes = 1
        
        data['step4'].x = torch.zeros(1, 1)  # Clinical pathway step 4 (surgery reports)
        data['step4'].num_nodes = 1

        data['master'].x = torch.zeros(1, 1)
        data['master'].num_nodes = 1
        
        # ═══════════════════════════════════════════════════════════════
        # EDGES - Layer 1: Leaves -> Steps
        # ═══════════════════════════════════════════════════════════════
        
        # Step 1: clinical + blood
        data['clinical', 'to_step1', 'step1'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['blood', 'to_step1', 'step1'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # Step 2: pathological + tma + images
        data['pathological', 'to_step2', 'step2'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['tma', 'to_step2', 'step2'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # Connect all lymph image nodes to step2 (one edge per image)
        num_lymph = data['lymph'].num_nodes
        lymph_src = torch.arange(num_lymph, dtype=torch.long)  # [0, 1, ..., num_lymph-1]
        lymph_dst = torch.zeros(num_lymph, dtype=torch.long)   # All connect to step2 node 0
        data['lymph', 'to_step2', 'step2'].edge_index = torch.stack([lymph_src, lymph_dst], dim=0)
        
        # Connect all tumor image nodes to step2 (one edge per image)
        num_tumor = data['tumor'].num_nodes
        tumor_src = torch.arange(num_tumor, dtype=torch.long)  # [0, 1, ..., num_tumor-1]
        tumor_dst = torch.zeros(num_tumor, dtype=torch.long)   # All connect to step2 node 0
        data['tumor', 'to_step2', 'step2'].edge_index = torch.stack([tumor_src, tumor_dst], dim=0)
        
        # Step 3: history
        data['history', 'to_step3', 'step3'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # Step 4: surgery reports
        data['surgery_report', 'to_step4', 'step4'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['surgery_desc', 'to_step4', 'step4'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # ═══════════════════════════════════════════════════════════════
        # EDGES - Layer 2: Temporal + Skip Connections + Self-loops
        # ═══════════════════════════════════════════════════════════════
        
        # Temporal sequence: step1 -> step2 -> step3 -> step4
        data['step1', 'temporal_next', 'step2'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['step2', 'temporal_next', 'step3'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['step3', 'temporal_next', 'step4'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # Skip connections
        data['step1', 'skip_2', 'step3'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Skip 1 step
        data['step1', 'skip_3', 'step4'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Skip 2 steps
        data['step2', 'skip_2', 'step4'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Skip 1 step
        
        # Self-loops
        data['step1', 'self', 'step1'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['step2', 'self', 'step2'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['step3', 'self', 'step3'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['step4', 'self', 'step4'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # ═══════════════════════════════════════════════════════════════
        # EDGES - Layer 3: All Steps -> Master
        # ═══════════════════════════════════════════════════════════════
        data['step1', 'to_master', 'master'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['step2', 'to_master', 'master'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['step3', 'to_master', 'master'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data['step4', 'to_master', 'master'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)        
        data['master', 'self', 'master'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # ═══════════════════════════════════════════════════════════════
        # PATIENT METADATA AND TARGETS
        # ═══════════════════════════════════════════════════════════════
        data.patient_id = torch.tensor(int(patient_id), dtype=torch.long)
        survival_status = self.data_clinical.loc[self.data_clinical.patient_id == patient_id].survival_status.values[0]
        data.event = torch.tensor(survival_status == "deceased", dtype=torch.long)
        days_to_last_information = self.data_clinical.loc[self.data_clinical.patient_id == patient_id].days_to_last_information.values[0]
        data.time = torch.tensor(days_to_last_information, dtype=torch.float32)
        
        return data
    
    def __getitem__(self, index: int) -> HeteroData:
        """Get hierarchical directed survival graph for patient at index."""
        patient_id = self.samples[index]
        
        # Extract all features
        clinical = self.__get_clinical__(patient_id)
        blood = self.__get_blood__(patient_id)
        patho = self.__get_pathological__(patient_id)
        cdm = self.__get_tma_cdm__(patient_id)
        
        h_ids, h_mask = self.__tokenize__(self.histories[patient_id], self.max_tokens_history)
        s_ids, s_mask = self.__tokenize__(self.surgeries[patient_id], self.max_tokens_surgery)
        r_ids, r_mask = self.__tokenize__(self.reports[patient_id], self.max_tokens_report)
        
        lymph = self.__get_lymphnode__(patient_id)
        tumor = self.__get_primarytumor__(patient_id)
        
        # Create and return hierarchical directed survival graph
        return self._create_hetero_graph(
            patient_id=patient_id,
            clinical=clinical, blood=blood, patho=patho, cdm=cdm,
            h_ids=h_ids, h_mask=h_mask,
            s_ids=s_ids, s_mask=s_mask,
            r_ids=r_ids, r_mask=r_mask,
            lymph=lymph, tumor=tumor,
        )

